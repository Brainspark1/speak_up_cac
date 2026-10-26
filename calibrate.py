"""Calibrate Whack-a-Word thresholds and mole vocabulary.

Runs (phrase, expected_mole) pairs through GAMEBERT extract_entities and RankedMapper,
then prints the TARGET words, top-3 scores, and pass/fail at the configured
thresholds. Also flags mole pairs whose descriptions overlap too much.

    .venv/bin/python calibrate.py                       # phrases from game_config.json
    .venv/bin/python calibrate.py --visible 4           # rank against 3 random distractors + expected
    .venv/bin/python calibrate.py --phrase "whack the cook=chef"
    .venv/bin/python calibrate.py --overlap-only

No microphone or GPU is needed. Only the NER model is loaded.
"""

import argparse
import itertools
import json
import logging
import random
import statistics
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import DEFAULT_CONFIG_PATH, build_mapping, load_config
from backend.entities import tokens
from backend.ranked_mapper import RankedMapper
from backend.resolver import Resolver


def load_phrases(args, config):
    pairs = [tuple(p) for p in config.get("calibration_phrases", [])]
    if args.phrases:
        with open(args.phrases) as f:
            pairs = [tuple(p) for p in json.load(f)]
    for p in args.phrase or []:
        text, _, mole = p.rpartition("=")
        pairs.append((text, mole))
    return pairs


def overlap_report(config, warn):
    moles = config["moles"]
    ids = list(moles)
    descs = [moles[i]["description"] for i in ids]
    tfidf = TfidfVectorizer().fit_transform(descs)
    sims = cosine_similarity(tfidf)
    twins = {frozenset(p) for p in config.get("twins", [])}
    rows = []
    for a, b in itertools.combinations(range(len(ids)), 2):
        shared = sorted(set(tokens(descs[a])) & set(tokens(descs[b])))
        if sims[a, b] >= warn or shared:
            rows.append((sims[a, b], ids[a], ids[b], shared, frozenset((ids[a], ids[b])) in twins))
    rows.sort(reverse=True)
    print(f"\n=== Description overlap (cosine >= {warn} or any shared word) ===")
    if not rows:
        print("  none")
    for sim, a, b, shared, is_twin in rows:
        flag = "TWIN (intended)" if is_twin else ("OVERLAP" if sim >= warn else "shared")
        print(f"  {flag:16s} {sim:.3f}  {a:14s} ~ {b:14s} shared: {', '.join(shared)}")

    within = [(i, w) for i in ids for w in set(tokens(moles[i]["description"]))
              if tokens(moles[i]["description"]).count(w) > 1]
    for i, w in within:
        print(f"  DUPLICATE word '{w}' inside {i}")
    short = [(i, len(tokens(moles[i]["description"]))) for i in ids
             if not 4 <= len(tokens(moles[i]["description"])) <= 12]
    for i, n in short:
        print(f"  LENGTH {i}: {n} words (aim for 6-12)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--phrases", help="JSON file of [[phrase, expected_mole], ...]")
    ap.add_argument("--phrase", action="append", help='"phrase=expected_mole" (repeatable)')
    ap.add_argument("--visible", type=int, default=0,
                    help="rank against N visible moles (expected + random distractors). 0 = all moles")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--overlap-warn", type=float, default=0.20)
    ap.add_argument("--overlap-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    t = config["thresholds"]
    print(f"Thresholds: MIN_SCORE={t['MIN_SCORE']} AMBIGUITY_MARGIN={t['AMBIGUITY_MARGIN']} "
          f"FALLBACK_MIN_SCORE={t['FALLBACK_MIN_SCORE']} FALLBACK_MARGIN={t['FALLBACK_MARGIN']} "
          f"tiers={t['tiers']}")

    if not args.overlap_only:
        from backend.voice import MoleVoiceController

        print(f"Loading {config['voice']['ner_model']} ...", flush=True)
        ner = MoleVoiceController.text_only(config["voice"]["ner_model"], device_backend=args.device,
                                            ner_aggregation=config["voice"].get("ner_aggregation", "first"))
        print(f"id2label: {ner.id2label}")
        resolver = Resolver(config, RankedMapper(json_data=build_mapping(config)))
        rng = random.Random(args.seed)
        all_ids = list(config["moles"])

        passed, scores = 0, {"target": [], "fallback": []}
        failures = []
        pairs = load_phrases(args, config)
        print(f"\n=== {len(pairs)} phrases, ranked against "
              f"{'ALL ' + str(len(all_ids)) if not args.visible else args.visible} moles ===")
        for phrase, expected in pairs:
            if expected not in config["moles"]:
                print(f"  SKIP unknown mole {expected!r}")
                continue
            if args.visible:
                others = rng.sample([i for i in all_ids if i != expected], args.visible - 1)
                visible = [expected] + others
            else:
                visible = all_ids
            text = phrase.lower()
            entities = ner.extract_entities(text)
            r = resolver.resolve(text, entities, [{"key": i, "id": i} for i in visible])
            got = r.get("mole", {}).get("id")
            ok = r["outcome"] == "whack" and got == expected
            passed += ok
            rank = {d["id"]: d["score"] for d in r.get("top", [])}
            exp_score = resolver.mapper.rank_targets(r.get("query") or text, visible)
            exp_score = dict(exp_score).get(expected, 0.0)
            if ok:
                scores[r["mode"]].append(r["score"])
            else:
                failures.append((phrase, expected, r))
            top = "  ".join(f"{d['id']}={d['score']:.3f}" for d in r.get("top", []))
            print(f"  {'PASS' if ok else 'FAIL'} {phrase!r:38s} -> {r['outcome']:15s} "
                  f"[{r.get('mode', '-')}] TARGET={r['target_words']} "
                  f"tier={r.get('tier', '-')} action={r.get('action', '-')}"
                  f"\n         expected {expected} ({exp_score:.3f})  top3: {top}")

        print(f"\n=== {passed}/{len(pairs)} passed ===")
        for mode, s in scores.items():
            if s:
                print(f"  correct-match scores [{mode}]: n={len(s)} min={min(s):.3f} "
                      f"median={statistics.median(s):.3f} max={max(s):.3f}")
        if failures:
            print("  failures:")
            for phrase, expected, r in failures:
                print(f"    {phrase!r}: expected {expected}, got {r['outcome']} "
                      f"{r.get('mole', {}).get('id', '')} (top1={r.get('top1', 0):.3f} top2={r.get('top2', 0):.3f})")

    overlap_report(config, args.overlap_warn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
