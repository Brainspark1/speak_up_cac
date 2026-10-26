"""Match resolution: transcript + GAMEBERT entities + visible moles -> outcome.

Pure logic. No audio, no model, no clock. Everything is driven by
game_config.json["thresholds"], so tests can pass mocked entities and a
mocked mapper.
"""

from .entities import group_words, tokens

TIER_ORDER = ["OK", "Great", "Perfect"]


class Resolver:
    def __init__(self, config, mapper):
        self.config = config
        self.mapper = mapper
        self.t = config["thresholds"]
        self.moles = config["moles"]
        self.golden_words = config.get("golden", {}).get("extra_description", "")
        self.query_stopwords = set(self.t.get("QUERY_STOPWORDS", []))
        self.correction_keywords = set(self.t.get("CORRECTION_KEYWORDS", []))
        self.correction_ignore = set(self.t.get("CORRECTION_IGNORE_WORDS", []))

    # ---------- per-round mapper state ----------

    def description_for(self, instance):
        desc = self.moles[instance["id"]]["description"]
        if instance.get("golden") and self.golden_words:
            desc = f"{desc} {self.golden_words}"
        return desc

    def set_visible(self, instances):
        """Restrict the mapper's targets to the moles on screen right now."""
        self.mapper.json_data["targets"] = {
            inst["id"]: {**self.moles[inst["id"]], "description": self.description_for(inst)}
            for inst in instances
            if inst["id"] in self.moles and inst["id"] != "enemy"
        }

    # ---------- pieces ----------

    def detect_correction(self, transcript, entities, target_words):
        words = [w for w in group_words(transcript, entities, "CORRECTION")
                 if w not in self.correction_ignore]
        if words:
            return True, words
        # GAMEBERT tags a bare "stop" as ACTION, so we also accept an utterance
        # made entirely of correction keywords when it names no target.
        toks = tokens(transcript)
        if toks and not target_words and set(toks) <= self.correction_keywords:
            return True, toks
        return False, []

    def tier_for(self, score, mode):
        tiers = self.t["tiers"] if mode == "target" else self.t.get("fallback_tiers", self.t["tiers"])
        if score >= tiers["Perfect"]:
            return "Perfect"
        if score >= tiers["Great"]:
            return "Great"
        return "OK"

    def resolve_action(self, action_words, transcript):
        query = " ".join(action_words) if action_words else transcript
        ranked = self.mapper.rank_actions(query)
        if ranked and ranked[0][1] >= self.t.get("ACTION_MIN_SCORE", 0.0) and ranked[0][1] > 0:
            return ranked[0][0], ranked[0][1]
        return self.t.get("DEFAULT_ACTION", "whack"), 0.0

    def verdict_for(self, instance, action, tier, mission_tag):
        mole = self.moles[instance["id"]]
        if mole.get("family") == "bomb" or instance.get("bomb"):
            return "defused" if action == "defuse" else "bomb"
        if instance.get("golden"):
            need = self.config["scoring"].get("golden_min_tier", "Great")
            if TIER_ORDER.index(tier) < TIER_ORDER.index(need):
                return "golden_resist"
        if mission_tag and mission_tag not in mole.get("tags", []):
            return "wrong_mission"
        return "hit"

    def sign_word_only(self, instance, query):
        """True when the only description word the player said was the sign's own word."""
        sign = self.moles[instance["id"]].get("sign_word")
        if not sign:
            return False
        desc_words = set(tokens(self.description_for(instance)))
        said = set(tokens(query)) & desc_words
        return said == {sign}

    # ---------- main entry ----------

    def resolve(self, transcript, entities, instances, mission_tag=None):
        transcript = (transcript or "").strip().lower()
        entities = entities or []
        target_words = group_words(transcript, entities, "TARGET")
        action_words = group_words(transcript, entities, "ACTION")
        correction, correction_words = self.detect_correction(transcript, entities, target_words)

        result = {
            "transcript": transcript,
            "target_words": target_words,
            "action_words": action_words,
            "correction": correction,
            "correction_words": correction_words,
        }

        if correction and not target_words:
            result["outcome"] = "correction_only"
            return result

        query_words = [w for w in target_words if w not in self.query_stopwords]
        if query_words:
            mode, query = "target", " ".join(query_words)
            min_score, margin = self.t["MIN_SCORE"], self.t["AMBIGUITY_MARGIN"]
        else:
            mode, query = "fallback", transcript
            min_score, margin = self.t["FALLBACK_MIN_SCORE"], self.t["FALLBACK_MARGIN"]
        result.update(mode=mode, query=query)

        by_id = {inst["id"]: inst for inst in instances if inst["id"] in self.moles}
        self.set_visible(list(by_id.values()))
        ranked = self.mapper.rank_targets(query, list(by_id.keys())) if query else []
        result["top"] = [{"id": i, "key": by_id[i].get("key"), "score": round(s, 4)} for i, s in ranked[:3]]

        top1 = ranked[0][1] if ranked else 0.0
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        result.update(top1=top1, top2=top2)

        # zero-score guard: SemanticMapper returns its first target at 0.0 when nothing matches
        if not ranked or top1 <= 0.0 or top1 < min_score:
            result["outcome"] = "no_match"
            return result
        if top1 - top2 < margin:
            result["outcome"] = "ambiguous"
            result["candidates"] = [by_id[ranked[0][0]].get("key"), by_id[ranked[1][0]].get("key")]
            result["candidate_ids"] = [ranked[0][0], ranked[1][0]]
            return result

        inst = by_id[ranked[0][0]]
        tier = self.tier_for(top1, mode)
        action, action_score = self.resolve_action(action_words, transcript)
        result.update(
            outcome="whack",
            mole={"key": inst.get("key"), "id": inst["id"]},
            score=round(top1, 4),
            tier=tier,
            action=action,
            action_score=round(action_score, 4),
            verdict=self.verdict_for(inst, action, tier, mission_tag),
            sign_word_only=self.sign_word_only(inst, query),
        )
        return result
