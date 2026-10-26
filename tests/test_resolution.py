"""Resolution logic tests. Mocked entities and scores only: no mic, model, or GPU."""

import copy

import pytest

from backend.arbiter import WhackArbiter
from backend.config import build_mapping, load_config, validate_config
from backend.entities import fix_mishearings, group_words
from backend.ranked_mapper import RankedMapper
from backend.resolver import Resolver
from backend.tracker import MoleTracker


@pytest.fixture(scope="module")
def config():
    return load_config()


class FakeMapper:
    """Returns preset scores so threshold/margin tests don't depend on TF-IDF."""

    def __init__(self, target_scores, action_scores=None):
        self.target_scores = target_scores
        self.action_scores = action_scores or {"whack": 0.5, "defuse": 0.0}
        self.json_data = {"targets": {}, "actions": {}}
        self.queries = []

    def rank_targets(self, text, target_ids):
        self.queries.append(text)
        ranked = [(i, self.target_scores.get(i, 0.0)) for i in target_ids]
        return sorted(ranked, key=lambda p: p[1], reverse=True)

    def rank_actions(self, text):
        return sorted(self.action_scores.items(), key=lambda p: p[1], reverse=True)


def ent(text, word, group, score=0.95):
    start = text.index(word)
    return {"entity_group": group, "word": word, "score": score, "start": start, "end": start + len(word)}


def moles(*ids, **flags):
    return [{"key": f"k_{i}", "id": i, **flags.get(i, {})} for i in ids]


# ---------- entity helpers ----------

def test_group_words_rebuilds_subword_fragments():
    # Real GAMEBERT output for "whack the cook": "whack" comes back as "##ha" + "##ck"
    text = "whack the cook"
    entities = [
        {"entity_group": "ACTION", "word": "##ha", "score": 0.69, "start": 1, "end": 3},
        {"entity_group": "ACTION", "word": "##ck", "score": 0.99, "start": 3, "end": 5},
        {"entity_group": "TARGET", "word": "cook", "score": 0.98, "start": 10, "end": 14},
    ]
    assert group_words(text, entities, "ACTION") == ["whack"]
    assert group_words(text, entities, "TARGET") == ["cook"]


def test_group_words_joins_split_word():
    text = "bonk the spaceman"
    entities = [
        {"entity_group": "TARGET", "word": "space", "start": 9, "end": 14},
        {"entity_group": "TARGET", "word": "##man", "start": 14, "end": 17},
    ]
    assert group_words(text, entities, "TARGET") == ["spaceman"]


def test_game_transcript_fixes_cover_knight_and_firefighter(config):
    fixes = config["voice"]["transcript_fixes"]
    text, changed = fix_mishearings("whack the night, fireman", fixes)
    assert text == "whack the knight, firefighter"
    assert changed == [("night", "knight"), ("fireman", "firefighter")]
    assert "knight" in config["voice"]["initial_prompt"]
    assert "firefighter" in config["voice"]["initial_prompt"]


# ---------- thresholds and margin ----------

def test_whack_above_min_score_with_clear_margin(config):
    r = Resolver(config, FakeMapper({"chef": 0.40, "pirate": 0.0}))
    text = "whack the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], moles("chef", "pirate"))
    assert out["outcome"] == "whack"
    assert out["mole"] == {"key": "k_chef", "id": "chef"}
    assert out["mode"] == "target"
    assert out["query"] == "cook"
    assert out["tier"] == "Great"
    assert out["verdict"] == "hit"


def test_lower_description_score_can_earn_great_tier(config):
    r = Resolver(config, FakeMapper({"chef": 0.30}))
    assert r.tier_for(0.30, "target") == "Great"
    assert r.tier_for(0.18, "fallback") == "Great"


def test_below_min_score_is_no_match(config):
    min_score = config["thresholds"]["MIN_SCORE"]
    r = Resolver(config, FakeMapper({"chef": min_score - 0.01}))
    text = "whack the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], moles("chef", "pirate"))
    assert out["outcome"] == "no_match"


def test_within_margin_is_ambiguous(config):
    margin = config["thresholds"]["AMBIGUITY_MARGIN"]
    r = Resolver(config, FakeMapper({"chef": 0.35, "baker": 0.35 - margin + 0.01}))
    text = "hit the kitchen"
    out = r.resolve(text, [ent(text, "kitchen", "TARGET")], moles("chef", "baker"))
    assert out["outcome"] == "ambiguous"
    assert set(out["candidates"]) == {"k_chef", "k_baker"}


def test_exactly_at_margin_is_not_ambiguous(config):
    margin = config["thresholds"]["AMBIGUITY_MARGIN"]
    r = Resolver(config, FakeMapper({"chef": 0.35, "baker": 0.35 - margin - 1e-9}))
    text = "hit the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], moles("chef", "baker"))
    assert out["outcome"] == "whack"


def test_zero_score_guard_rejects_first_target(config):
    # SemanticMapper returns its first target with 0.0 when nothing matches. Never whack it.
    cfg = copy.deepcopy(config)
    cfg["thresholds"]["MIN_SCORE"] = 0.0
    cfg["thresholds"]["FALLBACK_MIN_SCORE"] = 0.0
    r = Resolver(cfg, FakeMapper({}))
    out = r.resolve("banana", [], moles("chef", "pirate"))
    assert out["outcome"] == "no_match"


def test_no_visible_moles_is_no_match(config):
    r = Resolver(config, FakeMapper({"chef": 0.9}))
    text = "whack the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], [])
    assert out["outcome"] == "no_match"


def test_fallback_uses_full_transcript_and_lower_thresholds(config):
    t = config["thresholds"]
    score = (t["FALLBACK_MIN_SCORE"] + t["MIN_SCORE"]) / 2  # passes fallback, would fail target mode
    mapper = FakeMapper({"angry": score})
    r = Resolver(config, mapper)
    text = "smash the angry guy"
    out = r.resolve(text, [ent(text, "smash", "ACTION")], moles("angry", "happy"))
    assert out["mode"] == "fallback"
    assert mapper.queries[-1] == text
    assert out["outcome"] == "whack"


def test_fallback_margin_is_used_in_fallback_mode(config):
    t = config["thresholds"]
    r = Resolver(config, FakeMapper({"angry": 0.15, "happy": 0.15 - t["FALLBACK_MARGIN"] + 0.005}))
    out = r.resolve("smash the angry guy", [], moles("angry", "happy"))
    assert out["outcome"] == "ambiguous"


def test_query_stopwords_removed_from_target_words(config):
    mapper = FakeMapper({"sad": 0.4})
    r = Resolver(config, mapper)
    text = "whack the crying mole"
    out = r.resolve(text, [ent(text, "crying", "TARGET"), ent(text, "mole", "TARGET")], moles("sad", "happy"))
    assert mapper.queries[-1] == "crying"
    assert out["outcome"] == "whack"


def test_only_visible_moles_are_ranked(config):
    mapper = FakeMapper({"chef": 0.9, "pirate": 0.3})
    r = Resolver(config, mapper)
    text = "hit the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], moles("pirate", "wizard"))
    assert out["mole"]["id"] == "pirate"
    assert set(mapper.json_data["targets"]) == {"pirate", "wizard"}


# ---------- actions, bombs, golden, sign words ----------

def test_defuse_on_bomb_is_defused(config):
    r = Resolver(config, FakeMapper({"bomb": 0.4}, {"whack": 0.0, "defuse": 0.6}))
    text = "defuse the bomb"
    out = r.resolve(text, [ent(text, "defuse", "ACTION"), ent(text, "bomb", "TARGET")], moles("bomb", "chef"))
    assert out["action"] == "defuse"
    assert out["verdict"] == "defused"


def test_whack_on_bomb_is_bomb(config):
    r = Resolver(config, FakeMapper({"bomb": 0.4}, {"whack": 0.6, "defuse": 0.0}))
    text = "whack the bomb"
    out = r.resolve(text, [ent(text, "whack", "ACTION"), ent(text, "bomb", "TARGET")], moles("bomb", "chef"))
    assert out["verdict"] == "bomb"


def test_weak_action_falls_back_to_default(config):
    r = Resolver(config, FakeMapper({"chef": 0.4}, {"whack": 0.0, "defuse": 0.0}))
    text = "the cook"
    out = r.resolve(text, [ent(text, "cook", "TARGET")], moles("chef"))
    assert out["action"] == config["thresholds"]["DEFAULT_ACTION"]


def test_golden_needs_great_tier(config):
    t = config["thresholds"]["tiers"]
    golden = {"chef": {"golden": True}}
    text = "hit the cook"
    weak = Resolver(config, FakeMapper({"chef": t["Great"] - 0.01})).resolve(
        text, [ent(text, "cook", "TARGET")], moles("chef", **golden))
    strong = Resolver(config, FakeMapper({"chef": t["Great"]})).resolve(
        text, [ent(text, "cook", "TARGET")], moles("chef", **golden))
    assert weak["verdict"] == "golden_resist"
    assert strong["verdict"] == "hit"


def test_sign_word_only_flag(config):
    mapper = RankedMapper(json_data=build_mapping(config))
    r = Resolver(config, mapper)
    t1 = "whack the ocean"
    literal = r.resolve(t1, [ent(t1, "ocean", "TARGET")], moles("sign_ocean", "chef"))
    t2 = "whack the waves"
    creative = r.resolve(t2, [ent(t2, "waves", "TARGET")], moles("sign_ocean", "chef"))
    assert literal["outcome"] == creative["outcome"] == "whack"
    assert literal["sign_word_only"] is True
    assert creative["sign_word_only"] is False


# ---------- mission card tags ----------

def test_mission_tag_match_is_hit(config):
    r = Resolver(config, FakeMapper({"fish": 0.4}))
    text = "whack the fish"
    out = r.resolve(text, [ent(text, "fish", "TARGET")], moles("fish", "chef"), mission_tag="water")
    assert out["verdict"] == "hit"


def test_mission_tag_violation_is_wrong_mission(config):
    r = Resolver(config, FakeMapper({"astronaut": 0.4}))
    text = "whack the spaceman"
    out = r.resolve(text, [ent(text, "spaceman", "TARGET")], moles("astronaut", "fish"), mission_tag="water")
    assert out["outcome"] == "whack"
    assert out["verdict"] == "wrong_mission"


def test_mission_is_tag_check_not_similarity(config):
    # "water" appears in the ocean sign's description, but the umbrella is tagged water and the ocean
    # sign's similarity is irrelevant: the rule only looks at the whacked mole's tags.
    r = Resolver(config, FakeMapper({"umbrella": 0.4}))
    text = "hit the umbrella"
    out = r.resolve(text, [ent(text, "umbrella", "TARGET")], moles("umbrella", "sign_ocean"), mission_tag="water")
    assert out["verdict"] == "hit"


# ---------- correction detection ----------

def test_correction_entity_with_target_continues(config):
    r = Resolver(config, FakeMapper({"pirate": 0.4}))
    text = "no wait the pirate"
    out = r.resolve(text, [ent(text, "wait", "CORRECTION"), ent(text, "pirate", "TARGET")], moles("pirate", "chef"))
    assert out["correction"] is True
    assert out["outcome"] == "whack"
    assert out["mole"]["id"] == "pirate"


def test_correction_only(config):
    r = Resolver(config, FakeMapper({}))
    text = "not that one"
    out = r.resolve(text, [ent(text, "not that one", "CORRECTION")], moles("pirate"))
    assert out["correction"] is True
    assert out["outcome"] == "correction_only"


def test_bare_stop_tagged_action_is_correction(config):
    # GAMEBERT tags a bare "stop" as ACTION; the keyword fallback treats it as a correction.
    r = Resolver(config, FakeMapper({}))
    out = r.resolve("stop", [ent("stop", "stop", "ACTION")], moles("pirate"))
    assert out["outcome"] == "correction_only"


def test_filler_correction_span_is_ignored(config):
    # GAMEBERT tags "one" in "get the bread one" as CORRECTION; that must not cancel anything.
    r = Resolver(config, FakeMapper({"baker": 0.4}))
    text = "get the bread one"
    out = r.resolve(text, [ent(text, "get", "ACTION"), ent(text, "bread", "TARGET"),
                           ent(text, "one", "CORRECTION")], moles("baker", "chef"))
    assert out["correction"] is False
    assert out["outcome"] == "whack"


# ---------- wind-up and regret window ----------

@pytest.fixture
def arbiter():
    return WhackArbiter(windup_ms=500, regret_window_ms=300, refund_fraction=0.5)


def test_correction_during_windup_cancels(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "wrong_mission"}, now_ms=1000)
    effect = arbiter.correction(spoken_at_ms=1200, now_ms=1300)
    assert effect["effect"] == "cancelled"
    assert arbiter.strike_due(pending["id"], now_ms=1500) is None


def test_no_correction_strikes(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "hit"}, now_ms=1000)
    assert arbiter.strike_due(pending["id"], now_ms=1499) is None
    assert arbiter.strike_due(pending["id"], now_ms=1500)["id"] == pending["id"]


def test_late_correction_spoken_during_windup_refunds_half(arbiter):
    # Spoken during the wind-up but transcribed after the strike: regret refund.
    pending, _ = arbiter.begin_windup({"verdict": "wrong_mission"}, now_ms=1000)
    arbiter.strike_due(pending["id"], now_ms=1500)
    effect = arbiter.correction(spoken_at_ms=1400, now_ms=2600)
    assert effect["effect"] == "refund"
    assert effect["refund_fraction"] == 0.5


def test_correction_within_regret_window_refunds(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "wrong_mission"}, now_ms=1000)
    arbiter.strike_due(pending["id"], now_ms=1500)
    assert arbiter.correction(spoken_at_ms=1500 + 300, now_ms=3000)["effect"] == "refund"


def test_correction_after_regret_window_does_nothing(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "wrong_mission"}, now_ms=1000)
    arbiter.strike_due(pending["id"], now_ms=1500)
    assert arbiter.correction(spoken_at_ms=1500 + 301, now_ms=3000)["effect"] == "none"


def test_refund_only_once(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "wrong_mission"}, now_ms=1000)
    arbiter.strike_due(pending["id"], now_ms=1500)
    assert arbiter.correction(1600, 2000)["effect"] == "refund"
    assert arbiter.correction(1650, 2100)["effect"] == "none"


def test_regret_on_correct_hit_refunds_nothing(arbiter):
    pending, _ = arbiter.begin_windup({"verdict": "hit"}, now_ms=1000)
    arbiter.strike_due(pending["id"], now_ms=1500)
    effect = arbiter.correction(1600, 2000)
    assert effect["effect"] == "refund" and effect["refund_fraction"] == 0.0


def test_new_whack_supersedes_pending(arbiter):
    first, _ = arbiter.begin_windup({"mole": "a"}, now_ms=1000)
    second, superseded = arbiter.begin_windup({"mole": "b"}, now_ms=1200)
    assert superseded["id"] == first["id"]
    assert arbiter.strike_due(first["id"], now_ms=1600) is None
    assert arbiter.strike_due(second["id"], now_ms=1700)["whack"] == {"mole": "b"}


# ---------- tracker: utterance start vs mole window ----------

def test_mole_that_retreated_after_speech_started_still_counts():
    tr = MoleTracker(start_tolerance_ms=0)
    tr.update([{"key": "a", "id": "chef"}], now_ms=1000)
    tr.update([], now_ms=3000)  # retreats while Whisper is still busy
    assert [m["id"] for m in tr.candidates(spoken_at_ms=2500)] == ["chef"]
    assert tr.candidates(spoken_at_ms=3100) == []


def test_mole_that_spawned_after_speech_started_does_not_count():
    tr = MoleTracker(start_tolerance_ms=0)
    tr.update([{"key": "a", "id": "chef"}], now_ms=2000)
    assert tr.candidates(spoken_at_ms=1500) == []


def test_shapeshift_drops_old_identity_immediately():
    tr = MoleTracker(start_tolerance_ms=0)
    tr.update([{"key": "a", "id": "chef"}], now_ms=1000)
    tr.update([{"key": "a2", "id": "pirate", "shift_of": "a"}], now_ms=1500)
    assert [m["id"] for m in tr.candidates(spoken_at_ms=1200)] == []
    assert [m["id"] for m in tr.candidates(spoken_at_ms=1600)] == ["pirate"]


# ---------- config ----------

def test_enemy_id_rejected(config):
    cfg = copy.deepcopy(config)
    cfg["moles"]["enemy"] = {"description": "enemy bad"}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_ranked_mapper_matches_package_top1(config):
    """rank_targets()[0] must equal SemanticMapper.find_max_target_similarity on the same targets."""
    mapper = RankedMapper(json_data=build_mapping(config))
    for text in ["cook", "parrot", "the yellow fruit", "kitchen apron", "lava"]:
        ranked = mapper.rank_targets(text)
        name, score = mapper.find_max_target_similarity(text)
        assert ranked[0][1] == pytest.approx(float(score))
        assert ranked[0][0] == name or ranked[0][1] == pytest.approx(ranked[1][1])


# ---------- Whisper mishearing fixes ----------

def test_more_is_fixed_to_mole(config):
    from backend.entities import fix_mishearings
    v = config["voice"]
    fixes, skip = v["transcript_fixes"], v["transcript_fix_skip_after"]
    assert fix_mishearings("whack the crying more", fixes, skip) == ("whack the crying mole", [("more", "mole")])
    assert fix_mishearings("hit the more.", fixes, skip)[0] == "hit the mole."
    assert fix_mishearings("Whack that More!", fixes, skip)[0] == "Whack that mole!"


def test_more_kept_where_it_is_a_real_word(config):
    from backend.entities import fix_mishearings
    v = config["voice"]
    fixes, skip = v["transcript_fixes"], v["transcript_fix_skip_after"]
    for text in ["one more", "no more", "a little more", "any more"]:
        assert fix_mishearings(text, fixes, skip) == (text, [])
