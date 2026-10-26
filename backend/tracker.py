"""Tracks which mole instances were up when, using the server clock.

A whack counts if the utterance STARTED while the mole was up, so a mole
that retreats while Whisper is still working can still be hit. The client
sends round_state every time moles spawn, retreat, or shapeshift.

Each instance: {"key": unique per spawn, "id": mole identity, "golden"?, "shift_of"?}
A shapeshifter sends a new key with shift_of=<old key>. The old identity is
dropped right away, so only the current identity matches.
"""


class MoleTracker:
    def __init__(self, start_tolerance_ms=500, history_ms=10000):
        self.start_tolerance_ms = start_tolerance_ms
        self.history_ms = history_ms
        self.records = {}  # key -> {"inst", "up_at", "down_at"}
        self.mission_tag = None
        self.input_locked = False

    def update(self, visible, now_ms, mission_tag=None, input_locked=False):
        self.mission_tag = mission_tag or None
        self.input_locked = bool(input_locked)
        visible_keys = set()
        for inst in visible:
            key = inst.get("key") or inst["id"]
            inst = {**inst, "key": key}
            visible_keys.add(key)
            if inst.get("shift_of"):
                self.records.pop(inst["shift_of"], None)
            rec = self.records.get(key)
            if rec is None or rec["down_at"] is not None:
                self.records[key] = {"inst": inst, "up_at": now_ms, "down_at": None}
            else:
                rec["inst"] = inst
        for key, rec in self.records.items():
            if key not in visible_keys and rec["down_at"] is None:
                rec["down_at"] = now_ms
        cutoff = now_ms - self.history_ms
        self.records = {k: r for k, r in self.records.items()
                        if r["down_at"] is None or r["down_at"] >= cutoff}

    def reset(self):
        self.records.clear()
        self.mission_tag = None
        self.input_locked = False

    def candidates(self, spoken_at_ms):
        """Instances up when the utterance started. At most one instance per identity id."""
        chosen = {}
        for rec in self.records.values():
            up_ok = rec["up_at"] <= spoken_at_ms + self.start_tolerance_ms
            down_ok = rec["down_at"] is None or rec["down_at"] >= spoken_at_ms
            if not (up_ok and down_ok):
                continue
            mole_id = rec["inst"]["id"]
            prev = chosen.get(mole_id)
            # prefer the instance still on screen, then the most recent spawn
            if prev is None or (prev["down_at"] is not None and rec["down_at"] is None) \
                    or (rec["up_at"] > prev["up_at"] and (rec["down_at"] is None) == (prev["down_at"] is None)):
                chosen[mole_id] = rec
        return [r["inst"] for r in chosen.values()]

    def visible_now(self):
        return [r["inst"] for r in self.records.values() if r["down_at"] is None]
