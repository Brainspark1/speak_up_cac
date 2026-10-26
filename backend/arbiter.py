"""Hammer wind-up and correction timing.

Pure state machine. Callers pass in times (ms), so tests need no clock.

Corrections are judged by when the player STARTED speaking, not when the
result arrives. Voice results arrive ~1-2 s after speech starts
(end-of-phrase silence + Whisper + NER), so a "no!" said during a 500 ms
wind-up usually lands after the strike has already played:

  - correction arrives before the strike     -> wind-up cancelled, no penalty
  - correction started during the wind-up,
    or within REGRET_WINDOW_MS of the strike,
    but arrived after it                      -> regret: half the penalty refunded
"""


class WhackArbiter:
    def __init__(self, windup_ms=500, regret_window_ms=300, refund_fraction=0.5,
                 refundable_verdicts=("wrong_mission",)):
        self.windup_ms = windup_ms
        self.regret_window_ms = regret_window_ms
        self.refund_fraction = refund_fraction
        self.refundable = set(refundable_verdicts)
        self.pending = None      # {"whack": result, "started_at", "strike_at", "id"}
        self.last_strike = None  # {"whack": result, "windup_started_at", "struck_at", "refunded"}
        self._next_id = 1

    def begin_windup(self, whack, now_ms):
        """Start a wind-up. A newer whack replaces an older pending one (latest intent wins)."""
        superseded = self.pending
        self.pending = {
            "id": self._next_id,
            "whack": whack,
            "started_at": now_ms,
            "strike_at": now_ms + self.windup_ms,
        }
        self._next_id += 1
        return self.pending, superseded

    def strike_due(self, windup_id, now_ms):
        """Called when the wind-up timer fires. Returns the struck wind-up, or None if it was cancelled."""
        p = self.pending
        if p is None or p["id"] != windup_id or now_ms < p["strike_at"]:
            return None
        self.pending = None
        self.last_strike = {
            "whack": p["whack"],
            "windup_started_at": p["started_at"],
            "struck_at": p["strike_at"],
            "refunded": False,
        }
        return p

    def correction(self, spoken_at_ms, now_ms):
        """Apply a correction. Returns {"effect": "cancelled"|"refund"|"none", ...}."""
        p = self.pending
        if p is not None:
            # Strike not broadcast yet (even if its timer is a few ms late), so cancel it.
            self.pending = None
            return {"effect": "cancelled", "whack": p["whack"]}

        s = self.last_strike
        if s is not None and not s["refunded"]:
            started_in_windup = s["windup_started_at"] <= spoken_at_ms <= s["struck_at"]
            within_regret = s["struck_at"] <= spoken_at_ms <= s["struck_at"] + self.regret_window_ms
            if started_in_windup or within_regret:
                s["refunded"] = True
                refundable = s["whack"].get("verdict") in self.refundable
                return {
                    "effect": "refund",
                    "whack": s["whack"],
                    "refund_fraction": self.refund_fraction if refundable else 0.0,
                }
        return {"effect": "none"}
