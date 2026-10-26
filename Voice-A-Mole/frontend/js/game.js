// Game loop, spawning, scoring, and penalties. Recognition and match resolution run on the
// backend; this module reacts to its events and owns everything time- and score-related.

const TICK_MS = 50;
const SLOT_REUSE_MS = 800;
const TARGETED_HOLD_MS = 1500;
const MISSION_ROTATE_MS = 20000;
const START_LIVES = 3;
const FILLER = new Set(["the", "a", "an", "one", "that", "this", "it", "on", "of", "with", "and", "to", "uh", "um",
  "mole", "guy", "thing", "please", "now", "there", "is", "i", "you", "my", "me", "at", "in"]);

const rand = (arr) => arr[Math.floor(Math.random() * arr.length)];
const now = () => Date.now();

export class Game {
  constructor({ config, backend, board, sound, ui }) {
    this.cfg = config;
    this.backend = backend;
    this.board = board;
    this.sound = sound;
    this.ui = ui;
    this.timing = config.timing;
    this.scoring = config.scoring;
    this.penalties = config.penalties;
    this.running = false;
    this.knownWords = new Set();
    for (const m of Object.values(config.moles)) m.description.split(/\s+/).forEach((w) => this.knownWords.add(w));
    for (const a of Object.values(config.actions)) a.description.split(/\s+/).forEach((w) => this.knownWords.add(w));
  }

  // ---------------- lifecycle ----------------

  start(levelIndex, carry = null) {
    this.levelIndex = levelIndex;
    this.level = this.cfg.levels[levelIndex];
    this.score = 0;
    this.lives = carry?.lives ?? START_LIVES;
    this.streak = 0;
    this.consecMisses = 0;
    this.moles = new Map();   // key -> instance on screen
    this.gone = new Map();    // key -> instance that left recently (late whacks still count)
    this.slotFreeAt = [];
    this.spamTimes = [];
    this.lockedUntil = 0;
    this.lockReason = null;
    this.windups = new Map(); // windup_id -> key
    this.lastPenalty = null;
    this.mission = null;
    this.nextKey = 1;
    this.stats = {
      hits: 0, wrong: 0, missed: 0, bombs: 0, defused: 0, corrections: 0, ambiguous: 0, nomatch: 0,
      hitPoints: [], latencies: [], best: null, unmatched: new Map(), utterances: 0,
    };

    const cols = this.level.grid;
    this.board.build(cols);
    this.slotFreeAt = new Array(cols * cols).fill(0);
    this.startedAt = now();
    this.endAt = this.startedAt + this.level.duration_s * 1000;
    this.nextSpawnAt = this.startedAt + 800;
    if (this.level.mission) this.rotateMission();
    this.backend.send({ type: "reset" });
    this.sendRoundState();
    this.running = true;
    this.timer = setInterval(() => this.tick(), TICK_MS);
    this.ui.hud(this);
  }

  stop() {
    this.running = false;
    clearInterval(this.timer);
    for (const inst of this.moles.values()) clearTimeout(inst.missTimer);
    for (const inst of this.gone.values()) clearTimeout(inst.missTimer);
    this.backend.roundState([]);
  }

  end(reason) {
    if (!this.running) return;
    this.stop();
    const passed = reason === "time" && this.lives > 0 && this.score >= this.level.target_score;
    this.ui.summary(this, { reason, passed, hasNext: this.levelIndex + 1 < this.cfg.levels.length });
  }

  // ---------------- round state ----------------

  visibleForBackend() {
    return [...this.moles.values()].map((m) => {
      const out = { key: m.key, id: m.id };
      if (m.golden) out.golden = true;
      if (m.shiftOf) out.shift_of = m.shiftOf;
      return out;
    });
  }

  get inputLocked() { return now() < this.lockedUntil; }

  sendRoundState() {
    this.backend.roundState(this.visibleForBackend(), {
      missionTag: this.mission?.tag ?? null,
      inputLocked: this.inputLocked,
    });
    this._sentLocked = this.inputLocked;
  }

  lock(ms, reason) {
    this.lockedUntil = Math.max(this.lockedUntil, now() + ms);
    this.lockReason = reason;
    this.sendRoundState();
  }

  // ---------------- loop ----------------

  tick() {
    if (!this.running) return;
    const t = now();
    if (t >= this.endAt) return this.end("time");

    for (const inst of [...this.moles.values()]) {
      if (inst.shapeshifter && !inst.shifted && t >= inst.shiftAt) this.shapeshift(inst);
      const holdUntil = inst.retreatAt + (inst.targeted ? TARGETED_HOLD_MS : 0);
      if (!inst.laughed && !inst.targeted && t >= holdUntil - 400 && this.isTarget(inst)) {
        inst.laughed = true;
        this.board.laugh(inst.slot);
      }
      if (t >= holdUntil) this.retreat(inst, { timedOut: true });
    }

    if (t >= this.nextSpawnAt) {
      this.spawn();
      this.nextSpawnAt = t + this.level.spawn_interval_ms * (0.75 + Math.random() * 0.5);
    }
    if (this.mission && t >= this.mission.until) this.rotateMission();

    if (this._sentLocked && !this.inputLocked) {
      this.board.clearOverlay();
      this.sendRoundState();
    }
    this.ui.hud(this);
  }

  // ---------------- spawning ----------------

  freeSlots() {
    const busy = new Set([...this.moles.values()].map((m) => m.slot));
    const t = now();
    return this.slotFreeAt.map((at, i) => i).filter((i) => !busy.has(i) && this.slotFreeAt[i] <= t);
  }

  visibleIds() { return new Set([...this.moles.values()].map((m) => m.id)); }

  pickIdentity({ families = this.level.families, pool = null, preferTag = null } = {}) {
    const visible = this.visibleIds();
    let ids = pool ?? Object.keys(this.cfg.moles).filter((id) => families.includes(this.cfg.moles[id].family));
    ids = ids.filter((id) => !visible.has(id) && this.cfg.moles[id].family !== "bomb");
    if (!ids.length) return null;
    if (preferTag && Math.random() < 0.55) {
      const tagged = ids.filter((id) => (this.cfg.moles[id].tags || []).includes(preferTag));
      if (tagged.length) return rand(tagged);
    }
    if (!pool) {
      // pick a family first, so families with few moles still show up
      const fams = [...new Set(ids.map((id) => this.cfg.moles[id].family))];
      const fam = rand(fams);
      return rand(ids.filter((id) => this.cfg.moles[id].family === fam));
    }
    return rand(ids);
  }

  spawn() {
    const L = this.level;
    let slots = this.freeSlots();
    const room = L.max_visible - this.moles.size;
    if (!slots.length || room <= 0) return;
    const visible = this.visibleIds();
    const r = Math.random();

    if (L.bomb_chance && r < L.bomb_chance && !visible.has("bomb")) {
      return this.addMole("bomb", rand(slots), { bomb: true });
    }
    if (L.twin_chance && Math.random() < L.twin_chance && room >= 2 && slots.length >= 2) {
      const pairs = (this.cfg.twins || []).filter(([a, b]) => !visible.has(a) && !visible.has(b) &&
        L.families.includes(this.cfg.moles[a].family));
      if (pairs.length) {
        const [a, b] = rand(pairs);
        const s1 = rand(slots);
        slots = slots.filter((s) => s !== s1);
        this.addMole(a, s1, { twin: true });
        this.addMole(b, rand(slots), { twin: true });
        return;
      }
    }
    const preferTag = this.mission?.tag;
    if (L.golden_chance && Math.random() < L.golden_chance) {
      const id = this.pickIdentity({ preferTag });
      if (id) return this.addMole(id, rand(slots), { golden: true });
    }
    if (L.shapeshifter_chance && Math.random() < L.shapeshifter_chance) {
      const pool = (this.cfg.shapeshifter_pool || []).filter((id) => L.families.includes(this.cfg.moles[id]?.family));
      const id = this.pickIdentity({ pool });
      if (id) return this.addMole(id, rand(slots), { shapeshifter: true });
    }
    const id = this.pickIdentity({ preferTag });
    if (id) this.addMole(id, rand(slots), {});
  }

  addMole(id, slot, flags) {
    const t = now();
    const grace = this.timing.RECOGNITION_GRACE_MS;
    let base = this.level.base_window_ms;
    if (flags.golden) base *= this.cfg.golden?.window_factor ?? 0.6;
    const inst = {
      key: `m${this.nextKey++}`, id, slot, spawnedAt: t, windowMs: base + grace,
      retreatAt: t + base + grace, ...flags,
    };
    if (flags.shapeshifter) inst.shiftAt = t + base * 0.45;
    this.moles.set(inst.key, inst);
    this.board.showMole(slot, inst, this.cfg.moles[id]);
    this.sendRoundState();
    return inst;
  }

  shapeshift(inst) {
    const pool = (this.cfg.shapeshifter_pool || []).filter((id) => id !== inst.id);
    const newId = this.pickIdentity({ pool });
    inst.shifted = true;
    if (!newId) return;
    // New key + shift_of, so the backend drops the old identity right away.
    this.moles.delete(inst.key);
    const next = { ...inst, key: `m${this.nextKey++}`, id: newId, shiftOf: inst.key, shifted: true };
    this.moles.set(next.key, next);
    this.board.shapeshift(inst.slot);
    this.board.showMole(inst.slot, next, this.cfg.moles[newId]);
    this.sendRoundState();
    next.shiftOf = null;
  }

  // Does letting this mole escape count as a miss?
  isTarget(inst) {
    const mole = this.cfg.moles[inst.id];
    if (mole.family === "bomb" || inst.golden) return false;
    if (this.mission && !(mole.tags || []).includes(this.mission.tag)) return false;
    return true;
  }

  retreat(inst, { timedOut = false } = {}) {
    if (!this.moles.has(inst.key)) return;
    this.moles.delete(inst.key);
    this.board.hideMole(inst.slot);
    this.slotFreeAt[inst.slot] = now() + SLOT_REUSE_MS;
    inst.goneAt = now();
    this.gone.set(inst.key, inst);
    setTimeout(() => this.gone.delete(inst.key), 10000);
    if (timedOut && !inst.whacked && this.isTarget(inst)) {
      // Wait for in-flight recognition: a phrase that started while the mole was up still counts.
      inst.missTimer = setTimeout(() => {
        if (!inst.whacked && !inst.targeted && this.running) this.applyMiss(inst);
      }, this.timing.MISS_SETTLE_MS);
    }
    this.sendRoundState();
  }

  rotateMission() {
    const options = (this.cfg.missions || []).filter((m) => m.tag !== this.mission?.tag);
    const m = rand(options.length ? options : this.cfg.missions);
    this.mission = { ...m, until: now() + MISSION_ROTATE_MS };
    this.ui.mission(this.mission);
    this.sendRoundState();
  }

  // ---------------- scoring ----------------

  comboMult() { return Math.min(this.scoring.combo_cap, 1 + Math.floor(this.streak / this.scoring.combo_step)); }

  kindOf(inst) {
    const mole = this.cfg.moles[inst.id];
    if (inst.golden) return "golden";
    if (inst.shapeshifter) return "shapeshifter";
    if (inst.twin) return "twin";
    if (mole.family === "riddle") return "riddle";
    return "normal";
  }

  pointsFor(inst, msg) {
    const base = this.scoring.base_points[this.kindOf(inst)];
    const tierMult = this.scoring.tier_multipliers[msg.tier] ?? 1;
    const spoken = msg.spoken_at ?? now();
    const frac = Math.min(1, Math.max(0, (spoken - inst.spawnedAt) / inst.windowMs));
    const speed = 1 + this.scoring.speed_bonus_max * (1 - frac);
    const sign = msg.sign_word_only ? this.scoring.sign_word_multiplier : 1;
    return Math.round(base * tierMult * speed * this.comboMult() * sign);
  }

  addScore(delta) { this.score = Math.max(0, this.score + delta); }

  applyMiss(inst) {
    this.stats.missed++;
    this.streak = 0;
    this.consecMisses++;
    this.addScore(-this.penalties.missed_window_points);
    // the mole has already ducked, so its laugh is shown as floating text over the hole
    this.board.float(inst.slot, `😂 −${this.penalties.missed_window_points} missed`, "bad");
    this.sound.play("miss");
    if (this.consecMisses % this.penalties.misses_per_life === 0) this.loseLife(`${this.penalties.misses_per_life} misses in a row`);
  }

  loseLife(why) {
    this.lives--;
    this.ui.toast(`−1 life: ${why}`);
    if (this.lives <= 0) setTimeout(() => this.end("lives"), 600);
  }

  // ---------------- server events ----------------

  findInst(key) { return this.moles.get(key) || this.gone.get(key); }

  onUtterance(msg) {
    this.stats.utterances++;
    if (msg.latency_ms != null && msg.source === "voice") this.stats.latencies.push(msg.latency_ms);
  }

  onWhack(msg) {
    const key = msg.mole?.key;
    const inst = this.findInst(key);
    if (!inst || inst.whacked) return;
    if (msg.phase === "windup") {
      inst.targeted = true;
      this.windups.set(msg.windup_id, key);
      this.board.windup(inst.slot);
      this.sound.play("windup");
      return;
    }
    // strike
    this.windups.delete(msg.windup_id);
    inst.targeted = false;
    this.board.strike(inst.slot);
    this.resolveStrike(inst, msg);
  }

  resolveStrike(inst, msg) {
    const slot = inst.slot;
    const v = msg.verdict;
    if (v === "golden_resist") {
      this.board.float(slot, "Clang! Too weak: describe it more", "meh");
      this.sound.play("golden");
      return;
    }
    inst.whacked = true;
    clearTimeout(inst.missTimer);

    if (v === "hit") {
      const pts = this.pointsFor(inst, msg);
      this.addScore(pts);
      this.streak++;
      this.consecMisses = 0;
      this.stats.hits++;
      this.stats.hitPoints.push(pts);
      if (!this.stats.best || pts > this.stats.best.points) this.stats.best = { points: pts, transcript: msg.transcript };
      const signNote = msg.sign_word_only ? " (sign word ¼)" : "";
      this.board.whack(slot);
      this.board.float(slot, `+${pts} ${msg.tier}!${signNote}`, "good");
      this.board.flash(slot, "hit", 600);
      this.sound.play("hit");
    } else if (v === "defused") {
      const pts = Math.round(this.scoring.base_points.bomb_defused * this.comboMult());
      this.addScore(pts);
      this.stats.defused++;
      this.board.float(slot, `+${pts} Defused!`, "good");
      this.board.flash(slot, "hit", 600);
      this.sound.play("defuse");
    } else if (v === "bomb") {
      this.stats.bombs++;
      this.streak = 0;
      this.board.flash(slot, "wrong", 800);
      this.board.float(slot, "BOOM! −1 life", "bad");
      this.board.shake();
      this.sound.play("bomb");
      this.lock(this.timing.BOMB_LOCKOUT_MS, "bomb");
      this.board.overlay("💥 BOOM!", this.timing.BOMB_LOCKOUT_MS);
      this.loseLife("whacked a bomb");
    } else if (v === "wrong_mission") {
      const pen = this.penalties.wrong_mole_points;
      this.addScore(-pen);
      this.streak = 0;
      this.stats.wrong++;
      this.lastPenalty = { key: inst.key, amount: pen, refunded: false };
      this.board.whack(slot);
      this.board.flash(slot, "wrong", 800);
      this.board.float(slot, `−${pen} wrong mole!`, "bad");
      this.board.mud(this.timing.MUD_MS);
      this.sound.play("wrong");
    }
    setTimeout(() => this.retreat(inst), v === "hit" || v === "wrong_mission" ? 420 : 250);
  }

  onCorrection(msg) {
    const key = msg.whack?.mole?.key;
    const inst = key && this.findInst(key);
    if (msg.effect === "cancelled" || msg.effect === "superseded") {
      if (inst) {
        inst.targeted = false;
        this.board.lowerHammer(inst.slot);
        if (msg.effect === "cancelled") this.board.float(inst.slot, "Phew! Cancelled", "meh");
      }
      if (msg.effect === "cancelled") { this.stats.corrections++; this.sound.play("correction"); }
    } else if (msg.effect === "refund") {
      const p = this.lastPenalty;
      if (p && p.key === key && !p.refunded && msg.refund_fraction > 0) {
        const back = Math.round(p.amount * msg.refund_fraction);
        p.refunded = true;
        this.addScore(back);
        this.stats.corrections++;
        if (inst) this.board.float(inst.slot, `+${back} regret refund`, "meh");
        this.sound.play("correction");
      } else {
        this.sound.caption("[correction: nothing to undo]");
      }
    } else {
      this.sound.caption("[correction: nothing to cancel]");
    }
  }

  onAmbiguous(msg) {
    this.stats.ambiguous++;
    for (const key of msg.candidates || []) {
      const inst = this.findInst(key);
      if (inst) this.board.flash(inst.slot, "ambiguous", 1400);
    }
    this.sound.play("ambiguous");
    this.recordSpam();
  }

  onNoMatch(msg) {
    this.stats.nomatch++;
    const words = (msg.mode === "target" ? msg.target_words : (msg.transcript || "").split(/[^a-z0-9']+/)) || [];
    for (const w of words) {
      if (!w || FILLER.has(w) || this.knownWords.has(w)) continue;
      this.stats.unmatched.set(w, (this.stats.unmatched.get(w) || 0) + 1);
    }
    this.sound.play("nomatch");
    this.recordSpam();
  }

  recordSpam() {
    const t = now();
    this.spamTimes = this.spamTimes.filter((x) => t - x <= this.timing.SPAM_WINDOW_MS);
    this.spamTimes.push(t);
    if (this.spamTimes.length > this.timing.SPAM_MAX_EVENTS && !this.inputLocked) {
      this.spamTimes = [];
      this.lock(this.timing.SPAM_COOLDOWN_MS, "spam");
      this.board.overlay("🐢 Slow down!", this.timing.SPAM_COOLDOWN_MS);
    }
  }
}
