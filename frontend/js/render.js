// DOM rendering: board, moles, hammer, effects, transcript highlighting.

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export class Board {
  constructor(el, { mudEl, overlayEl }) {
    this.el = el;
    this.mudEl = mudEl;
    this.overlayEl = overlayEl;
    this.holes = [];
    this.showLabels = false;
    this._timers = new Map();
  }

  build(cols) {
    this.el.style.setProperty("--cols", cols);
    this.el.innerHTML = "";
    this.holes = [];
    for (let i = 0; i < cols * cols; i++) {
      const hole = document.createElement("div");
      hole.className = "hole";
      hole.dataset.slot = i;
      hole.innerHTML = `<div class="pit"><div class="mole"></div></div><div class="ring"></div><div class="hammer" aria-hidden="true">🔨</div>`;
      this.el.appendChild(hole);
      this.holes.push(hole);
    }
  }

  showMole(slot, inst, mole) {
    const hole = this.holes[slot];
    const moleEl = hole.querySelector(".mole");
    hole.querySelectorAll(".clue,.tag-label").forEach((n) => n.remove());
    moleEl.className = "mole";
    if (inst.golden) moleEl.classList.add("golden");
    if (mole.family === "bomb") moleEl.classList.add("bomb");

    const v = mole.visual || {};
    let face = `<div class="mole-face default"></div>`;
    let hat = "", prop = "", sign = "", badge = "";
    if (mole.family === "bomb") {
      face = `<div class="mole-face">💣</div><div class="fuse" aria-hidden="true">✸</div>`;
      badge = `<div class="mole-badge" title="bomb">⚠️</div>`;
    } else {
      if (v.face) face = `<div class="mole-face">${v.face}</div>`;
      if (v.hat) hat = `<div class="mole-hat">${v.hat}</div>`;
      if (v.prop) prop = `<div class="mole-prop">${v.prop}</div>`;
      if (mole.family === "sign") sign = `<div class="mole-sign">${esc(mole.label)}</div>`;
      if (mole.family === "riddle") sign = `<div class="mole-sign">?</div>`;
      if (inst.golden) badge = `<div class="mole-badge" title="golden">★</div>`;
      else if (inst.shapeshifter) badge = `<div class="mole-badge" title="shapeshifter">🌀</div>`;
      else if (inst.twin) badge = `<div class="mole-badge" title="twin">👯</div>`;
    }
    moleEl.innerHTML = `<div class="mole-body">${hat}${face}${prop}${sign}${badge}</div>`;

    if (mole.family === "riddle") {
      const clue = document.createElement("div");
      clue.className = "clue";
      clue.textContent = mole.clue;
      hole.appendChild(clue);
    }
    if (this.showLabels) {
      const label = document.createElement("div");
      label.className = "tag-label";
      label.textContent = mole.family === "riddle" ? "riddle" : mole.family === "sign" ? `sign: ${mole.label}` : mole.label;
      hole.appendChild(label);
    }
    hole.setAttribute("aria-label", `${mole.family} mole${inst.golden ? ", golden" : ""}`);
    hole.classList.remove("hit", "wrong", "ambiguous", "laugh", "windup", "strike", "targeted");
    hole.classList.add("up");
  }

  shapeshift(slot) {
    const m = this.holes[slot].querySelector(".mole");
    m.classList.add("shifting");
    setTimeout(() => m.classList.remove("shifting"), 900);
  }

  hideMole(slot) {
    const hole = this.holes[slot];
    if (!hole) return;
    hole.classList.remove("up");
    hole.querySelectorAll(".clue,.tag-label").forEach((n) => n.remove());
    hole.removeAttribute("aria-label");
  }

  flash(slot, cls, ms = 700) {
    const hole = this.holes[slot];
    if (!hole) return;
    hole.classList.add(cls);
    const key = `${slot}:${cls}`;
    clearTimeout(this._timers.get(key));
    this._timers.set(key, setTimeout(() => hole.classList.remove(cls), ms));
  }

  windup(slot) {
    const hole = this.holes[slot];
    if (!hole) return;
    hole.classList.remove("strike");
    hole.classList.add("windup", "targeted");
  }

  strike(slot) {
    const hole = this.holes[slot];
    if (!hole) return;
    hole.classList.remove("windup", "targeted");
    this.flash(slot, "strike", 300);
  }

  whack(slot) {
    const mole = this.holes[slot]?.querySelector(".mole");
    if (!mole) return;
    mole.classList.remove("whacked");
    void mole.offsetWidth;
    mole.classList.add("whacked");
  }

  lowerHammer(slot) {
    const hole = this.holes[slot];
    if (hole) hole.classList.remove("windup", "targeted", "strike");
  }

  float(slot, text, kind = "good") {
    const hole = this.holes[slot];
    if (!hole) return;
    const f = document.createElement("div");
    f.className = `float ${kind}`;
    f.textContent = text;
    hole.appendChild(f);
    setTimeout(() => f.remove(), 1250);
  }

  laugh(slot) { this.flash(slot, "laugh", 900); }

  shake() {
    this.el.classList.remove("shake");
    void this.el.offsetWidth;
    this.el.classList.add("shake");
  }

  mud(ms) {
    this.mudEl.hidden = false;
    this.mudEl.style.inset = `${10 + Math.random() * 20}% ${10 + Math.random() * 25}% ${15 + Math.random() * 20}% ${10 + Math.random() * 25}%`;
    clearTimeout(this._mudTimer);
    this._mudTimer = setTimeout(() => { this.mudEl.hidden = true; }, ms);
  }

  overlay(text, ms) {
    this.overlayEl.textContent = text;
    this.overlayEl.hidden = false;
    clearTimeout(this._ovTimer);
    if (ms) this._ovTimer = setTimeout(() => { this.overlayEl.hidden = true; }, ms);
  }

  clearOverlay() { this.overlayEl.hidden = true; }
}

// Transcript with entity groups highlighted, using the start/end offsets from GAMEBERT.
export function taggedTranscript(text, entities = []) {
  if (!text) return "";
  const marks = new Array(text.length).fill(null);
  for (const e of entities) {
    if (e.start == null || e.end == null) continue;
    let s = e.start, en = e.end;
    while (s > 0 && /[\w']/.test(text[s - 1])) s--;
    while (en < text.length && /[\w']/.test(text[en])) en++;
    for (let i = s; i < en; i++) marks[i] = e.entity_group;
  }
  let html = "", cur = null, buf = "";
  const flush = () => {
    if (!buf) return;
    html += cur ? `<span class="tagged tag-${cur}" title="${cur}">${esc(buf)}</span>` : esc(buf);
    buf = "";
  };
  for (let i = 0; i < text.length; i++) {
    const m = text[i] === " " && marks[i - 1] !== marks[i + 1] ? null : marks[i];
    if (m !== cur) { flush(); cur = m; }
    buf += text[i];
  }
  flush();
  return html;
}

export function tagLegend(entities = []) {
  const groups = [...new Set(entities.map((e) => e.entity_group))];
  return groups.map((g) => `<span class="tagged tag-${g} tag-legend">${g}</span>`).join(" ");
}

export { esc };
