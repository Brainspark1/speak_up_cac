// Synthesized sound cues (no audio files), each with a text caption.

const CAPTIONS = {
  hit: "[bonk!]",
  miss: "[mole laughs: hee-hee!]",
  wrong: "[splat! wrong mole]",
  bomb: "[BOOM!]",
  defuse: "[snip… bomb defused]",
  correction: "[phew, hammer lowered]",
  windup: "[hammer raised]",
  ambiguous: "[hmm? which one]",
  nomatch: "[didn't catch that]",
  golden: "[clang! too weak for gold]",
  levelup: "[fanfare]",
};

export class Sound {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.captions = true;
    this.captionEl = null;
    this._captionTimer = null;
  }

  _ctx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  _tone(freq, dur, { type = "sine", gain = 0.2, slideTo = null, delay = 0 } = {}) {
    const ctx = this._ctx();
    const t = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    if (slideTo) osc.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.connect(g).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + dur + 0.02);
  }

  _noise(dur, gain = 0.4) {
    const ctx = this._ctx();
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length) ** 2;
    const src = ctx.createBufferSource();
    const g = ctx.createGain();
    g.gain.value = gain;
    src.buffer = buf;
    src.connect(g).connect(ctx.destination);
    src.start();
  }

  play(name) {
    this.caption(CAPTIONS[name]);
    if (!this.enabled) return;
    try {
      switch (name) {
        case "hit": this._tone(180, 0.12, { type: "square", gain: 0.25, slideTo: 60 }); this._noise(0.05, 0.2); break;
        case "miss": [520, 620, 520, 620].forEach((f, i) => this._tone(f, 0.08, { type: "triangle", gain: 0.12, delay: i * 0.09 })); break;
        case "wrong": this._tone(300, 0.3, { type: "sawtooth", gain: 0.15, slideTo: 90 }); this._noise(0.15, 0.25); break;
        case "bomb": this._noise(0.8, 0.6); this._tone(80, 0.6, { type: "sine", gain: 0.4, slideTo: 30 }); break;
        case "defuse": this._tone(1200, 0.05, { type: "square", gain: 0.1 }); this._tone(900, 0.2, { gain: 0.15, delay: 0.08 }); break;
        case "correction": this._tone(440, 0.1, { gain: 0.15 }); this._tone(660, 0.15, { gain: 0.15, delay: 0.1 }); break;
        case "windup": this._tone(200, 0.25, { type: "triangle", gain: 0.08, slideTo: 500 }); break;
        case "ambiguous": this._tone(500, 0.1, { gain: 0.12 }); this._tone(700, 0.12, { gain: 0.12, delay: 0.12 }); break;
        case "nomatch": this._tone(250, 0.12, { type: "triangle", gain: 0.08 }); break;
        case "golden": this._tone(1500, 0.3, { type: "triangle", gain: 0.12 }); break;
        case "levelup": [523, 659, 784, 1046].forEach((f, i) => this._tone(f, 0.18, { type: "square", gain: 0.1, delay: i * 0.12 })); break;
      }
    } catch (e) { /* audio unavailable: captions still work */ }
  }

  caption(text) {
    if (!this.captions || !this.captionEl || !text) return;
    this.captionEl.textContent = text;
    clearTimeout(this._captionTimer);
    this._captionTimer = setTimeout(() => { this.captionEl.textContent = ""; }, 1600);
  }
}
