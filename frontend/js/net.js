// WebSocket + HTTP client for the local backend. The frontend never captures audio;
// the backend listens on its own microphone and pushes events here.

export class Backend {
  constructor(base = location.origin) {
    this.base = base;
    this.wsUrl = base.replace(/^http/, "ws") + "/ws";
    this.handlers = new Set();
    this.ws = null;
    this.connected = false;
    this.retryMs = 1000;
    this.lastRoundState = null;
  }

  on(fn) { this.handlers.add(fn); return () => this.handlers.delete(fn); }
  emit(msg) { for (const fn of this.handlers) fn(msg); }

  connect() {
    const ws = new WebSocket(this.wsUrl);
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true;
      this.retryMs = 1000;
      this.emit({ type: "_connection", connected: true });
      if (this.lastRoundState) this.send(this.lastRoundState);
    };
    ws.onmessage = (e) => {
      try { this.emit(JSON.parse(e.data)); } catch (err) { console.error("bad message", err); }
    };
    ws.onclose = () => {
      const was = this.connected;
      this.connected = false;
      this.emit({ type: "_connection", connected: false, was });
      setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 1.6, 8000);
    };
    ws.onerror = () => ws.close();
  }

  send(msg) {
    if (msg.type === "round_state") this.lastRoundState = msg;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  roundState(visibleMoles, { missionTag = null, inputLocked = false } = {}) {
    this.send({ type: "round_state", visible_moles: visibleMoles, mission_tag: missionTag, input_locked: inputLocked });
  }

  async health() { return (await fetch(this.base + "/health")).json(); }
  async config() { return (await fetch(this.base + "/config")).json(); }

  async match(text, { visibleMoles = null, missionTag = null, play = false } = {}) {
    const res = await fetch(this.base + "/match", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, visible_moles: visibleMoles, mission_tag: missionTag, play }),
    });
    return res.json();
  }
}
