import { Backend } from "./net.js";
import { Sound } from "./audio.js";
import { Board, taggedTranscript, tagLegend, esc } from "./render.js";
import { Game } from "./game.js";

const $ = (id) => document.getElementById(id);
const backend = new Backend();
const sound = new Sound();
sound.captionEl = $("captions");
const board = new Board($("board"), { mudEl: $("mud"), overlayEl: $("overlay-msg") });

let config = null;
let health = null;
let game = null;
let screen = "start";
let lastUtterance = null;
const opts = loadOpts();

// ---------------- options (per-viewer conveniences) ----------------

function loadOpts() {
  try { return JSON.parse(localStorage.getItem("whack-opts")) || {}; } catch { return {}; }
}
function saveOpts() {
  try { localStorage.setItem("whack-opts", JSON.stringify(opts)); } catch { /* storage unavailable */ }
}
function bindOpt(id, key, apply) {
  const el = $(id);
  if (key in opts) el.checked = opts[key];
  const run = () => { opts[key] = el.checked; saveOpts(); apply(el.checked); };
  el.addEventListener("change", run);
  apply(el.checked);
}
bindOpt("opt-labels", "labels", (v) => { board.showLabels = v; });
bindOpt("opt-captions", "captions", (v) => { sound.captions = v; });
bindOpt("opt-sound", "sound", (v) => { sound.enabled = v; });
bindOpt("opt-debug", "debug", (v) => { $("debug").hidden = !v; });
bindOpt("opt-typed", "typed", () => updateTypedForm());
document.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "d" && !(e.target instanceof HTMLInputElement)) $("opt-debug").click();
});

function typedMode() { return $("opt-typed").checked || !voiceAvailable(); }
function voiceAvailable() { return health && health.phase === "listening"; }
function updateTypedForm() {
  $("typed-form").hidden = !typedMode();
  if (screen === "game" && typedMode()) $("typed-input").focus();
}

// ---------------- screens ----------------

function show(name) {
  screen = name;
  for (const s of ["start", "game", "summary"]) $(`screen-${s}`).hidden = s !== name;
  if (name === "start") sendMicTestState();
}

function banner(text, kind = "error") {
  const b = $("banner");
  if (!text) { b.hidden = true; return; }
  b.textContent = text;
  b.className = `banner ${kind === "info" ? "info" : ""}`;
  b.hidden = false;
}

// ---------------- health ----------------

const PHASES = {
  starting: "Starting backend…",
  loading_models: "Loading models (Whisper + GAMEBERT)… first run downloads them",
  calibrating: "🎙️ Calibrating microphone — stay quiet…",
  listening: "✅ Listening on the backend's microphone",
  typed_only: "⌨️ Typed mode (no microphone)",
  error: "❌ Backend error",
};

function renderHealth(h) {
  health = h;
  const phase = $("phase");
  phase.textContent = PHASES[h.phase] || h.phase;
  phase.className = `phase ${h.phase}`;
  const rows = [
    ["Model loaded", h.model_loaded ? "yes" : "no"],
    ["NER model", h.ner_model],
    ["Labels", h.id2label ? Object.values(h.id2label).join(", ") : "—"],
    ["Whisper", `${h.whisper_model} on ${h.backend}`],
    ["Mic calibrated", h.mic_calibrated ? "yes" : "no"],
  ];
  if (h.energy_threshold != null) rows.push(["Mic energy threshold", h.energy_threshold]);
  if (h.mic_error) rows.push(["Mic", h.mic_error]);
  if (h.mic_warning) rows.push(["Mic warning", h.mic_warning]);
  if (h.error) rows.push(["Error", h.error]);
  $("health").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  const ready = h.model_loaded && config;
  $("btn-start").disabled = !ready;
  $("btn-start").textContent = ready ? "Start game" : "Loading…";

  if (h.phase === "error") banner(`Backend error: ${h.error || "model failed to load"}. Check the backend terminal.`);
  else if (h.mic_error && h.voice_enabled !== false) banner(`Microphone unavailable, so typed mode is on. (${h.mic_error})`);
  else if (h.mic_warning) banner(`${h.mic_warning} Unmute it and restart the backend, or use typed mode.`);
  else if (h.phase === "calibrating") banner("Calibrating microphone. Please stay quiet for 2 seconds.", "info");
  else banner(null);
  updateTypedForm();
}

async function pollHealthUntilReady() {
  try {
    renderHealth(await backend.health());
  } catch {
    $("phase").textContent = "❌ Backend offline";
    banner("Backend offline. Start it with:  .venv/bin/python -m backend.app   then reload this page.");
  }
}

// ---------------- mic test ----------------

function sendMicTestState() {
  if (!config) return;
  // rank against every mole so the mic test shows a real top 3
  backend.roundState(Object.keys(config.moles).map((id) => ({ key: id, id })));
}

function renderMicTest(msg) {
  $("mt-transcript").textContent = msg.transcript || (msg.dropped ? `(${msg.dropped})` : "…");
  $("mt-tags").innerHTML = msg.entities?.length
    ? taggedTranscript(msg.transcript, msg.entities) + " " + tagLegend(msg.entities)
    : "<em>no entities tagged</em>";
  $("mt-top").innerHTML = (msg.top || []).map((t) => `${esc(t.id)} <small>${t.score.toFixed(3)}</small>`).join(" · ")
    + (msg.mode ? ` <small>[${msg.mode}${msg.query ? `: “${esc(msg.query)}”` : ""}]</small>` : "")
    + (msg.outcome ? ` → <strong>${esc(msg.outcome)}</strong>` : "");
  $("mt-latency").textContent = msg.latency_ms != null
    ? `${Math.round(msg.latency_ms)} ms processing${msg.audio_ms ? ` + ${Math.round(msg.audio_ms)} ms audio` : ""} (${msg.source})`
    : "";
}

$("mt-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("mt-input").value.trim();
  if (!text || !config) return;
  try {
    const visible = Object.keys(config.moles).map((id) => ({ key: id, id }));
    const r = await backend.match(text, { visibleMoles: visible });
    if (r.error) return banner(`Can't test yet: ${r.error}`, "info");
    renderMicTest({ ...r, source: "typed" });
  } catch {
    banner("Backend offline. Is it running?");
  }
});

// ---------------- game ----------------

const ui = {
  hud(g) {
    $("hud-score").textContent = g.score;
    $("hud-lives").textContent = "❤️".repeat(Math.max(0, g.lives)) + "🖤".repeat(Math.max(0, 3 - g.lives));
    $("hud-lives").setAttribute("aria-label", `${g.lives} lives`);
    $("hud-combo").textContent = `×${g.comboMult()}${g.streak ? ` (${g.streak})` : ""}`;
    $("hud-level").textContent = g.level.level;
    $("hud-time").textContent = Math.max(0, Math.ceil((g.endAt - Date.now()) / 1000));
  },
  mission(m) {
    $("mission").hidden = !m;
    if (m) $("mission").textContent = m.text;
  },
  toast(text) { sound.caption(text); setBubbleResult(text, "bad"); },
  summary(g, info) { showSummary(g, info); },
};

function startLevel(index, carry = null) {
  if (game) game.stop();
  $("mission").hidden = true;
  game = new Game({ config, backend, board, sound, ui });
  show("game");
  game.start(index, carry);
  setBubble("Say a mole's name…", []);
  setBubbleResult("");
  updateTypedForm();
}

$("btn-start").addEventListener("click", () => {
  sound.play("levelup");
  const index = config.levels.findIndex((l) => String(l.level) === $("opt-level").value);
  startLevel(Math.max(0, index));
});
$("btn-quit").addEventListener("click", () => { game?.stop(); show("start"); });
$("typed-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("typed-input");
  const text = input.value.trim();
  input.value = "";
  if (!text || !game?.running) return;
  try {
    const r = await backend.match(text, { play: true });
    if (r.error) setBubbleResult(`Backend not ready: ${r.error}`, "bad");
  } catch {
    setBubbleResult("Backend offline", "bad");
  }
});

function setBubble(text, entities) {
  $("bubble-text").innerHTML = entities?.length ? taggedTranscript(text, entities) : esc(text);
}
function setBubbleResult(text, kind = "") {
  const el = $("bubble-result");
  el.textContent = text;
  el.className = `bubble-result ${kind}`;
}

function moleName(id) {
  const m = config.moles[id];
  if (!m) return id;
  if (m.family === "riddle") return `riddle (${m.answer})`;
  return m.label;
}

// ---------------- server events ----------------

backend.on((msg) => {
  switch (msg.type) {
    case "_connection":
      if (msg.connected) { pollHealthUntilReady(); if (screen === "start") sendMicTestState(); }
      else { $("phase").textContent = "❌ Backend offline"; banner("Backend offline. Reconnecting…"); $("btn-start").disabled = true; }
      return;
    case "status":
      if (msg.health) renderHealth(msg.health);
      if (msg.error) banner(`Voice error: ${msg.error}`);
      return;
    case "utterance":
      lastUtterance = msg;
      renderDebug(msg);
      if (screen === "start") return renderMicTest(msg);
      if (screen !== "game" || !game?.running) return;
      game.onUtterance(msg);
      setBubble(msg.transcript || "…", msg.entities);
      if (msg.fixed_words?.length) {
        const note = msg.fixed_words.map(([h, t]) => `“${esc(h)}” → ${esc(t)}`).join(", ");
        $("bubble-text").insertAdjacentHTML("beforeend", ` <small class="fixed-note">(heard ${note})</small>`);
      }
      if (msg.dropped) setBubbleResult(msg.dropped === "too_long" ? "Too long. Keep it short!" : "Didn't catch that", "meh");
      else if (msg.locked) setBubbleResult(game.lockReason === "spam" ? "Slow down!" : "Input locked", "meh");
      return;
  }
  if (screen !== "game" || !game?.running) return;
  switch (msg.type) {
    case "whack":
      if (msg.phase === "windup") {
        const extra = msg.action === "defuse" ? " (defuse)" : "";
        setBubbleResult(`→ ${moleName(msg.mole.id)}${extra}  [${msg.tier}, ${msg.score.toFixed(2)}]`, "good");
      }
      game.onWhack(msg);
      break;
    case "ambiguous":
      setBubbleResult(`Which one? ${msg.candidate_ids.map(moleName).join(" or ")}. Use a word only one of them has.`, "meh");
      game.onAmbiguous(msg);
      break;
    case "no_match": {
      const words = msg.mode === "target" && msg.target_words?.length ? ` (“${msg.target_words.join(" ")}”)` : "";
      setBubbleResult(`No mole matched${words}`, "meh");
      game.onNoMatch(msg);
      break;
    }
    case "correction":
      if (msg.effect === "cancelled") setBubbleResult("Correction: whack cancelled", "meh");
      else if (msg.effect === "refund") setBubbleResult("Correction: too late, but half the penalty is refunded", "meh");
      game.onCorrection(msg);
      break;
  }
  renderDebug(lastUtterance, msg);
});

function renderDebug(u, result) {
  if ($("debug").hidden || !u) return;
  const lines = [
    `source      ${u.source}`,
    `transcript  ${u.transcript}`,
    `latency     ${u.latency_ms ?? "-"} ms  (whisper ${u.whisper_ms ?? "-"} / ner ${u.ner_ms ?? "-"})`,
    `audio       ${u.audio_ms ?? "-"} ms`,
    `mode/query  ${u.mode ?? "-"} / ${u.query ?? "-"}`,
    `top3        ${(u.top || []).map((t) => `${t.id}=${t.score.toFixed(3)}`).join("  ")}`,
    `outcome     ${u.outcome ?? "-"}`,
  ];
  if (result && result.type !== "utterance") lines.push(`event       ${result.type}${result.phase ? "/" + result.phase : ""} ${result.verdict ?? result.effect ?? ""}`);
  if (game?.stats.latencies.length) {
    const l = game.stats.latencies;
    lines.push(`avg latency ${Math.round(l.reduce((a, b) => a + b, 0) / l.length)} ms over ${l.length}`);
  }
  $("debug").textContent = lines.join("\n");
}

// ---------------- summary ----------------

let lastSummary = null;

function showSummary(g, info) {
  lastSummary = { g, info };
  const s = g.stats;
  const attempts = s.hits + s.wrong + s.missed + s.bombs;
  const avg = (a) => (a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : 0);
  $("sum-title").textContent = info.reason === "lives" ? "Out of lives!" : info.passed ? `Level ${g.level.level} cleared! 🎉` : "Time's up!";
  const rows = [
    ["Score", `${g.score} / ${g.level.target_score} to pass`],
    ["Accuracy", attempts ? `${Math.round((100 * s.hits) / attempts)}% (${s.hits} hits, ${s.wrong} wrong, ${s.missed} missed, ${s.bombs} bombs)` : "—"],
    ["Average score per hit", avg(s.hitPoints) || "—"],
    ["Best phrase", s.best ? `“${s.best.transcript}” (+${s.best.points})` : "—"],
    ["Average latency", s.latencies.length ? `${avg(s.latencies)} ms (${s.latencies.length} voice utterances)` : "— (typed only)"],
    ["Bombs defused", s.defused],
    ["Corrections", s.corrections],
    ["Ambiguous / no match", `${s.ambiguous} / ${s.nomatch}`],
  ];
  $("sum-stats").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  const words = [...s.unmatched.entries()].sort((a, b) => b[1] - a[1]);
  $("sum-words").innerHTML = words.length
    ? words.map(([w, n]) => `<span class="chip">${esc(w)}${n > 1 ? ` ×${n}` : ""}</span>`).join("")
    : "<em>none 🎉</em>";
  $("btn-next").hidden = !(info.passed && info.hasNext);
  if (info.passed) sound.play("levelup");
  show("summary");
}

$("btn-next").addEventListener("click", () => startLevel(lastSummary.g.levelIndex + 1, { lives: lastSummary.g.lives }));
$("btn-retry").addEventListener("click", () => startLevel(lastSummary.g.levelIndex));
$("btn-menu").addEventListener("click", () => show("start"));

// ---------------- boot ----------------

async function boot() {
  try {
    config = await backend.config();
    $("opt-level").innerHTML = config.levels.map((l) => `<option value="${l.level}">${l.level}</option>`).join("");
    await pollHealthUntilReady();
    sendMicTestState();
  } catch {
    $("phase").textContent = "❌ Backend offline";
    banner("Backend offline. Start it with:  .venv/bin/python -m backend.app   then reload this page.");
    setTimeout(boot, 3000);
  }
}
backend.connect();
boot();

// Console/debug hook: window.whack.game is the running Game.
window.whack = { get game() { return game; }, get config() { return config; }, backend };
