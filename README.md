# Whack-a-Word 🔨

A voice-controlled whack-a-mole game. You whack moles by **saying** who they are
("whack the cook", "hit the one with the parrot", "defuse the bomb"). Speech goes
through [`voice-speak-up`](https://pypi.org/project/voice-speak-up/) v1.0.1:
microphone → Whisper → `Saggarwal/GAMEBERT` token tagging (ACTION / TARGET / CORRECTION)
→ TF-IDF matching against each mole's description.

```
backend/            FastAPI + WebSocket server, MoleVoiceController, RankedMapper, resolver
frontend/           Plain HTML/CSS/ES modules (no build step), served by the backend at /
game_config.json    The one file you edit: vocabulary, thresholds, timings, levels
calibrate.py        Checks phrases against the thresholds; flags overlapping descriptions
tests/              Resolution-logic unit tests (no mic, model or GPU needed)
```

## Setup

Use Python **3.11 or 3.12**. `speech_recognition` imports `aifc`/`audioop`, which
were removed in Python 3.13. The install is large (torch, transformers, Whisper backends).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

`pyaudio` needs PortAudio headers to build: `sudo dnf install portaudio-devel`
(Fedora), `sudo apt install portaudio19-dev` (Debian/Ubuntu), or `brew install portaudio` (macOS).

### Importing the package

The package's top-level module is named `speak-up` (with a hyphen), so a normal
`import` statement can't load it. Use:

```python
import importlib
su = importlib.import_module("speak-up")
VoiceController, SemanticMapper = su.VoiceController, su.SemanticMapper
```

## Running

The microphone is the one on the **machine running the backend**. The browser
never captures audio; it only receives events. So run both on the same computer.

```bash
.venv/bin/python -m backend.app                    # voice + typed, backend from game_config.json
.venv/bin/python -m backend.app --backend cuda     # NVIDIA GPU
.venv/bin/python -m backend.app --typed-only       # no mic / Whisper; GAMEBERT only
```

Then open **http://localhost:8000**. The first run downloads Whisper and GAMEBERT.
**Stay quiet** while the start screen says "Calibrating microphone" (2 seconds).
Use the **Mic test** panel to see what was heard, which words were tagged, and the
top-3 moles before you play.

### Whisper backends

| `--backend` | Whisper engine | Notes |
|---|---|---|
| `mps`  | `mlx_whisper` | Apple Silicon. The package's own path. |
| `cuda` | `faster_whisper` float16 | NVIDIA. CTranslate2 needs **CUDA 12** cuBLAS/cuDNN. If you see `libcublas.so.12 is not found`, run `.venv/bin/pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`. The backend preloads them automatically. |
| `cpu`  | `faster_whisper` int8 | Added by `MoleVoiceController`; v1.0.1 itself raises `ValueError` for `"cpu"`. |

Measured with `tiny.en`, per utterance after warm-up: cpu ≈ 350–450 ms, cuda (RTX 4050) ≈ 45–100 ms.
GAMEBERT adds ≈ 3–6 ms. Recording continues until `speech_recognition` detects about
0.8 s of silence, rather than cutting off at a fixed phrase length. You can adjust the
silence interval with `voice.pause_threshold` in the config.

If Whisper fails on its first transcription (for example, missing CUDA libs), the
server does **not** start listening. It falls back to typed mode and shows the error.
If the mic reads silence during calibration (muted or wrong input device), you get a
"microphone looks silent" warning.

### Typed mode

Tick **Typed-input mode** on the start screen. It is also turned on automatically if the
mic or Whisper is unavailable. Typed commands go through `POST /match` and run the
same GAMEBERT + resolution path as speech.

## How matching works

For each utterance (`backend/resolver.py`):

1. **CORRECTION** entity → cancel the wind-up in progress, then keep going if a TARGET
   was also said ("no, the pirate!").
2. Query = **TARGET-tagged words only**, minus filler (`QUERY_STOPWORDS`). If there
   are none, use the full transcript with the lower `FALLBACK_*` thresholds.
3. Rank the visible moles with `RankedMapper.rank_targets`. This uses the same
   TfidfVectorizer + cosine method as the package, but returns every score.
4. `top1 < MIN_SCORE` → **no_match**. `top1 − top2 < AMBIGUITY_MARGIN` → **ambiguous**
   (both glow, no penalty). Otherwise → **whack** top1. The action ("whack" vs
   "defuse") comes from `rank_actions`.

GAMEBERT quirks, handled in code:

- With `aggregation_strategy="simple"`, words come back as subword fragments
  (`whack` → `##ha`, `##ck`; `spaceman` → `space`, `##man`). Words are rebuilt from
  the `start`/`end` offsets, never from `word`.
- A bare "stop" is tagged ACTION, so an utterance made only of
  `CORRECTION_KEYWORDS` also counts as a correction.
- It sometimes tags filler as CORRECTION ("get the bread **one**",
  "**it's** the pirate"). Spans made only of `CORRECTION_IGNORE_WORDS` are ignored.

### Timing

- A whack counts if the utterance **started** while the mole was up. The backend
  tracks spawn/retreat times from `round_state` messages.
- Wind-up: the hammer hangs over the mole for `WINDUP_MS` before striking. A
  correction that arrives before the strike cancels it with no penalty.
- Voice results land roughly 1–2 s after you start speaking. So a "no!" said *during*
  the wind-up, or within `REGRET_WINDOW_MS` after the strike, but transcribed after
  the strike, gets the **regret refund** (half the wrong-mole penalty back).
  Corrections are judged by when you started speaking, not by when they arrived.

## Tuning `game_config.json`

The backend writes the `targets`/`actions` mapping for `voice-speak-up` from this file
on startup (`backend/.generated_mapping.json`).

- **Descriptions are the game's vocabulary.** Matching is lexical, so a word only
  matches if it literally appears in a description. Include the label, useful
  synonyms, and distinctive visual or thematic clues; keep descriptions a similar
  length (about **9–12 words**). Scores depend on length (one word against an
  n-word description scores ≈ 1/√n), so uneven lengths make the tiers unfair.
- Words shared by visible moles lose weight. That is the point of **twins**. Shared
  words between non-twins make ambiguity more likely.
- Never use the id `enemy`; `SemanticMapper` always skips it.
- The end-of-round summary lists **words that never matched**. Add the good ones to
  descriptions.

Then run:

```bash
.venv/bin/python calibrate.py               # phrases in config, ranked against ALL moles (hardest case)
.venv/bin/python calibrate.py --visible 4   # ranked against 4 visible moles, like a real round
.venv/bin/python calibrate.py --phrase "smash the grumpy one=angry"
.venv/bin/python calibrate.py --overlap-only
```

### Previous threshold reference (v1.0.1, `tiny.en`, original 7–9-word descriptions)

| | measured | config |
|---|---|---|
| 1 TARGET word, correct mole | 0.27 – 0.33 | `MIN_SCORE` 0.20 (wrong moles score 0 unless they share a word) |
| 2 TARGET words | 0.38 – 0.42 | `tiers.Great` 0.30 |
| 3 TARGET words | ≈ 0.50 | `tiers.Perfect` 0.46 |
| full-transcript fallback | 0.13 – 0.25 | `FALLBACK_MIN_SCORE` 0.10, `FALLBACK_MARGIN` 0.04, `fallback_tiers.Great` 0.18 |

So the tiers reward describing a mole with more of its words. The lowered Great cutoff
makes a solid single-word description more likely to earn the Great bonus. A golden mole
still needs a Great tier to count as a hit.

These values differ from the starting defaults in the build brief
(MIN 0.25, Great 0.40, Perfect 0.55). Those defaults left single words against
9-word descriptions borderline, and made Perfect almost impossible. Re-run
`calibrate.py` whenever you change descriptions.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

The tests cover thresholds, the margin, the zero-score guard, fallback mode,
correction detection, wind-up / regret timing, mission-card tags, bomb / golden /
sign-word verdicts, the utterance-start tracker, and `RankedMapper` agreeing with
the package's own top-1.

## API

- `GET /health`: phase (`loading_models` / `calibrating` / `listening` / `typed_only` / `error`),
  model loaded, GAMEBERT `id2label`, backend, mic calibrated, and mic errors and warnings.
- `GET /config`: the game config.
- `POST /match {text, visible_moles?, mission_tag?, play?}`: GAMEBERT + resolution on
  typed text. When `visible_moles` is omitted, the live round state is used.
  `play: true` also runs the wind-up and broadcasts events over the WebSocket.
- `WS /ws`: server → client `utterance`, `whack` (`phase: windup | strike`),
  `ambiguous`, `no_match`, `correction` (`effect: cancelled | refund | superseded | none`),
  `status`. Client → server `round_state {visible_moles, mission_tag, input_locked}` and `reset`.
