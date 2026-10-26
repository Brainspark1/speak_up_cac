"""Helpers for GAMEBERT entity output (HF pipeline, aggregation_strategy="simple").

GAMEBERT's "simple" aggregation can split one word into several entities with
subword fragments: "whack" -> "##ha", "##ck" and "spaceman" -> "space", "##man".
So we never join e["word"]. We use the start/end offsets, widen each span to
the whole word in the transcript, and dedupe.
"""

import re

_WORD_RE = re.compile(r"[a-z0-9']+")


def _expand(text, start, end):
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "'"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] == "'"):
        end += 1
    return start, end


def group_words(text, entities, group):
    """Whole words in `text` tagged with entity_group == group, in order."""
    spans = []
    for e in entities:
        if e.get("entity_group") != group:
            continue
        start, end = e.get("start"), e.get("end")
        if start is None or end is None:
            word = str(e.get("word", "")).replace("##", "").strip()
            idx = text.find(word) if word else -1
            if idx < 0:
                continue
            start, end = idx, idx + len(word)
        span = _expand(text, int(start), int(end))
        if span not in spans:
            spans.append(span)
    words = []
    for start, end in sorted(spans):
        words.extend(_WORD_RE.findall(text[start:end].lower()))
    # dedupe while keeping order (overlapping spans can repeat a word)
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def tokens(text):
    return _WORD_RE.findall(text.lower())


def clean_entities(entities):
    """JSON-safe copies (pipeline scores are numpy float32)."""
    out = []
    for e in entities or []:
        out.append({
            "entity_group": e.get("entity_group"),
            "word": e.get("word"),
            "score": float(e.get("score", 0.0)),
            "start": int(e["start"]) if e.get("start") is not None else None,
            "end": int(e["end"]) if e.get("end") is not None else None,
        })
    return out


def fix_mishearings(text, fixes, skip_after=()):
    """Replace words Whisper commonly mishears ("more" -> "mole").

    fixes: {heard_word: intended_word}. A word right after one of skip_after
    ("one more", "no more") is left alone, since there it's a real word.
    Returns (fixed_text, [(heard, intended), ...]).
    """
    if not fixes or not text:
        return text, []
    skip = {w.lower() for w in skip_after}
    changed = []
    prev = [None]

    def repl(m):
        word = m.group(0)
        target = fixes.get(word.lower())
        before, prev[0] = prev[0], word.lower()
        if target is None or before in skip:
            return word
        changed.append((word.lower(), target))
        return target

    return re.sub(r"[A-Za-z']+", repl, text), changed
