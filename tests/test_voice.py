from backend.config import load_config
from backend.voice import MoleVoiceController


def test_long_utterances_are_not_limited_by_config_or_callback():
    config = load_config()
    assert config["voice"].get("phrase_time_limit") is None

    transcript = "please whack the very cheerful mole carrying a bright yellow trophy over there"
    events = []
    controller = MoleVoiceController.__new__(MoleVoiceController)
    controller.audio_to_numpy = lambda audio: [0.0] * 16000
    controller.transcribe_audio = lambda audio: transcript
    controller.extract_entities = lambda text: []
    controller.transcript_fixes = {}
    controller.transcript_fix_skip_after = ()
    controller.emit = events.append

    controller.audio_callback(None, None)

    assert len(events) == 1
    assert events[0]["transcript"] == transcript
    assert events[0]["dropped"] is None
