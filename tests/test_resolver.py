"""Unit tests for strict command phrase parsing."""

from desktop_automation.resolver import DesktopCommandResolver


def test_scroll_phrase_resolves():
    command = DesktopCommandResolver().resolve("SCROLL DOWN!")
    assert command.action == "scroll"
    assert command.direction == "down"
    assert command.amount == "page"


def test_named_link_phrase_extracts_name():
    resolver = DesktopCommandResolver()
    assert resolver.resolve("click the Documentation link").link_name == "documentation"
    assert resolver.resolve("click on the About this demo link").link_name == "about this demo"


def test_click_without_specific_name_is_rejected():
    resolver = DesktopCommandResolver()
    assert resolver.resolve("click the link") is None
    assert resolver.resolve("click something") is None


def test_unknown_phrase_is_rejected():
    assert DesktopCommandResolver().resolve("maybe click that") is None


def test_stop_resolves():
    assert DesktopCommandResolver().resolve("stop").action == "stop"
