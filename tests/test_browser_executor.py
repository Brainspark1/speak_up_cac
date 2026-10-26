"""Playwright tests use local HTML content, never external websites."""

from playwright.sync_api import sync_playwright

from desktop_automation.browser_executor import BrowserExecutor
from desktop_automation.commands import DesktopCommand


class Approve:
    """Test policy that approves clicks without requiring terminal input."""
    def confirm_link_click(self, link_name: str) -> bool:
        return True


class Reject:
    """Test policy that refuses a click."""
    def confirm_link_click(self, link_name: str) -> bool:
        return False


def test_named_link_clicks_exact_match():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content("""
          <a href="#docs">Documentation</a>
          <a href="#other">Documentation guide</a>
          <div id="docs"></div><div id="other"></div>
        """)
        executor = BrowserExecutor(page, Approve())

        result = executor.execute(
            DesktopCommand(action="click_link", link_name="Documentation")
        )

        assert 'Clicked the link named "Documentation"' in result
        assert page.url.endswith("#docs")
        browser.close()


def test_duplicate_exact_names_are_rejected():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content('<a href="#one">Help</a><a href="#two">Help</a>')
        executor = BrowserExecutor(page, Approve())

        result = executor.execute(
            DesktopCommand(action="click_link", link_name="Help")
        )

        assert "Multiple links" in result
        assert page.url.endswith("about:blank")
        browser.close()


def test_configured_synonym_matches_link_name():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content('<a href="#step-one">Step One</a>')
        executor = BrowserExecutor(page, Approve())

        result = executor.execute(
            DesktopCommand(action="click_link", link_name="Step 2")
        )

        assert 'Clicked the link named "Step 2"' in result
        assert page.url.endswith("#step-one")
        browser.close()


def test_configured_synonym_rejects_multiple_matching_links():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(
            '<a href="#one">Step One</a><a href="#two">Step Two</a>'
        )
        executor = BrowserExecutor(page, Approve())

        result = executor.execute(
            DesktopCommand(action="click_link", link_name="Step 2")
        )

        assert "Multiple links" in result
        assert page.url.endswith("about:blank")
        browser.close()


def test_declined_confirmation_does_not_click():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content('<a href="#docs">Documentation</a><div id="docs"></div>')
        executor = BrowserExecutor(page, Reject())

        result = executor.execute(
            DesktopCommand(action="click_link", link_name="Documentation")
        )

        assert "cancelled" in result
        assert page.url.endswith("about:blank")
        browser.close()
