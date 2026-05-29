"""Unit tests for the Iron-Man-style session_start briefing composer."""
from __future__ import annotations

import random
from datetime import datetime

import httpx
import pytest

from jarvis_cli.briefing import (
    WeatherCache,
    WeatherSnapshot,
    _clean_llm_output,
    _format_date,
    _format_time,
    _format_weather,
    _greeting,
    _is_usable_briefing,
    compose_briefing,
)
from jarvis_cli.config import SessionBriefingConfig
from jarvis_cli.phrase.providers.base import PhraseProvider


# --- greeting buckets -------------------------------------------------------


@pytest.mark.parametrize(
    "hour,expected_starts",
    [
        (6, "Good morning"),
        (11, "Good morning"),
        (12, "Good afternoon"),
        (17, "Good afternoon"),
        (18, "Good evening"),
        (21, "Good evening"),
        (22, "A late hour"),
        (3, "A late hour"),
    ],
)
def test_greeting_buckets(hour: int, expected_starts: str) -> None:
    assert _greeting(hour).startswith(expected_starts)


# --- time / date formatting -------------------------------------------------


@pytest.mark.parametrize(
    "h,m,phrase",
    [
        (10, 0, "ten o'clock in the morning"),
        (14, 15, "a quarter past two in the afternoon"),
        (22, 30, "half past ten at night"),
        (17, 45, "a quarter to six in the afternoon"),
        (8, 37, "eight thirty-seven in the morning"),
        (23, 3, "eleven oh three at night"),  # single-digit minute → "oh"
        (0, 5, "twelve oh five at night"),  # midnight cluster
        (12, 5, "twelve oh five in the afternoon"),  # noon cluster
    ],
)
def test_format_time(h: int, m: int, phrase: str) -> None:
    assert _format_time(datetime(2026, 5, 24, h, m)) == phrase


def test_format_date_uses_ordinal_word() -> None:
    assert _format_date(datetime(2026, 5, 24)) == "Sunday, the twenty-fourth of May"
    assert _format_date(datetime(2026, 5, 1)) == "Friday, the first of May"
    assert _format_date(datetime(2026, 5, 31)) == "Sunday, the thirty-first of May"


# --- weather format ---------------------------------------------------------


def test_format_weather_includes_city_temp_humidity_and_wind() -> None:
    w = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Partly cloudy",
        humidity=100, wind_kph=16, wind_dir="ESE",
    )
    text = _format_weather(w)
    assert "Shanghai" in text
    assert "22 degrees" in text
    assert "partly cloudy" in text  # lower-cased so it flows in prose
    assert "100 percent" in text
    assert "east-southeast" in text


def test_format_weather_collapses_to_still_air_under_calm_wind() -> None:
    w = WeatherSnapshot(
        city="X", temp_c=10, condition="Clear", humidity=50,
        wind_kph=2, wind_dir="N",
    )
    assert "still air" in _format_weather(w)


# --- weather cache ----------------------------------------------------------


class _FakeFetch:
    """Counts how many times the underlying fetcher is invoked."""

    def __init__(self, snap: WeatherSnapshot | None) -> None:
        self.snap = snap
        self.calls = 0

    async def __call__(self, city: str, timeout: float) -> WeatherSnapshot | None:
        self.calls += 1
        return self.snap


@pytest.mark.asyncio
async def test_weather_cache_serves_fresh_value_without_refetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Clear", humidity=70,
        wind_kph=10, wind_dir="E",
    )
    fake = _FakeFetch(snap)
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", fake)

    cache = WeatherCache(ttl_seconds=600)
    a = await cache.get("Shanghai", timeout=1.0)
    b = await cache.get("Shanghai", timeout=1.0)

    assert a is snap and b is snap
    assert fake.calls == 1, "TTL window should suppress the second fetch"


@pytest.mark.asyncio
async def test_weather_cache_refetches_after_ttl_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Clear", humidity=70,
        wind_kph=10, wind_dir="E",
    )
    fake = _FakeFetch(snap)
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", fake)

    # ttl=0 means every call is stale → fetcher is invoked each time.
    cache = WeatherCache(ttl_seconds=0)
    await cache.get("Shanghai", timeout=1.0)
    await cache.get("Shanghai", timeout=1.0)
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_weather_cache_caches_none_so_outages_dont_flood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When wttr.in is down, we still cache the None result for the TTL
    window — otherwise every session_start would re-hit the dead service."""
    fake = _FakeFetch(None)
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", fake)
    cache = WeatherCache(ttl_seconds=600)
    await cache.get("Shanghai", timeout=1.0)
    await cache.get("Shanghai", timeout=1.0)
    assert fake.calls == 1


# --- end-to-end composer ----------------------------------------------------


@pytest.mark.asyncio
async def test_compose_briefing_includes_greeting_time_date_and_weather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Partly cloudy",
        humidity=100, wind_kph=16, wind_dir="ESE",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    text, lang = await compose_briefing(
        cfg, now=datetime(2026, 5, 24, 22, 30), cache=cache,
    )

    assert lang == "en"
    assert text.startswith("A late hour, sir.")
    assert "half past ten at night" in text
    assert "Sunday, the twenty-fourth of May" in text
    assert "Shanghai" in text
    assert "22 degrees" in text


@pytest.mark.asyncio
async def test_compose_briefing_degrades_to_time_only_when_weather_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(None))

    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    text, _ = await compose_briefing(
        cfg, now=datetime(2026, 5, 24, 9, 0), cache=cache,
    )

    assert text.startswith("Good morning, sir.")
    assert "ten o'clock" not in text  # sanity: don't crash with weather missing
    assert "degrees" not in text  # weather phrase completely omitted
    # Closing nudge is randomized across a small set — assert any one fired.
    from jarvis_cli.briefing import _SIGN_OFFS
    assert any(s in text for s in _SIGN_OFFS), text


# --- template rotation (offline variety) -----------------------------------


class _FixedSeqProvider(PhraseProvider):
    """Returns canned strings in order; used to exercise the LLM path
    without standing up Ollama."""

    name = "fake"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.outputs:
            return ""
        return self.outputs.pop(0)


@pytest.mark.asyncio
async def test_offline_briefings_rotate_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an LLM, successive calls must read differently — the user
    explicitly asked for variation. A seeded RNG with many iterations
    proves the templates aren't all collapsing onto one variant."""
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Partly cloudy",
        humidity=80, wind_kph=10, wind_dir="ESE",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    rng = random.Random(42)
    seen: set[str] = set()
    for _ in range(20):
        text, _ = await compose_briefing(
            cfg, now=datetime(2026, 5, 25, 9, 0), cache=cache, rng=rng,
        )
        seen.add(text)
    assert len(seen) >= 3, f"expected variety, got {len(seen)} unique line(s)"


@pytest.mark.asyncio
async def test_llm_briefing_passes_humor_clause_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user's humor_level config must reach the LLM — otherwise
    setting it does nothing observable. Inspect the system message we
    actually sent."""
    snap = WeatherSnapshot(
        city="X", temp_c=10, condition="Clear", humidity=50,
        wind_kph=2, wind_dir="N",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    llm = _FixedSeqProvider(["Sir, the watch reads ten — shall we?"])
    cfg = SessionBriefingConfig(city="X")
    cache = WeatherCache(ttl_seconds=600)
    await compose_briefing(
        cfg, now=datetime(2026, 5, 25, 10, 0), cache=cache, llm=llm,
        humor_level=3,
    )
    sys_msg = llm.calls[0][0]["content"]
    assert "sardonic" in sys_msg.lower()

    # Confirm a different humor_level produces a different system prompt.
    llm2 = _FixedSeqProvider(["Sir, deadpan greeting."])
    await compose_briefing(
        cfg, now=datetime(2026, 5, 25, 10, 0), cache=cache, llm=llm2,
        humor_level=0,
    )
    sys_msg_0 = llm2.calls[0][0]["content"]
    assert "deadpan" in sys_msg_0.lower()
    assert sys_msg != sys_msg_0


@pytest.mark.asyncio
async def test_llm_briefing_takes_precedence_when_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Clear", humidity=60,
        wind_kph=5, wind_dir="N",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    llm = _FixedSeqProvider([
        "Good morning, sir. The clock has just struck nine; Shanghai is clear and twenty-two degrees. Where shall we begin?",
    ])
    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    text, lang = await compose_briefing(
        cfg, now=datetime(2026, 5, 25, 9, 0), cache=cache, llm=llm,
    )
    assert lang == "en"
    assert "Shanghai" in text
    assert "Where shall we begin" in text
    # Must have actually called the LLM, not used a template.
    assert len(llm.calls) == 1
    # System prompt mentions JARVIS identity so we know it was wired right.
    assert "JARVIS" in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_llm_briefing_falls_back_to_template_on_unusable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the LLM returns empty / garbled output, we must still speak
    *something* — fall through to the rotating templates."""
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Clear", humidity=60,
        wind_kph=5, wind_dir="N",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    llm = _FixedSeqProvider([""])  # empty → unusable
    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    text, _ = await compose_briefing(
        cfg, now=datetime(2026, 5, 25, 9, 0), cache=cache, llm=llm,
        rng=random.Random(0),
    )
    # One of the offline templates must have rendered.
    assert text.startswith("Good morning, sir.")
    assert "Shanghai" in text


@pytest.mark.asyncio
async def test_llm_briefing_falls_back_when_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = WeatherSnapshot(
        city="X", temp_c=10, condition="Clear", humidity=50,
        wind_kph=2, wind_dir="N",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    class _Boom(PhraseProvider):
        name = "boom"
        async def generate(self, messages):  # type: ignore[override]
            raise RuntimeError("ollama is down")

    cfg = SessionBriefingConfig(city="X")
    cache = WeatherCache(ttl_seconds=600)
    text, _ = await compose_briefing(
        cfg, now=datetime(2026, 5, 25, 9, 0), cache=cache, llm=_Boom(),
        rng=random.Random(0),
    )
    # The exception must not propagate; the briefing degrades to a template.
    assert "How may" in text or "Shall we" in text or "service" in text \
        or "Awaiting" in text or "Where shall" in text or "agenda" in text \
        or "Ready" in text


@pytest.mark.asyncio
async def test_llm_briefing_rejected_when_it_contradicts_time_of_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user surfaced this on 2026-05-25: qwen3:8b said 'Good afternoon'
    at 02:36 local. We must catch this server-side and fall back to the
    (accurate) template rather than speaking the wrong period."""
    snap = WeatherSnapshot(
        city="Shanghai", temp_c=22, condition="Clear", humidity=60,
        wind_kph=5, wind_dir="N",
    )
    monkeypatch.setattr("jarvis_cli.briefing.fetch_weather", _FakeFetch(snap))

    llm = _FixedSeqProvider([
        "Good afternoon, sir. It's a pleasant day in Shanghai. Shall we?",
    ])
    cfg = SessionBriefingConfig(city="Shanghai")
    cache = WeatherCache(ttl_seconds=600)
    text, _ = await compose_briefing(
        cfg,
        now=datetime(2026, 5, 25, 2, 30),  # 02:30 — definitely night
        cache=cache, llm=llm, rng=random.Random(0),
    )
    # The contradictory LLM line must NOT reach the user.
    assert "afternoon" not in text.lower()
    # The template fallback path opens with the late-hour greeting.
    assert text.startswith("A late hour, sir.")


def test_contradicts_period_catches_wrong_period_greetings():
    from jarvis_cli.briefing import _contradicts_period

    assert _contradicts_period("Good afternoon, sir.", "night") is True
    assert _contradicts_period("Good morning, sir.", "night") is True
    assert _contradicts_period("Good evening, sir.", "night") is True
    # Correct period passes through.
    assert _contradicts_period("A late hour, sir. Shall we?", "night") is False
    assert _contradicts_period("Good morning, sir.", "morning") is False
    # Substring false-positive guard: 'morning star' must not trip the
    # 'morning' check, but the test really matters for the common case.
    # Our anchored patterns ('good morning', 'this morning') avoid this.
    assert _contradicts_period(
        "Good evening — the morning star is up.", "evening",
    ) is False


def test_clean_llm_output_strips_quotes_and_label_prefix() -> None:
    assert _clean_llm_output('"Good evening, sir. Shall we?"') == "Good evening, sir. Shall we?"
    assert _clean_llm_output("Jarvis: At your service, sir.") == "At your service, sir."
    assert _clean_llm_output("'A pleasure, sir.'") == "A pleasure, sir."


@pytest.mark.parametrize("bad", ["", "   ", "x", "no", "a" * 401])
def test_is_usable_briefing_rejects_obvious_junk(bad: str) -> None:
    assert _is_usable_briefing(bad) is False


def test_is_usable_briefing_accepts_normal_lines() -> None:
    assert _is_usable_briefing("Good morning, sir. How may I assist?") is True


@pytest.mark.asyncio
async def test_fetch_weather_returns_none_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network/HTTP failures must not raise — they must degrade to None
    so the briefing can fall back to a time-only line."""
    from jarvis_cli import briefing

    class _BoomTransport(httpx.MockTransport):
        def __init__(self) -> None:
            super().__init__(lambda req: (_ for _ in ()).throw(
                httpx.ConnectError("nope"),
            ))

    async def _explode(city: str, timeout: float) -> dict:
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(briefing, "_fetch_weather_raw", _explode)
    assert await briefing.fetch_weather("Shanghai", timeout=0.1) is None
