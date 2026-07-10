"""Iron-Man-style opening briefing: greeting + time + weather.

Composes the line spoken on `session_start` events. No LLM round-trip —
the text is built from local clock + a wttr.in lookup, then handed
straight to TTS via the daemon's existing `event.text` bypass.

Design choices:
- English only. The voice clone is English (Jarvis); reading Chinese
  with the cloned voice loses the identity. See feedback memory.
- Weather is best-effort: a timeout, network error, or parse failure
  degrades to a time-only line rather than failing the briefing.
- Weather lookups are TTL-cached per-city so opening N sessions in a
  burst hits wttr.in once, not N times.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from .config import SessionBriefingConfig
from .phrase.providers.base import PhraseProvider
from .types import Lang

_WTTR_URL = "https://wttr.in/{city}?format=j1"


# --- number / time spelling -------------------------------------------------
# A tiny num2words is cheaper than pulling in a dep. Range covered: 0-59
# (minutes, day-of-month) plus 1-12 (clock hours).
_UNDER_20 = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty")
_HOURS = (
    "twelve", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven",
)
_ORDINAL_UNDER_20 = (
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
    "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth",
)
_ORDINAL_TENS = {20: "twentieth", 30: "thirtieth"}


def _num_word(n: int) -> str:
    if n < 20:
        return _UNDER_20[n]
    tens, units = divmod(n, 10)
    if units == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_UNDER_20[units]}"


def _ordinal_word(n: int) -> str:
    if n < 20:
        return _ORDINAL_UNDER_20[n]
    tens, units = divmod(n, 10)
    if units == 0:
        return _ORDINAL_TENS.get(n, f"{_TENS[tens]}ieth")
    return f"{_TENS[tens]}-{_ORDINAL_UNDER_20[units]}"


def _period(hour_24: int) -> str:
    if 5 <= hour_24 < 12:
        return "in the morning"
    if 12 <= hour_24 < 18:
        return "in the afternoon"
    if 18 <= hour_24 < 22:
        return "in the evening"
    return "at night"


def _greeting(hour_24: int) -> str:
    if 5 <= hour_24 < 12:
        return "Good morning, sir."
    if 12 <= hour_24 < 18:
        return "Good afternoon, sir."
    if 18 <= hour_24 < 22:
        return "Good evening, sir."
    return "A late hour, sir."


def _format_time(now: datetime) -> str:
    """Render a clock time in natural English: 'half past ten in the evening'.

    Special-cases the hourly landmarks (o'clock / quarter past / half past /
    quarter to). The TTS handles word-only input far more consistently than
    digit-laden strings like "22:31".
    """
    h12 = now.hour % 12
    m = now.minute
    period = _period(now.hour)
    if m == 0:
        return f"{_HOURS[h12]} o'clock {period}"
    if m == 15:
        return f"a quarter past {_HOURS[h12]} {period}"
    if m == 30:
        return f"half past {_HOURS[h12]} {period}"
    if m == 45:
        next_h = (h12 + 1) % 12
        return f"a quarter to {_HOURS[next_h]} {period}"
    # Off-landmark: spell both halves. Minutes 1-9 take "oh" so the TTS
    # doesn't run "eleven three" together as a single number.
    minute = f"oh {_num_word(m)}" if 1 <= m < 10 else _num_word(m)
    return f"{_HOURS[h12]} {minute} {period}"


def _format_date(now: datetime) -> str:
    """E.g. 'Sunday, the twenty-fourth of May'."""
    weekday = now.strftime("%A")
    month = now.strftime("%B")
    day_word = _ordinal_word(now.day)
    return f"{weekday}, the {day_word} of {month}"


# --- weather ----------------------------------------------------------------


@dataclass(frozen=True)
class WeatherSnapshot:
    city: str
    temp_c: int
    condition: str
    humidity: int
    wind_kph: int
    wind_dir: str  # wttr.in's 16-point compass abbreviation, e.g. "ESE"


_COMPASS = {
    "N": "north", "NNE": "north-northeast", "NE": "northeast",
    "ENE": "east-northeast", "E": "east", "ESE": "east-southeast",
    "SE": "southeast", "SSE": "south-southeast", "S": "south",
    "SSW": "south-southwest", "SW": "southwest", "WSW": "west-southwest",
    "W": "west", "WNW": "west-northwest", "NW": "northwest",
    "NNW": "north-northwest",
}


def _wind_phrase(kph: int) -> str:
    if kph < 5:
        return "still air"
    if kph < 12:
        return "a gentle breeze"
    if kph < 25:
        return "a brisk wind"
    return "a strong wind"


def _format_weather(w: WeatherSnapshot) -> str:
    direction = _COMPASS.get(w.wind_dir, w.wind_dir.lower())
    if w.wind_kph < 5:
        wind = "still air outside"
    else:
        wind = f"{_wind_phrase(w.wind_kph)} from the {direction}"
    return (
        f"In {w.city} it is {w.temp_c} degrees Celsius and {w.condition.lower()}, "
        f"with humidity at {w.humidity} percent and {wind}."
    )


def detect_city() -> str:
    """Derive a queryable city name from `/etc/localtime`.

    macOS symlinks `/etc/localtime` into `…/zoneinfo/<Region>/<City>`;
    we take the tail. Falls back to 'Shanghai' on any read error so we
    always have *something* to hand to wttr.in.

    We deliberately don't trust wttr.in's IP geolocation — VPNs throw
    it off (we've seen 'Tokyo' returned from a Shanghai connection
    routed through a JP exit).
    """
    try:
        tz_target = os.readlink("/etc/localtime")
        tail = tz_target.rsplit("/", 1)[-1]
        return tail or "Shanghai"
    except OSError:
        return "Shanghai"


async def _fetch_weather_raw(city: str, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(_WTTR_URL.format(city=city))
        r.raise_for_status()
        return r.json()


def _parse_weather(data: dict[str, Any], fallback_city: str) -> WeatherSnapshot:
    current = data["current_condition"][0]
    area = data["nearest_area"][0]
    city = area["areaName"][0]["value"] or fallback_city
    return WeatherSnapshot(
        city=city,
        temp_c=int(current["temp_C"]),
        condition=current["weatherDesc"][0]["value"].strip(),
        humidity=int(current["humidity"]),
        wind_kph=int(current["windspeedKmph"]),
        wind_dir=current["winddir16Point"],
    )


async def _fetch_wttr(city: str, timeout: float) -> WeatherSnapshot | None:
    """wttr.in source. Returns None on any failure so `fetch_weather` can fall
    through to the next source rather than raising."""
    try:
        data = await _fetch_weather_raw(city, timeout)
        return _parse_weather(data, fallback_city=city)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        logger.warning("wttr.in fetch failed for {!r}: {}", city, exc)
        return None


# --- open-meteo source (free, no API key) -----------------------------------

_OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → short, TTS-friendly descriptions.
# See the "WMO Weather interpretation codes (WW)" table at open-meteo.com/en/docs.
_WMO_DESC: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

_COMPASS_16 = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


def _wmo_condition(code: int) -> str:
    """Map a WMO weather code to a spoken-English description."""
    return _WMO_DESC.get(code, "unsettled weather")


def _deg_to_compass(deg: float) -> str:
    """Wind bearing in degrees → 16-point compass abbreviation (e.g. 229→SW),
    which `_format_weather` already expands to words via `_COMPASS`."""
    return _COMPASS_16[int((deg % 360) / 22.5 + 0.5) % 16]


def _parse_open_meteo(
    data: dict[str, Any], loc: dict[str, Any], fallback_city: str,
) -> WeatherSnapshot:
    cur = data["current"]
    return WeatherSnapshot(
        city=loc.get("name") or fallback_city,
        temp_c=round(float(cur["temperature_2m"])),
        condition=_wmo_condition(int(cur["weather_code"])),
        humidity=int(cur["relative_humidity_2m"]),
        wind_kph=round(float(cur["wind_speed_10m"])),
        wind_dir=_deg_to_compass(float(cur["wind_direction_10m"])),
    )


async def _fetch_open_meteo(city: str, timeout: float) -> WeatherSnapshot | None:
    """open-meteo source: geocode the city to coordinates, then read its
    current conditions. Returns None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            geo = await client.get(
                _OPEN_METEO_GEOCODE_URL, params={"name": city, "count": 1},
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return None
            loc = results[0]
            fc = await client.get(
                _OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,weather_code,"
                        "wind_speed_10m,wind_direction_10m"
                    ),
                    "wind_speed_unit": "kmh",
                },
            )
            fc.raise_for_status()
            return _parse_open_meteo(fc.json(), loc, fallback_city=city)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        logger.warning("open-meteo fetch failed for {!r}: {}", city, exc)
        return None


# Sources tried in order; the first to return a snapshot wins. wttr.in first
# (no geocoding round-trip needed), open-meteo as a robust fallback. Names, not
# function objects, so `fetch_weather` resolves them at call time — keeps the
# list monkeypatch-friendly in tests and trivial to extend with more sources.
_WEATHER_SOURCES: tuple[str, ...] = ("_fetch_wttr", "_fetch_open_meteo")


async def fetch_weather(city: str, timeout: float) -> WeatherSnapshot | None:
    """Try each weather source in order; return the first successful snapshot,
    or None if all fail (so callers degrade to a time-only briefing)."""
    for name in _WEATHER_SOURCES:
        source = globals()[name]
        snap = await source(city, timeout)
        if snap is not None:
            return snap
    return None


class WeatherCache:
    """Per-city TTL cache with a fetch lock so simultaneous briefings
    don't fan out into N parallel wttr.in calls."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = ttl_seconds
        self._entries: dict[str, tuple[float, WeatherSnapshot | None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _fresh(self, city: str, now: float) -> tuple[bool, WeatherSnapshot | None]:
        entry = self._entries.get(city)
        if entry and now - entry[0] < self.ttl:
            return True, entry[1]
        return False, None

    async def get(self, city: str, timeout: float) -> WeatherSnapshot | None:
        now = time.monotonic()
        ok, snap = self._fresh(city, now)
        if ok:
            return snap
        lock = self._locks.setdefault(city, asyncio.Lock())
        async with lock:
            # Double-check after acquiring the lock — another coroutine
            # may have populated the cache while we were waiting.
            ok, snap = self._fresh(city, time.monotonic())
            if ok:
                return snap
            snap = await fetch_weather(city, timeout)
            self._entries[city] = (time.monotonic(), snap)
            return snap


# --- offline template variants ---------------------------------------------
# Each new session should sound a touch different. When the LLM path is
# unreachable we rotate among these instead of always reading the same
# canned line. Keep them all in Jarvis voice (deferential, dry, English).

# Slot vocab (all already English prose):
#   {greeting}  → "Good evening, sir." / "A late hour, sir." / etc.
#   {time}      → "half past ten at night"
#   {date}      → "Sunday, the twenty-fourth of May"
#   {weather}   → full sentence; capitalized; ends with period.
#   {sign_off}  → randomized closing nudge ("How may I assist?", ...)
_TEMPLATE_VARIANTS_WITH_WEATHER: tuple[str, ...] = (
    "{greeting} The local time is {time}, {date}. {weather} {sign_off}",
    "{greeting} It is {time} on {date}. {weather} {sign_off}",
    "{greeting} The clock reads {time}; {date}. {weather} {sign_off}",
    "{greeting} {weather} The hour is {time}, {date}. {sign_off}",
    "{greeting} {time}, {date}. {weather} {sign_off}",
    "{greeting} Just past the {time} mark on {date}. {weather} {sign_off}",
)

_TEMPLATE_VARIANTS_NO_WEATHER: tuple[str, ...] = (
    "{greeting} The local time is {time}, {date}. {sign_off}",
    "{greeting} It is {time}, {date}. {sign_off}",
    "{greeting} The clock reads {time} on {date}. {sign_off}",
    "{greeting} {time}, {date}. {sign_off}",
)

_SIGN_OFFS: tuple[str, ...] = (
    "How may I be of service?",
    "How may I assist?",
    "At your service.",
    "Shall we begin?",
    "Ready when you are, sir.",
    "What is first on the agenda?",
    "Awaiting your instructions.",
    "Where shall we start?",
)


def _render_template(
    *, greeting: str, time_phrase: str, date_phrase: str,
    weather: WeatherSnapshot | None, rng: random.Random,
) -> str:
    sign_off = rng.choice(_SIGN_OFFS)
    if weather is not None:
        tmpl = rng.choice(_TEMPLATE_VARIANTS_WITH_WEATHER)
        return tmpl.format(
            greeting=greeting, time=time_phrase, date=date_phrase,
            weather=_format_weather(weather), sign_off=sign_off,
        ).strip()
    tmpl = rng.choice(_TEMPLATE_VARIANTS_NO_WEATHER)
    return tmpl.format(
        greeting=greeting, time=time_phrase, date=date_phrase, sign_off=sign_off,
    ).strip()


# --- LLM-phrased path ------------------------------------------------------


_LLM_SYSTEM_PROMPT_TEMPLATE = (
    "You are JARVIS, Tony Stark's British AI butler from the Iron Man films. "
    "Speak in his original cadence: brief, dry, deferential, never sycophantic. "
    "Address the user as '{addr}'. Compose ONE opening line for a new session. "
    "{humor}\n"
    "Hard rules:\n"
    "• English only. No markdown. No quotes around your response.\n"
    "• One or two sentences. Total under 60 words.\n"
    "• Mention the local time naturally — paraphrase, do not read digits.\n"
    "• HONOR the provided time-of-day (morning / afternoon / evening / "
    "night). NEVER invent a different period — saying 'good afternoon' "
    "at 2 AM is a failure.\n"
    "• Mention the weather and city naturally if context is provided.\n"
    "• End with a quiet invitation to begin (e.g. 'How may I assist?', "
    "'Shall we?', 'At your service.', 'Where shall we start?').\n"
    "• Vary your phrasing every call — never repeat the same opening twice."
)

# Mirror of phrase/prompt.py:_HUMOR_CLAUSES, tuned for the briefing context.
# Indexed by humor_level (0-3); out-of-range values clamp.
_BRIEFING_HUMOR_CLAUSES: tuple[str, ...] = (
    "Tone: strictly deadpan — no jokes, no asides, no flourishes.",
    "Tone: calm and courteous, with a hint of dry wit when natural.",
    "Tone: dry, banter-prone wit in the MCU Jarvis register — a wry "
    "observation about the hour or the weather is welcome when it fits.",
    "Tone: openly sardonic, in the manner of an old butler who has seen "
    "this all before — light teasing is welcome, sycophancy is not.",
)


def _briefing_humor_clause(level: int) -> str:
    return _BRIEFING_HUMOR_CLAUSES[
        max(0, min(len(_BRIEFING_HUMOR_CLAUSES) - 1, level))
    ]


def _llm_user_prompt(
    *, greeting: str, time_phrase: str, date_phrase: str,
    weather: WeatherSnapshot | None,
    now: datetime,
) -> str:
    """Hand the LLM ready-made English fragments — no number formatting or
    timezone math on its end. It just has to weave them into a varied line.

    `now` is passed in so we can hand the LLM an unambiguous 24-hour clock
    and explicit period-of-day label. Without those, qwen3:8b sometimes
    invents 'good afternoon' at 2 AM.
    """
    period = _period(now.hour).replace("in the ", "").replace("at ", "")
    lines = [
        f"Greeting (use verbatim or honor its time period): {greeting}",
        f"Time of day (in words): {time_phrase}",
        f"Exact 24-hour clock: {now.hour:02d}:{now.minute:02d}",
        f"Period of day: {period}   ← do NOT contradict this",
        f"Date (in words): {date_phrase}",
    ]
    if weather is not None:
        direction = _COMPASS.get(weather.wind_dir, weather.wind_dir.lower())
        lines.append(f"City: {weather.city}")
        lines.append(f"Temperature: {weather.temp_c} degrees Celsius")
        lines.append(f"Conditions: {weather.condition.lower()}")
        lines.append(f"Humidity: {weather.humidity} percent")
        lines.append(f"Wind: {_wind_phrase(weather.wind_kph)} from the {direction}")
    else:
        lines.append("(Weather unavailable — speak only of the time.)")
    lines.append("\nCompose the opening line now.")
    return "\n".join(lines)


def _is_usable_briefing(text: str) -> bool:
    """Reject obvious junk so we don't speak the LLM's failure modes."""
    if not text or len(text) < 12 or len(text) > 400:
        return False
    # Strip wrapping quotes the model sometimes adds despite instructions.
    stripped = text.strip().strip('"').strip("'").strip()
    if not stripped:
        return False
    return True


# Substrings we look for when verifying the LLM honored the period of day.
# Words like 'morning star' or 'evening dress' would false-positive a naive
# `in text` check, so we anchor on the actual greeting/phrasing patterns
# the model uses: "good <period>" and "in the <period>" / "this <period>".
_PERIOD_PATTERNS: dict[str, tuple[str, ...]] = {
    "morning":   ("good morning",   "this morning",   "in the morning"),
    "afternoon": ("good afternoon", "this afternoon", "in the afternoon"),
    "evening":   ("good evening",   "this evening",   "in the evening"),
    "night":     ("good night",     "tonight",        "at night", "late hour"),
}


def _contradicts_period(text: str, expected_period: str) -> bool:
    """True iff the LLM output names a DIFFERENT period of day than the
    one we asked for. qwen3:8b sometimes greets with 'Good afternoon' at
    2 AM despite explicit instructions; we catch that here and let the
    caller fall through to the (accurate) offline template path."""
    lower = text.lower()
    for period, patterns in _PERIOD_PATTERNS.items():
        if period == expected_period:
            continue
        if any(p in lower for p in patterns):
            return True
    return False


def _clean_llm_output(text: str) -> str:
    """Trim wrappers the model occasionally adds (quotes, leading 'Jarvis:')."""
    out = text.strip()
    if out.startswith(("Jarvis:", "JARVIS:", "Response:", "Output:")):
        out = out.split(":", 1)[1].strip()
    if (out.startswith('"') and out.endswith('"')) or \
            (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()
    return out


async def _compose_briefing_via_llm(
    provider: PhraseProvider,
    *,
    greeting: str, time_phrase: str, date_phrase: str,
    weather: WeatherSnapshot | None,
    humor_level: int,
    now: datetime,
    address: str = "sir",
) -> str | None:
    system = _LLM_SYSTEM_PROMPT_TEMPLATE.format(
        humor=_briefing_humor_clause(humor_level),
        addr=address,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _llm_user_prompt(
            greeting=greeting, time_phrase=time_phrase,
            date_phrase=date_phrase, weather=weather, now=now,
        )},
    ]
    try:
        raw = await provider.generate(messages)
    except Exception as exc:  # noqa: BLE001 — never fail the briefing
        logger.warning("briefing LLM ({}) failed: {}", provider.name, exc)
        return None
    cleaned = _clean_llm_output(raw)
    if not _is_usable_briefing(cleaned):
        logger.warning("briefing LLM returned unusable output: {!r}", raw[:120])
        return None
    expected_period = _period(now.hour).replace("in the ", "").replace("at ", "")
    if _contradicts_period(cleaned, expected_period):
        logger.warning(
            "briefing LLM contradicted period of day (expected {!r}); "
            "falling back to template. Output: {!r}",
            expected_period, cleaned[:120],
        )
        return None
    return cleaned


# --- public entrypoint ------------------------------------------------------


async def compose_briefing(
    cfg: SessionBriefingConfig,
    *,
    now: datetime | None = None,
    cache: WeatherCache | None = None,
    llm: PhraseProvider | None = None,
    rng: random.Random | None = None,
    humor_level: int = 1,
    address: str = "sir",
) -> tuple[str, Lang]:
    """Build the spoken text + language for a session_start event.

    Always returns ('<line>', 'en') — the Jarvis voice clone is English.

    Strategy: if `llm` is provided, ask it to phrase a fresh line (each call
    gets a different one). If that fails or no LLM is wired, fall back to
    rotating among hand-written template variants so we still vary by call.

    `humor_level` (0-3) shapes the LLM tone; `address` is how the line
    addresses the user. Templates ignore both — they're fixed phrasings,
    calibrated to land in the level-1 "sir" register.
    """
    now = now or datetime.now()
    rng = rng or random.Random()
    greeting = _greeting(now.hour)
    time_phrase = _format_time(now)
    date_phrase = _format_date(now)

    city = cfg.city or detect_city()
    weather: WeatherSnapshot | None = None
    if cache is not None:
        weather = await cache.get(city, cfg.weather_timeout_seconds)

    if llm is not None:
        line = await _compose_briefing_via_llm(
            llm,
            greeting=greeting, time_phrase=time_phrase,
            date_phrase=date_phrase, weather=weather,
            humor_level=humor_level, now=now, address=address,
        )
        if line:
            return line, "en"

    return _render_template(
        greeting=greeting, time_phrase=time_phrase, date_phrase=date_phrase,
        weather=weather, rng=rng,
    ), "en"
