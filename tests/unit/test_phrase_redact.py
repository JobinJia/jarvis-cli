from jarvis_cli.phrase.redact import scrub, speakable


def test_scrub_replaces_home_path(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    assert scrub("rm -rf /Users/jobin/tmp/x") == "rm -rf ~/tmp/x"


def test_scrub_does_not_corrupt_when_home_is_root(monkeypatch):
    monkeypatch.setenv("HOME", "/")
    # Should NOT replace every '/' with '~'
    out = scrub("rm -rf /tmp/foo")
    assert out == "rm -rf /tmp/foo"


def test_scrub_redacts_openai_key():
    out = scrub("curl -H 'auth: sk-abcdef1234567890ABCDEF' x")
    assert "sk-abcdef" not in out
    assert "<REDACTED>" in out


def test_scrub_redacts_eleven_key():
    out = scrub("ELEVENLABS=sk_1234567890abcdefABCDEF")
    assert "<REDACTED>" in out


def test_scrub_redacts_github_pat():
    out = scrub("token ghp_abcdefghijklmnopqrstuvwxyz123456")
    assert "<REDACTED>" in out


def test_scrub_redacts_aws_key():
    out = scrub("AKIA1234567890ABCDEF foo")
    assert "<REDACTED>" in out


def test_scrub_redacts_slack_token():
    out = scrub("xoxb-12345-67890-abcdefgABCDEFG")
    assert "<REDACTED>" in out


def test_scrub_redacts_long_hex_token():
    out = scrub("commit deadbeef1234567890cafebabe1234567890aabbcc11")
    assert "<REDACTED>" in out
    assert "deadbeef" not in out


def test_scrub_truncates_to_200_chars():
    long = "x" * 500
    assert len(scrub(long)) == 200


def test_scrub_disabled_only_truncates():
    long = "/Users/jobin/" + "x" * 300
    out = scrub(long, enabled=False)
    assert len(out) == 200
    assert out.startswith("/Users/jobin/")  # HOME NOT replaced


def test_scrub_empty_string_is_empty():
    assert scrub("") == ""


# --- speakable: unspeakable ID shortening (the 2026-08-28 cicada buzz) ------


def test_speakable_shortens_uuid_to_stub():
    out = speakable("reference f140d0cf-bd61-4af4-a874-985f5fae898b lost")
    assert out == "reference f140 lost"


def test_speakable_shortens_long_hex_run():
    assert speakable("commit deadbeefcafe4321feed") == "commit dead"


def test_speakable_keeps_short_hex_and_words():
    text = "commit deadbeef on branch feature-cafe"
    assert speakable(text) == text


def test_speakable_shortens_figma_style_key():
    out = speakable("file key hp6VI6CyREYbtiukMR6BmN, depth 2")
    assert out == "file key hp6V, depth 2"


def test_speakable_shortens_epoch_ms_timestamp():
    assert speakable("at 1787933139368 exactly") == "at 1787 exactly"


def test_speakable_keeps_long_plain_words_and_small_numbers():
    text = "internationalization finished at 2026, node 2113-149"
    assert speakable(text) == text


def test_speakable_empty_string_is_empty():
    assert speakable("") == ""


def test_scrub_applies_speakable_even_when_disabled():
    out = scrub("session f140d0cf-bd61-4af4-a874-985f5fae898b", enabled=False)
    assert out == "session f140"


def test_scrub_secret_hex_still_redacted_not_stubbed():
    # 40+ hex chars is secret-shaped — must stay <REDACTED>, not a 4-char stub.
    out = scrub("commit deadbeef1234567890cafebabe1234567890aabbcc11")
    assert "<REDACTED>" in out
