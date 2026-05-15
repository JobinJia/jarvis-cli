from jarvis_cc.phrase.redact import scrub


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
