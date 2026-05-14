from pathlib import Path

from jarvis_cc.phrase.language import detect_for


def test_detect_returns_zh_when_cwd_has_chinese_claude_md(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "本项目是一个使用 Valaxy 的中文博客。" * 10
    )
    assert detect_for(str(tmp_path)) == "zh"


def test_detect_returns_en_when_cwd_has_english_claude_md(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "This project is a personal blog built with Valaxy." * 10
    )
    assert detect_for(str(tmp_path)) == "en"


def test_detect_falls_back_to_readme_when_no_claude_md(tmp_path: Path):
    (tmp_path / "README.md").write_text("中文项目说明" * 10)
    assert detect_for(str(tmp_path)) == "zh"


def test_detect_defaults_to_en_when_no_signal(tmp_path: Path):
    # Empty dir → fall back to English (Jarvis's native tongue)
    assert detect_for(str(tmp_path)) == "en"


def test_detect_handles_none_cwd():
    assert detect_for(None) == "en"
