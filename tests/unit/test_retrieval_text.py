from jarvis_cli.retrieval.text import deslug, lexical_tokens


def test_deslug_handles_separators_and_camel():
    assert deslug("deploy-to-vercel") == "deploy to vercel"
    assert deslug("ckm:ui-styling") == "ckm ui styling"
    assert deslug("writeTests") == "write Tests"


def test_lexical_tokens_ascii():
    toks = lexical_tokens("deploy a Vercel app")
    assert "deploy" in toks
    assert "vercel" in toks
    assert "app" in toks
    assert "a" not in toks  # single ASCII char dropped


def test_lexical_tokens_cjk_bigrams():
    toks = lexical_tokens("帮我看下之前的会话")
    assert "帮我看下之前的会话" in toks  # full CJK run
    assert "会话" in toks  # bigram extracted
    assert "之前" in toks  # bigram extracted


def test_cjk_bigrams_enable_keyword_matching():
    query_toks = lexical_tokens("帮我看下之前的会话")
    keyword_toks = lexical_tokens("会话")
    assert query_toks & keyword_toks  # intersection is non-empty


def test_short_cjk_not_bigram_exploded():
    toks = lexical_tokens("测试")
    assert toks == {"测试"}  # 2-char CJK run: no bigrams to extract
