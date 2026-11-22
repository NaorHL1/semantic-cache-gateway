from semantic_cache_gateway.canonical import canonicalize

BASE = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}

def test_unknown_parameter_changes_hash():
    # Fail-closed: a parameter we have never seen must produce a miss.
    assert canonicalize(BASE).exact_hash != canonicalize({**BASE, "verbosity": "low"}).exact_hash

def test_denylisted_fields_do_not_change_hash():
    noisy = {**BASE, "stream": True, "user": "u1", "max_tokens": 50, "metadata": {"a": 1}}
    assert canonicalize(BASE).exact_hash == canonicalize(noisy).exact_hash

def test_key_order_does_not_change_hash():
    reordered = {"messages": BASE["messages"], "model": "gpt-4o"}
    assert canonicalize(BASE).exact_hash == canonicalize(reordered).exact_hash
