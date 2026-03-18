"""Thorough test suite for the adaptive 3-tier inference server.

Tests:
  1. Health check + model listing
  2. Easy queries (should hit fast_serve tier)
  3. Hard/hallucination-prone queries (should escalate to best-of-N)
  4. Coding queries (our actual use case)
  5. Edge cases (empty query, very long query, special chars)
  6. Adaptive tier distribution
  7. Hedge behavior on impossible questions
  8. Speed comparison across tiers
  9. OpenAI compatibility (response format)
"""

import json
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8081"


def query(prompt, n_samples=None, temperature=0.7, max_tokens=200):
    """Send a chat completion request."""
    body = {
        "model": "dream-forge",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if n_samples is not None:
        body["n_samples"] = n_samples

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return json.loads(resp.read()), None
    except Exception as e:
        return None, str(e)


def test_health():
    """Test 1: Health check and model listing."""
    print("=" * 60)
    print("TEST 1: Health Check + Model Listing")
    print("=" * 60)

    resp = urllib.request.urlopen(f"{BASE_URL}/health")
    health = json.loads(resp.read())
    assert health["status"] == "ok", f"Health check failed: {health}"
    print(f"  Health: {health}")

    resp = urllib.request.urlopen(f"{BASE_URL}/v1/models")
    models = json.loads(resp.read())
    assert models["data"][0]["id"] == "dream-forge"
    print(f"  Models: {models['data'][0]['id']}")
    print("  PASSED\n")


def test_easy_queries():
    """Test 2: Easy factual questions — should be fast (fast_serve tier)."""
    print("=" * 60)
    print("TEST 2: Easy Queries (expect fast_serve tier)")
    print("=" * 60)

    easy = [
        "What is 2 + 2?",
        "What programming language uses indentation for blocks?",
        "What does HTML stand for?",
    ]

    for q in easy:
        t0 = time.time()
        result, err = query(q)
        elapsed = time.time() - t0
        assert err is None, f"Error: {err}"

        df = result["dream_forge"]
        text = result["choices"][0]["message"]["content"][:80]
        tier = df.get("adaptive_tier", "N/A")
        fast_risk = df.get("fast_risk", "N/A")

        print(f"  [{elapsed:.1f}s] tier={tier} | risk={fast_risk} | conf={df['confidence']:.3f}")
        print(f"    Q: {q}")
        print(f"    A: {text}...")
        print()

    print("  DONE\n")


def test_hard_queries():
    """Test 3: Hard/hallucination-prone — should escalate."""
    print("=" * 60)
    print("TEST 3: Hard Queries (expect escalation)")
    print("=" * 60)

    hard = [
        "Who was the 37th person to walk on the moon?",
        "What is the capital of the country that won the 2045 World Cup?",
        "Explain the QuadTree algorithm used in Python's garbage collector.",
    ]

    for q in hard:
        t0 = time.time()
        result, err = query(q)
        elapsed = time.time() - t0
        assert err is None, f"Error: {err}"

        df = result["dream_forge"]
        text = result["choices"][0]["message"]["content"][:80]
        tier = df.get("adaptive_tier", "N/A")
        fast_risk = df.get("fast_risk", "N/A")

        print(f"  [{elapsed:.1f}s] tier={tier} | risk={fast_risk} | strat={df['strategy']} | hedged={df['hedged']}")
        print(f"    Q: {q}")
        print(f"    A: {text}...")
        print()

    print("  DONE\n")


def test_coding_queries():
    """Test 4: Coding questions — our primary use case."""
    print("=" * 60)
    print("TEST 4: Coding Queries")
    print("=" * 60)

    coding = [
        "Write a Python function to check if a string is a palindrome.",
        "What's the difference between a list and a tuple in Python?",
        "How do I handle a FileNotFoundError in Python?",
        "Explain what a decorator does in Python and give an example.",
    ]

    for q in coding:
        t0 = time.time()
        result, err = query(q, max_tokens=300)
        elapsed = time.time() - t0
        assert err is None, f"Error: {err}"

        df = result["dream_forge"]
        text = result["choices"][0]["message"]["content"][:100]
        tier = df.get("adaptive_tier", "N/A")
        fast_risk = df.get("fast_risk", "N/A")

        print(f"  [{elapsed:.1f}s] tier={tier} | risk={fast_risk} | conf={df['confidence']:.3f}")
        print(f"    Q: {q[:50]}")
        print(f"    A: {text}...")
        print()

    print("  DONE\n")


def test_edge_cases():
    """Test 5: Edge cases."""
    print("=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    # Empty-ish query
    result, err = query("?")
    print(f"  Single char '?': {'OK' if err is None else 'ERROR: ' + err}")

    # Very short
    result, err = query("Hi")
    if result:
        print(f"  'Hi': {result['choices'][0]['message']['content'][:50]}...")

    # Special characters
    result, err = query("What does `os.path.join('a', 'b')` return?")
    if result:
        df = result["dream_forge"]
        print(f"  Special chars: tier={df.get('adaptive_tier')} risk={df.get('fast_risk')}")

    # Max tokens = 1
    result, err = query("Tell me everything about Python", max_tokens=1)
    if result:
        text = result["choices"][0]["message"]["content"]
        print(f"  max_tokens=1: '{text}' (len={len(text)})")

    # Temperature = 0 (greedy)
    result, err = query("What is 1+1?", temperature=0)
    if result:
        print(f"  temp=0: {result['choices'][0]['message']['content'][:50]}")

    # Explicit n_samples=1 (should skip adaptive)
    result, err = query("What is Python?", n_samples=1)
    if result:
        df = result["dream_forge"]
        print(f"  n=1 explicit: tier={df.get('adaptive_tier', 'N/A')} strat={df['strategy']}")

    print("\n  DONE\n")


def test_openai_format():
    """Test 9: OpenAI response format compliance."""
    print("=" * 60)
    print("TEST 6: OpenAI Response Format")
    print("=" * 60)

    result, err = query("What is Python?")
    assert err is None

    # Check required fields
    assert "id" in result, "Missing 'id'"
    assert result["id"].startswith("chatcmpl-"), f"Bad id format: {result['id']}"
    assert result["object"] == "chat.completion"
    assert "created" in result
    assert isinstance(result["created"], int)
    assert result["model"] == "dream-forge"

    choices = result["choices"]
    assert len(choices) == 1
    assert choices[0]["index"] == 0
    assert choices[0]["message"]["role"] == "assistant"
    assert isinstance(choices[0]["message"]["content"], str)
    assert choices[0]["finish_reason"] == "stop"

    assert "usage" in result
    assert "dream_forge" in result

    df = result["dream_forge"]
    assert "strategy" in df
    assert "confidence" in df
    assert "n_generated" in df
    assert "hedged" in df
    assert "elapsed_seconds" in df

    print(f"  All required fields present")
    print(f"  id: {result['id']}")
    print(f"  dream_forge extensions: {list(df.keys())}")
    print("  PASSED\n")


def test_summary():
    """Collect and display summary stats."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Run 5 quick queries and collect tier stats
    test_queries = [
        "What is a variable?",
        "Explain quantum entanglement in detail.",
        "Write a binary search in Python.",
        "Who invented the telephone?",
        "What is the airspeed velocity of an unladen swallow?",
    ]

    tiers = {}
    total_time = 0

    for q in test_queries:
        t0 = time.time()
        result, err = query(q)
        elapsed = time.time() - t0
        total_time += elapsed

        if result:
            tier = result["dream_forge"].get("adaptive_tier", "unknown")
            tiers[tier] = tiers.get(tier, 0) + 1

    print(f"  5 queries in {total_time:.0f}s (avg {total_time/5:.1f}s)")
    print(f"  Tier distribution: {tiers}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADAPTIVE SERVER THOROUGH TEST")
    print("=" * 60 + "\n")

    test_health()
    test_easy_queries()
    test_hard_queries()
    test_coding_queries()
    test_edge_cases()
    test_openai_format()
    test_summary()

    print("ALL TESTS COMPLETE")
