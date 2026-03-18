"""Run the 16 over-compliance prompts through the live adaptive server.

Tests whether Qwen pushes back on false premises, fabricated APIs,
bad assumptions, and tool hallucinations — with our adaptive detection.

Run: uv run python tests/run_compliance_bench.py
"""

import io
import json
import sys
import time
import urllib.request
from src.eval.compliance_bench import load_bench, ComplianceResult

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


BASE_URL = "http://127.0.0.1:8081"


def query_server(prompt, max_tokens=300):
    body = json.dumps({
        "model": "dream-forge",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    elapsed = time.time() - t0
    return result, elapsed


def main():
    bench = load_bench()

    print(f"Running {len(bench)} compliance prompts through adaptive server")
    print(f"{'=' * 100}")

    results = []
    by_category = {}
    total_time = 0

    for i, prompt in enumerate(bench):
        r, elapsed = query_server(prompt.query)
        total_time += elapsed

        df = r["dream_forge"]
        response = r["choices"][0]["message"]["content"]
        lower = response.lower()

        good_found = [k for k in prompt.keywords_good if k.lower() in lower]
        bad_found = [k for k in prompt.keywords_bad if k.lower() in lower]

        pushed_back = len(good_found) > 0 and len(bad_found) == 0
        over_complied = len(bad_found) > 0 and len(good_found) == 0
        mixed = len(good_found) > 0 and len(bad_found) > 0

        tier = df.get("adaptive_tier", "N/A")
        hedged = df["hedged"]
        strat = df["strategy"]

        if pushed_back:
            status = "PUSHED BACK"
            icon = "+"
        elif over_complied:
            status = "OVER-COMPLIED"
            icon = "X"
        elif mixed:
            status = "MIXED"
            icon = "~"
        else:
            status = "AMBIGUOUS"
            icon = "?"

        cat = prompt.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "pushed_back": 0, "over_complied": 0, "mixed": 0, "ambiguous": 0}
        by_category[cat]["total"] += 1
        if pushed_back:
            by_category[cat]["pushed_back"] += 1
        elif over_complied:
            by_category[cat]["over_complied"] += 1
        elif mixed:
            by_category[cat]["mixed"] += 1
        else:
            by_category[cat]["ambiguous"] += 1

        print(f"\n[{icon}] {prompt.id} ({cat}) | {elapsed:.1f}s | tier={tier} | hedged={hedged}")
        print(f"    Q: {prompt.query[:80]}")
        print(f"    A: {response[:120]}...")
        if good_found:
            print(f"    Good keywords: {good_found}")
        if bad_found:
            print(f"    BAD keywords: {bad_found}")
        print(f"    Status: {status}")

        results.append({
            "id": prompt.id,
            "category": cat,
            "status": status,
            "pushed_back": pushed_back,
            "over_complied": over_complied,
            "hedged": hedged,
            "tier": tier,
            "strategy": strat,
            "elapsed": elapsed,
            "good_keywords": good_found,
            "bad_keywords": bad_found,
            "response_preview": response[:200],
        })

    # Summary
    total_pb = sum(1 for r in results if r["pushed_back"])
    total_oc = sum(1 for r in results if r["over_complied"])
    total_mixed = sum(1 for r in results if not r["pushed_back"] and not r["over_complied"] and r["good_keywords"])
    total_hedged = sum(1 for r in results if r["hedged"])

    print(f"\n{'=' * 100}")
    print(f"COMPLIANCE BENCHMARK RESULTS")
    print(f"{'=' * 100}")
    print(f"Total: {len(bench)} prompts | {total_time:.0f}s ({total_time/len(bench):.1f}s avg)")
    print(f"Pushed back (correct): {total_pb}/{len(bench)} ({total_pb/len(bench):.0%})")
    print(f"Over-complied (bad):   {total_oc}/{len(bench)} ({total_oc/len(bench):.0%})")
    print(f"Mixed:                 {total_mixed}/{len(bench)}")
    print(f"Hedged by detector:    {total_hedged}/{len(bench)} ({total_hedged/len(bench):.0%})")

    print(f"\nBy category:")
    for cat, stats in sorted(by_category.items()):
        pb = stats["pushed_back"]
        oc = stats["over_complied"]
        print(f"  {cat:<25} pushed_back={pb}/{stats['total']} over_complied={oc}/{stats['total']}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_prompts": len(bench),
        "pushed_back": total_pb,
        "over_complied": total_oc,
        "hedged": total_hedged,
        "pushback_rate": total_pb / len(bench),
        "by_category": by_category,
        "results": results,
    }
    from pathlib import Path
    out_path = Path("data/compliance_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
