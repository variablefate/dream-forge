"""Overnight parameter sweep for adaptive server thresholds.

Tests different TIER1/TIER2 threshold combinations against the 50-question
validation set (25 easy + 25 hard from TriviaQA). Logs all results to
data/parameter_sweep_results.json.

Run: uv run python tests/parameter_sweep.py

No server needed — uses the fast scorer directly on cached sample texts.
"""

import json
import time
import numpy as np
from pathlib import Path
from src.runtime.fast_scorer import FastScorer

def run_sweep():
    print("Loading fast scorer...")
    scorer = FastScorer.from_pretrained()

    # Load cached samples (same as our 50-question validation)
    print("Loading TriviaQA samples...")
    correct_texts = []
    incorrect_texts = []
    with open('data/trivia_samples/full_samples.jsonl', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            samples = entry.get('samples', [])
            n_correct = sum(1 for s in samples if s.get('correct'))
            text = samples[0]['text'] if samples else ''
            if n_correct == len(samples) and len(samples) == 10:
                correct_texts.append(text)
            elif n_correct == 0:
                incorrect_texts.append(text)

    # Use more samples for statistical power
    import random
    random.seed(42)
    n_test = min(200, len(correct_texts), len(incorrect_texts))
    correct_sample = random.sample(correct_texts, n_test)
    incorrect_sample = random.sample(incorrect_texts, n_test)

    print(f"Scoring {n_test} correct + {n_test} incorrect = {n_test*2} samples...")

    # Score all samples
    t0 = time.time()
    correct_risks = scorer.score_batch(correct_sample)
    incorrect_risks = scorer.score_batch(incorrect_sample)
    score_time = time.time() - t0
    print(f"Scored in {score_time:.1f}s ({score_time/(n_test*2)*1000:.1f}ms/sample)")

    # Sweep thresholds
    tier1_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    tier2_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    # Time estimates (seconds) based on our benchmarks
    TIME_FAST_SERVE = 4    # pass@1 no CETT
    TIME_BEST_OF_2 = 35    # best-of-2 with CETT
    TIME_BEST_OF_4 = 73    # best-of-4 with CETT

    results = []

    print(f"\nSweeping {len(tier1_values)} x {len(tier2_values)} = {len(tier1_values)*len(tier2_values)} combinations...")
    print(f"{'TIER1':>6} {'TIER2':>6} | {'Fast%':>6} {'Bo2%':>6} {'Bo4%':>6} | {'Leaked':>7} {'FalseEsc':>9} | {'AvgTime':>8} {'Score':>8}")
    print("-" * 85)

    for t1 in tier1_values:
        for t2 in tier2_values:
            if t2 <= t1:
                continue  # TIER2 must be > TIER1

            # Classify each sample
            fast_serve_correct = 0
            fast_serve_incorrect = 0  # LEAKED hallucination
            bo2_correct = 0
            bo2_incorrect = 0
            bo4_correct = 0
            bo4_incorrect = 0

            for risk in correct_risks:
                if risk < t1:
                    fast_serve_correct += 1
                elif risk < t2:
                    bo2_correct += 1
                else:
                    bo4_correct += 1

            for risk in incorrect_risks:
                if risk < t1:
                    fast_serve_incorrect += 1  # DANGER: hallucination served fast
                elif risk < t2:
                    bo2_incorrect += 1
                else:
                    bo4_incorrect += 1

            total = n_test * 2
            fast_pct = (fast_serve_correct + fast_serve_incorrect) / total
            bo2_pct = (bo2_correct + bo2_incorrect) / total
            bo4_pct = (bo4_correct + bo4_incorrect) / total

            leaked = fast_serve_incorrect
            leaked_pct = leaked / n_test  # % of hallucinations that leaked
            false_escalated = bo2_correct + bo4_correct  # correct answers unnecessarily escalated
            false_esc_pct = false_escalated / n_test

            # Average time estimate
            avg_time = (
                (fast_serve_correct + fast_serve_incorrect) * TIME_FAST_SERVE +
                (bo2_correct + bo2_incorrect) * TIME_BEST_OF_2 +
                (bo4_correct + bo4_incorrect) * TIME_BEST_OF_4
            ) / total

            # Composite score: minimize time while keeping leaked < 5%
            # Penalize heavily if leaked > 5%
            leak_penalty = max(0, (leaked_pct - 0.05)) * 1000
            score = (1.0 / avg_time) * 100 - leak_penalty

            result = {
                'tier1': t1,
                'tier2': t2,
                'fast_serve_pct': round(fast_pct, 3),
                'best_of_2_pct': round(bo2_pct, 3),
                'best_of_4_pct': round(bo4_pct, 3),
                'leaked_count': leaked,
                'leaked_pct': round(leaked_pct, 4),
                'false_escalated_count': false_escalated,
                'false_escalated_pct': round(false_esc_pct, 4),
                'avg_time_estimate': round(avg_time, 1),
                'score': round(score, 2),
            }
            results.append(result)

            print(f"{t1:>6.2f} {t2:>6.2f} | {fast_pct:>5.1%} {bo2_pct:>5.1%} {bo4_pct:>5.1%} | "
                  f"{leaked:>3}/{n_test} ({leaked_pct:>4.1%}) {false_esc_pct:>8.1%} | "
                  f"{avg_time:>7.1f}s {score:>7.1f}")

    # Sort by score (best first)
    results.sort(key=lambda r: r['score'], reverse=True)

    # Top 10
    print("\n" + "=" * 85)
    print("TOP 10 CONFIGURATIONS (best score = fast + safe)")
    print("=" * 85)
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: TIER1={r['tier1']:.2f} TIER2={r['tier2']:.2f} | "
              f"avg={r['avg_time_estimate']:.1f}s | "
              f"leaked={r['leaked_pct']:.1%} | "
              f"fast_serve={r['fast_serve_pct']:.1%} | "
              f"score={r['score']:.1f}")

    # Best safe configuration (leaked <= 2%)
    safe_results = [r for r in results if r['leaked_pct'] <= 0.02]
    if safe_results:
        best_safe = safe_results[0]
        print(f"\nBEST SAFE (<=2% leak): TIER1={best_safe['tier1']:.2f} TIER2={best_safe['tier2']:.2f} | "
              f"avg={best_safe['avg_time_estimate']:.1f}s | leaked={best_safe['leaked_pct']:.1%}")

    # Best zero-leak
    zero_leak = [r for r in results if r['leaked_count'] == 0]
    if zero_leak:
        best_zero = zero_leak[0]
        print(f"BEST ZERO-LEAK: TIER1={best_zero['tier1']:.2f} TIER2={best_zero['tier2']:.2f} | "
              f"avg={best_zero['avg_time_estimate']:.1f}s")

    # Save full results
    output = {
        'sweep_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_correct': n_test,
        'n_incorrect': n_test,
        'total_samples': n_test * 2,
        'top_10': results[:10],
        'best_safe': safe_results[0] if safe_results else None,
        'best_zero_leak': zero_leak[0] if zero_leak else None,
        'all_results': results,
        'current_config': {'tier1': 0.20, 'tier2': 0.40},
    }

    output_path = Path('data/parameter_sweep_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f"\nFull results saved to {output_path}")


if __name__ == '__main__':
    run_sweep()
