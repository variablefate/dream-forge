"""Quick stress test for adaptive server — run with uv run python tests/stress_test.py"""
import json, time, urllib.request

def q(prompt, **kw):
    body = {'model': 'dream-forge', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': kw.get('temperature', 0.7), 'max_tokens': kw.get('max_tokens', 80)}
    if 'n_samples' in kw:
        body['n_samples'] = kw['n_samples']
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        'http://127.0.0.1:8081/v1/chat/completions',
        data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=600)
    return json.loads(resp.read()), time.time() - t0

queries = [
    ('What color is the sky?', 'easy'),
    ('How many days in a week?', 'easy'),
    ('What language uses curly braces for blocks?', 'easy'),
    ('How do you read a file line by line in Python?', 'coding'),
    ('What is the difference between == and is in Python?', 'coding'),
    ('Explain list comprehension in Python', 'coding'),
    ("What was the name of Napoleon's pet goldfish?", 'impossible'),
    ('Who was the 12th person to step on Mars?', 'impossible'),
    ('What Python method splits a string and returns the last element?', 'tricky'),
    ('What does os.path.read_file() do?', 'tricky'),
    ('What is 7 * 8?', 'math'),
    ('True or False: Python is compiled', 'boolean'),
]

print(f'Running {len(queries)} queries...')
print(f'{"":->90}')
print(f'{"Query":<45} {"Tier":<12} {"Time":>6} {"Risk":>6} {"Hedged":>7} {"Strat":<12}')
print(f'{"":->90}')

tier_times = {}
tier_counts = {}
total_time = 0

for prompt, category in queries:
    try:
        r, t = q(prompt)
        total_time += t
        df = r['dream_forge']
        tier = df.get('adaptive_tier', 'fixed')
        risk = df.get('fast_risk', -1)
        text = r['choices'][0]['message']['content'][:35].replace('\n', ' ')

        tier_times[tier] = tier_times.get(tier, 0) + t
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        print(f'{prompt[:44]:<45} {tier:<12} {t:>5.1f}s {risk:>5.2f} {str(df["hedged"]):>7} {df["strategy"]:<12} {text}')
    except Exception as e:
        print(f'{prompt[:44]:<45} ERROR: {str(e)[:40]}')

print(f'{"":->90}')
print(f'Total: {total_time:.0f}s for {len(queries)} queries ({total_time/len(queries):.1f}s avg)')
print()
print('Tier breakdown:')
for tier in sorted(tier_counts.keys()):
    avg = tier_times[tier] / tier_counts[tier]
    print(f'  {tier}: {tier_counts[tier]} queries, avg {avg:.1f}s')
