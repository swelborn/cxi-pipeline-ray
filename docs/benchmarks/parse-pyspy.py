#!/usr/bin/env python3
"""Parse py-spy speedscope JSON and emit top-N functions (self + total time)."""
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

# Speedscope schema: shared.frames, profiles[].type, profiles[].samples (list of stacks), profiles[].weights
frames = data["shared"]["frames"]
print(f"# {path.name}: {len(frames)} frames")
profiles = data.get("profiles", [])
print(f"# profiles: {len(profiles)}")

total_self = Counter()
total_any = Counter()
n_samples = 0
for prof in profiles:
    samples = prof.get("samples", [])
    weights = prof.get("weights", [])
    if not samples:
        continue
    for stack, w in zip(samples, weights):
        if not stack:
            continue
        n_samples += 1
        # self-time = leaf frame
        leaf = frames[stack[-1]]
        fname = f"{leaf.get('name', '?')}  [{leaf.get('file', '?')}:{leaf.get('line', '?')}]"
        total_self[fname] += w
        # any-frame = any in stack
        seen_in_stack = set()
        for fidx in stack:
            f = frames[fidx]
            key = f"{f.get('name', '?')}  [{f.get('file', '?')}]"
            seen_in_stack.add(key)
        for k in seen_in_stack:
            total_any[k] += w

total_w = sum(total_self.values()) or 1

print(f"\n# total samples: {n_samples}  total weight: {total_w}")
print("\n## Top 20 by SELF time (leaf frame) — 'where the CPU was spending time' ##")
for fn, w in total_self.most_common(20):
    print(f"{100*w/total_w:6.2f}%  {w:8}  {fn}")

print("\n## Top 25 by TOTAL time (any stack frame) — 'paths through which time flows' ##")
for fn, w in total_any.most_common(25):
    print(f"{100*w/total_w:6.2f}%  {w:8}  {fn}")
