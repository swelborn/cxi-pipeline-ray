# Downstream CXI Compatibility — is sharded output acceptable?

**Task:** 23 — survey consumers of CXI files produced by the demo pipeline
and determine whether multi-file-per-run output (Axis 2) is tolerable, or
whether downstream tooling assumes exactly one CXI file per run (forcing
Axis 1).

**Feeds:** Task 24 decision doc (`docs/design/bottleneck-fix-decision.md`).

---

## Verdict (top-level)

**SHARDED OK (provisional).**

Every CXI-consuming artifact in `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml`
either operates on a single file supplied by the caller, enumerates files
from a directory listing, or is already written against the existing
chunked output (filenames of the form
`peaknet_cxi_<timestamp>_chunk0000.cxi`, `_chunk0001.cxi`, …).

The existing single-actor writer already emits N files per run: a new
chunk is produced every `buffer_size` batches (see
`cxi_pipeline_ray/core/file_writer.py:236`). Downstream therefore already
tolerates N-file output for any non-trivial run. An Axis-2 change would
multiply `N` by the number of writer actors — it does not introduce
file-plurality where there was none.

**Why "provisional":** no downstream indexing workflow (e.g. CrystFEL
`indexamajig`, Cheetah) is wired into this repo; the comment reference
`peaknet-with-cheetah-integration.yaml` cited in `cxi_writer-v*.yaml` is a
phantom (no such file in the tree). If such a pipeline exists off-repo,
it was not reachable from this iteration's inventory step, and its
owner's verdict is captured as PENDING in the stakeholder table below.

A follow-up to land this verdict as final (not provisional) requires:
1. Confirmation from the colleague who reported the bottleneck
   (owner of the `--nevents=16` workaround cited in `_01.md`) that
   increasing the file count per run is acceptable.
2. Confirmation from whoever owns downstream indexing/Cheetah that
   running over a `*.cxi` glob is supported and preferred to a
   post-hoc merge.

If either of those returns a hard single-file requirement, the verdict
flips to SINGLE-FILE REQUIRED and Task 24 should pick Axis 1 regardless
of where the bottleneck lies. See the "What would change the verdict"
section at the end.

---

## Inventory — consumers of `.cxi` in `proj-stream-to-ml`

Search: `rg -l "cxi" -tpy -tcfg -tjson` + notebook scan via Python JSON
load for `.ipynb` files. Conducted on sdfiana025 on 2026-04-22 via the
`amsc-demo` bridge session rooted at
`/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml`.

Classification key:
- **1-file-assumed** — code path assumes exactly one CXI file per run
  (e.g. hard-coded filename, no file selection logic that would survive
  an N-file run).
- **globs-multiple** — code enumerates files in a directory or accepts a
  glob pattern.
- **agnostic** — takes one CXI path as input from the caller; the plurality
  question is the caller's concern, not this tool's.
- **producer** — *writes* CXI files; not a consumer.

| File | Type | Classification | Notes |
|---|---|---|---|
| `check_cxi.ipynb` | notebook | agnostic (1 path at a time) | `h5_path = 'peaknet_673m_results/peaknet_cxi_20251014_105841_423771_chunk0000.cxi'` — picks one chunk file, user re-runs cell to switch. Already confronts chunked filenames. |
| `check_cxi-v2.ipynb` | notebook | agnostic (1 path at a time) | Same pattern: `h5_path = '...chunk0007.cxi'` and `...chunk0000.cxi`. Multiple chunk indices referenced across cells. |
| `check_geom.ipynb` | notebook | agnostic | References `.cxi` in text / markdown only; no programmatic file selection. |
| `inspect_writer.ipynb` | notebook | globs-multiple (via help text) + per-cell 1-path | Shows `viz__cxi_writer_from_dump.py --cxi test_cxi_output_debug/test_cxi_*.cxi` as documented usage — glob pattern in CLI. Cells themselves each pick one explicit `*_chunk0003.cxi`, etc. |
| `inspect_q2.ipynb` | notebook | agnostic (H5, not CXI) | Opens HDF5 Q2 dumps, not CXI. No CXI path selection here. |
| `inspect_writer_marimo.py` | script (marimo) | **globs-multiple** | `cxi_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.cxi')])` and then a dropdown. Already built for N-file output; will just show N×M files instead of M. |
| `visualize_cxi.py` | script | 1-file-assumed (but trivial) | `f = h5py.File('peaknet_673m_results/peaknet_cxi_20251014_120638_538315_chunk0010.cxi', 'r')` — single hard-coded path. Already points at `chunk0010` (11th chunk of a run), proving the author edits the string to pick a file. Any change in chunk count is handled by changing the string. |
| `viz__cxi_writer_from_dump.py` | script | **globs-multiple** | Help text: `python viz__cxi_writer_from_dump.py --cxi test_cxi_output_debug/test_cxi_*.cxi --idx 0`. Explicitly documents a glob pattern. |
| `compare_events.py` | script | agnostic | CLI: `python compare_events.py <cxi_file>`. One file at a time; caller passes whichever file is of interest. |
| `debug_cxi_shapes.py` | script | agnostic | CLI: `python debug_cxi_shapes.py <cxi_file>`. Same pattern. |
| `validate_alignment.py` | script | agnostic | CLI: `python validate_alignment.py <cxi_file> [event_index]`. Same pattern. |
| `test_cxi_writer_from_dump.py` | script / harness | producer + single-path self-read | Reads back files it wrote; not a downstream consumer in the production sense. |
| `test_simple_q2_to_cxi.py` | script / harness | producer + single-path self-read | Same. |
| `bare_bones_q2_to_cxi.py` | script | producer | Direct CXI writer prototype; no consumer role. |
| `simple_q2_to_cxi.py` | script | producer | Direct CXI writer prototype. |
| `multi_actor_q2_to_cxi.py` | script | producer (Axis-2 reference) | 460-line standalone Axis-2 prototype. By construction produces N files per run; already exercised downstream. |
| `cxi_writer*.yaml` / `run-config/**/cxi_writer.yaml` | configs | producer config | Writer configs, not consumers. Multiple versions in the tree (`v2`, `v3`, `v4`, `mfxl1047723`, snapshots under `run-config/<timestamp>/`). |

### Summary counts

- Producers / self-tests: 5
- 1-file-assumed (hard-coded in script body): 1 (`visualize_cxi.py` —
  a single `h5py.File(...)` call with a literal path; trivial to re-point
  at any other chunk)
- Agnostic (CLI-driven per-file): 3 scripts + 3 notebooks (each cell
  opens one file at a time; multiple distinct chunk indices already
  used across cells)
- Globs-multiple (directory listing or glob pattern): 2 scripts
  (`inspect_writer_marimo.py`, `viz__cxi_writer_from_dump.py`)

**No consumer in this inventory requires exactly one CXI file per run.**
Every consumer either takes one file at a time (delegating file
selection to a human or a wrapper script) or already enumerates multiple
files.

---

## Stakeholder table

Human outreach is asynchronous and did not complete during this
iteration. The rows below record who needs to be asked, what they own,
and the message to send. Verdict column is PENDING until a reply is
recorded and the doc is re-committed.

| Stakeholder | Consumer owned | Date asked | Verdict | Quoted response / link |
|---|---|---|---|---|
| Author of `check_cxi.ipynb` / `check_cxi-v2.ipynb` (repo owner, cwang31) | Offline inspection notebooks | 2026-04-22 (self-review) | **SHARDED OK** — notebooks already open chunked files by index; N×M instead of M is no change to the workflow | Self-review: every cell picks one `*_chunk<NNNN>.cxi`; the chunk count is already variable. |
| Bottleneck reporter (owner of the `--nevents=16` workaround, quoted in `_01.md`) | End-to-end demo runs + downstream inspection | PENDING | PENDING | Draft question: "If the CXI writer produces `N_actors × N_chunks` files per run instead of `1 × N_chunks`, does that break your downstream inspection or indexing workflow? Is a post-hoc merge step acceptable?" Needs Slack/email reach-out after this iteration. |
| Downstream indexing owner (CrystFEL `indexamajig` / Cheetah) — identity unknown from this inventory | Indexing / hit-finding on CXI output | PENDING | PENDING | No indexing code reachable from `proj-stream-to-ml`; the `peaknet-with-cheetah-integration.yaml` comment-reference in `cxi_writer-v*.yaml` is a phantom file. Follow-up: ask in LCLS peak-finding Slack or direct message to the DAQ/analysis lead who runs this pipeline end-to-end. |

---

## Open unknowns

Explicit list per the Task-23 spec's "open unknowns" requirement.

1. **Which downstream indexing tool, if any, is authoritative?** Cheetah,
   CrystFEL `indexamajig`, something else, or none for this demo so far?
   `indexamajig` accepts a `.lst` file listing input CXI files and
   supports glob-style input lists — if this is the consumer, sharded
   output is trivially OK. Cheetah's behavior depends on version. Needs
   confirmation from the DAQ/analysis owner.
2. **Are there external users of the demo output (other LCLS groups)?**
   The inventory covered only files under `proj-stream-to-ml`. If any
   other group pulls CXI output from this pipeline, their assumptions
   are not captured here. No evidence of external consumers was visible
   in this repo; assumed none until someone surfaces one.
3. **Would a trivial post-hoc merge utility (concatenate N CXI files
   into 1) materially change the verdict?** If stakeholder #2 above
   returns "single file required for my workflow", a small merge script
   would let Axis 2 remain viable. Scope: a ~50-line Python utility that
   opens all shards, concatenates `entry_1/data_1/data`,
   `entry_1/result_1/{nPeaks, peakXPosRaw, peakYPosRaw}`, etc. along
   `axis=0` and rewrites `nPeaks` accordingly. Scoped as a follow-up
   task, not landed here.
4. **Is there any downstream tool that ASSUMES the file is named
   `<run_id>.cxi` (singular, no chunk suffix)?** A directory-listing
   consumer that does `h5py.File(f"{run_id}.cxi")` would break on the
   current writer output too (not just post-Axis-2). Not seen in this
   inventory, but called out because the test would be the same for
   both cases.

---

## What would change the verdict

- If stakeholder #2 (bottleneck reporter / downstream indexing owner)
  replies with a hard "my tool requires a single file" → verdict flips
  to **SINGLE-FILE REQUIRED**. Task 24 should then pick Axis 1 only,
  and the decision doc should note that Axis 1's back-of-envelope
  speedup (≈ min(K, num_panels) = min(K, 4) for B=1, C=4 — ceiling at
  4× for the current demo shape) is the available headroom.
- If stakeholder replies "file plurality is fine but please also keep a
  merge utility available" → verdict stays **SHARDED OK** but a
  follow-up task should scope the merge utility.
- If stakeholder replies "don't care either way" → verdict stays
  **SHARDED OK**; proceed with the Task 24 recommendation that
  combines this verdict with the Task 22 benchmark (peak finding
  dominates at 99%, so Axis 1 is the chosen axis regardless of sharded
  vs. single-file — Axis 2 alone would buy < 0.5% of per-batch time on
  the bottleneck stage).

---

## Inventory commands (for re-running)

```bash
# From the proj-stream-to-ml bridge session (amsc-demo, rooted at the
# demo repo):
bridge --session amsc-demo bash \
  'rg -l "cxi" --type-add "cfg:*.{yaml,cfg,ini,json}" -tpy -tcfg -tjson .'

# Notebook scan (Python JSON parse, since ripgrep over .ipynb is noisy):
bridge --session amsc-demo bash 'python3 -c "
import json
for f in [\"check_cxi.ipynb\", \"check_cxi-v2.ipynb\", \"check_geom.ipynb\",
          \"inspect_writer.ipynb\", \"inspect_q2.ipynb\"]:
    nb = json.load(open(f))
    for cell in nb.get(\"cells\", []):
        src = \"\".join(cell.get(\"source\", [])) if isinstance(cell.get(\"source\"), list) else cell.get(\"source\", \"\")
        for kw in [\".cxi\", \"h5py.File\", \"glob\", \"os.listdir\", \"listdir\"]:
            if kw in src:
                for line in src.split(chr(10)):
                    if kw in line:
                        print(f, \":\", line.strip()[:140])
                break
"'
```

---

## Companion docs

- `_01.md` (in `proj-stream-to-ml`) — problem statement, Open Question
  #1 ("Why was Axis 2 not taken originally?") and #4 ("Is there a
  tolerance for multi-file output?") feed this doc.
- `docs/benchmarks/2026-04-22-baseline-realistic.md` — Task 22 verdict:
  peak finding dominates at 99% of writer time; CXI/HDF5 write paths
  are invisible at < 0.5%.
- `docs/design/bottleneck-fix-decision.md` (pending — Task 24) — will
  combine this verdict with the Task 22 benchmark and pick an axis.
