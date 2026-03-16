# Sleep/Wake Fine-Tuning Research Project

## Context

Build a self-improving local model system where Qwen3.5 9B examines its own reasoning under different conditions (high-temp "dreaming" vs low-temp "waking"), compares against a stronger model's recorded solutions (Claude/Codex), and uses the contrastive signal to improve both capability and reliability. This is a combined system: teacher-guided SFT/LoRA for capability (which is distillation in practice) + H-neuron intervention/calibration for reliability. Both serve the same goal: a local model that gets better over time.

Restructured after Codex review into 3 lanes that separate "definitely useful" from "scientifically risky."

**Hardware**: RTX 5080 16GB (sm_120 Blackwell, CUDA 12.8+), Windows 11, Python 3.12 via uv
**Platform risk**: bitsandbytes does NOT officially support Windows (community builds only). RTX 5080 sm_120 may need PyTorch nightly for full support. **WSL2 is the recommended fallback** if native Windows has compatibility issues — not just for Hermes, but for the entire bitsandbytes/CUDA stack. Day-0 compat_check tests native Windows first; if it fails, switch to WSL2 for all GPU operations.
**Target model**: Qwen3.5 9B (~10B actual params; "9B" is the marketing label. 32 layers, 4096 hidden dim, `down_proj` in all 32 FFN layers)
**One model end-to-end**: Qwen3.5-9B everywhere (detection, training, deployment). 8-bit base + LoRA adapter training (NOT QLoRA 4-bit — Unsloth warns against it for Qwen3.5). Day-0 compat check must verify 8-bit LoRA fits 16GB. Fallback: all-4B (all phases switch together — detection, training, inference, deployment).
**Deployment — two artifacts** (path-dependent on Lane B outcome):
- **Research runtime**: Python inference with sleep/wake. Full H-neuron path adds: forward hooks + adaptive gating + probe confidence at inference. Detector-only path: detector used for training data selection and eval analysis only (NOT runtime inference). No-H-neuron path: no probe features.
- **Daily-use GGUF Q8** for LM Studio:
  - Full H-neuron: capability LoRA + H-neuron weight scaling + learned hedging
  - Detector-only: capability LoRA + learned hedging (no H-neuron scaling)
  - No H-neuron: capability LoRA + learned hedging only
- **Canonical state**: fp16/merged safetensors — master copy; GGUF exported only after passing evaluation
**Agent**: Pi coding agent (native Windows) primary; Hermes Agent (WSL2 only) optional
**Privacy**: Fully local — no API calls for judging. Use Qwen + embedding similarity.
**Reference**: [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons) (arxiv 2512.01797)

### Codex Review Notes
- **Intervention method — nuanced**: The H-Neurons repo has an internal discrepancy. The README recommends "modulating activations directly during the forward pass" (forward hooks, non-destructive). The actual code in `intervene_model.py` does permanent weight modification (`weight.data[:, neurons] *= alpha`). Both are mathematically equivalent for linear layers, but we'll implement **forward hooks first** (non-destructive, allows rapid alpha experimentation) and weight modification only for final export.
- **Think mode**: Qwen3.5 9B does NOT support `/think`/`/no_think` prompt strings. Must use `enable_thinking=True|False` parameter in `tokenizer.apply_chat_template()`. **Thinking is ON by default for ALL Qwen3.5 models** (verified from the Jinja2 template in tokenizer_config.json — identical for 0.8B through 27B+, no size conditional). The earlier "conflicting reports" were confusion with Qwen3 (different model family). **Always set `enable_thinking` explicitly in every code path regardless.**
- **Codex was right** on: dataset size (5/20 too small, need ~1000+ balanced), starting with QA benchmark, deferring infrastructure until core validates, Hermes requires WSL2.
- **Codex's 3-lane structure adopted** per user agreement.

---

## Key Concepts

- **H-Neurons**: <0.1% of neurons that drive hallucinations. Found by comparing activations during correct vs incorrect outputs.
- **CETT**: Metric scoring each neuron's contribution to output. Used to identify H-neurons.
- **down_proj**: Weight matrix in each transformer layer's MLP. Where H-neurons are detected and suppressed.
- **LoRA**: Parameter-efficient fine-tuning — trains small adapter layers (~50MB) on top of a frozen base. For Qwen3.5-9B: use 8-bit base + LoRA adapters (NOT QLoRA 4-bit — Unsloth flags it for this model family).
- **Unsloth**: Library that makes LoRA training faster. Native Qwen3.5 support, Windows via Anaconda. Supports activation checkpointing (`use_gradient_checkpointing="unsloth"`) to reduce training VRAM.
- **GGUF**: File format for LM Studio / llama.cpp. Export target after fine-tuning.
- **LanceDB**: File-based vector database. No server, just files on disk.

---

## Project Structure

```
autoresearcher/
├── pyproject.toml
├── src/
│   ├── capture/              # Lane A: Experiment extraction
│   │   ├── schema.py         # Pydantic data models (5 stages + optional artifacts)
│   │   ├── cli.py            # Manual experiment entry + batch tools
│   │   └── parser.py         # (future) Codex CLI parser
│   ├── store/                # Shared: Vector DB
│   │   ├── db.py             # LanceDB wrapper
│   │   └── embeddings.py     # all-MiniLM-L6-v2 local embeddings
│   ├── reproduce/            # Lane B: H-neuron reproduction on QA
│   │   ├── dataset.py        # TriviaQA download + consistency filtering
│   │   ├── sampler.py        # Multi-sample generation (temp=1.0, 10x per question)
│   │   ├── hooks.py          # Activation extraction hooks on down_proj
│   │   ├── cett.py           # CETT metric computation
│   │   ├── detect.py         # L1-regularized logistic regression → H-neuron map
│   │   ├── intervene.py      # Weight scaling: down_proj.weight.data[:, neurons] *= alpha
│   │   └── evaluate.py       # Accuracy before/after intervention
│   ├── engine/               # Lane C: Sleep/wake on personal workflows
│   │   ├── model_loader.py   # Load Qwen3.5-9B (8-bit text-only for probing/wake-dream; merged fp16 for baking/export)
│   │   ├── wake.py           # Low-temp inference (what Qwen "knows")
│   │   ├── dream.py          # High-temp multi-sampling (where Qwen is uncertain)
│   │   ├── compare.py        # Local comparison: Qwen judge + embedding similarity
│   │   ├── calibrate.py      # Apply H-neuron findings from Lane B to workflow domain
│   │   └── tune.py           # 8-bit LoRA via Unsloth (9B) + GGUF export
│   ├── orchestrator/         # Infrastructure (build after Lane B validates)
│   │   ├── daemon.py         # Background service watching experiments/
│   │   ├── cycle.py          # One full sleep/wake cycle
│   │   └── agent_api.py      # CLI + JSON interface for Pi/Hermes
│   ├── ui/                   # Progress tracking (build after Lane B validates)
│   │   └── dashboard.py      # Rich terminal UI
│   ├── share/                # Distributed sharing (build after Lane C validates)
│   │   ├── package.py        # Package H-neuron maps + LoRA adapters
│   │   └── protocol.py       # Upload/download correction packages
│   └── eval/
│       └── metrics.py        # Accuracy, hallucination rate, calibration (ECE/Brier), per-domain
├── skills/
│   └── dream-forge/
│       └── SKILL.md          # Claude Code skill
├── agent/
│   └── pi/
│       └── sleep-wake.ts     # Pi extension (native Windows)
├── experiments/              # Captured experiment JSON files
├── models/                   # Checkpoints, H-neuron maps, LoRA adapters
├── corrections/              # Packaged corrections for sharing
└── data/                     # LanceDB + activation cache
```

---

## Lane A: Experiment Capture (useful regardless of whether H-neurons work)

### A.1 Data schema (`src/capture/schema.py`)
```python
class Experiment:
    id: UUID
    source: "claude" | "codex" | "manual"
    timestamp: datetime
    project: str
    # 5 core stages
    problem: str
    breakdown: list[str]
    proposed_solutions: list[str]
    review_issues: list[ReviewRound]   # round number, issues found, corrections made
    final_plan: str
    # Resolution status and grouping
    status: str                        # "resolved" | "unresolved" — only resolved feeds SFT/calibration
    task_group_id: str                 # shared across experiments about the same underlying task/bug/feature
    # The actual final answer/implementation — only required for resolved experiments
    reference_solution: str | None     # final resolved outcome (code, answer, config change). None if unresolved.
    resolution_type: str | None        # "code_change" | "answer" | "config_change" | "research_finding"
    # Artifacts — split by label strength
    # Task context — strictly split by temporal provenance
    pre_solution_context: list[ContextFile] | None   # files available BEFORE solving {path, content, revision, provenance: "error_trace"|"user_provided"|"retrieved_pre"|"materialized_fallback"}
    post_solution_artifacts: list[ContextFile] | None # files only available AFTER solving {path, content, revision, provenance: "diff"|"test_output"|"retrieved_post"}
    repo_hash: str | None              # git revision hash at time of task — all retrieval must use this revision
    repo_dirty: bool = False           # True if working tree had uncommitted changes at task start
    git_diff: str | None                  # net file changes (git diff from before to after) — stored for provenance/debugging, NOT the SFT training target
    git_start_hash: str | None            # commit hash at task start (for provenance)
    # Synthetic replay fields (optional, for C.2 dream-generated data):
    synthetic: bool = False
    generator: str | None                 # "false_premise" | "counterfactual" | "replay" | "high_temp"
    parent_experiment_id: UUID | None
    generation_depth: int = 0             # 0 = real experiment, 1 = synthetic child (max depth)
    commands_run: list[str] | None     # commands/tests executed during resolution
    error_output: str | None           # the original error/failure that prompted the task
    constraints: list[str] | None      # e.g., "must not break existing tests", "Python 3.10 only"
    # Tier 1a (strong labels — feed neuron claims + calibrator):
    test_results: TestResult | None    # passed/failed + output + assertion details
    # Tier 1b (weaker labels — useful context, not for neuron claims alone):
    build_results: BuildResult | None  # passed/failed + output
    lint_results: LintResult | None    # passed/failed + output
    error_logs: list[str] | None       # negative evidence only
    # Note: per-file diffs removed — use git_diff (consolidated) for all diff needs. Parse per-file if needed.
    # Lifecycle
    resolves_experiment_id: UUID | None  # if this resolved capture supersedes an earlier unresolved one, link it here
    superseded: bool = False             # True once a resolved child exists; hidden from default retrieval
    # Metadata
    tags: list[str]                    # domain tags (python, android, web, etc.)
```

### A.2 Vector DB (`src/store/`)
- **LanceDB** — file-based, multiple vector columns
- **all-MiniLM-L6-v2** embeddings (~91MB model, CPU, 384-dim, truncates at 256 word pieces)
- Embed `problem` and `breakdown` as always-present vector columns
- `reference_solution` vectors: use **separate tables** for resolved vs unresolved experiments (`experiments_resolved` and `experiments_unresolved`). This is safer than null vectors + filter discipline — one missed filter would pollute search results. The resolved table has the `reference_solution` vector column; the unresolved table omits it entirely. Promotion from unresolved to resolved = insert into resolved table + mark original as superseded.
- Artifacts stored as structured JSON columns — queryable but not embedded
- Operations: insert, search_similar, filter_by_tag, get_by_id
- **Default retrieval filter**: `WHERE superseded != true` — superseded unresolved parents hidden from normal search

### A.2.1 Memory governance layer (hippocampal-inspired lifecycle model)
Lightweight lifecycle controls on the experiment store. NOT a full hippocampal memory system — just the data hygiene parts.

**Three-tier data surface** (maps to hippo's buffer/episodic/semantic):
- **Buffer**: Unresolved captures, in-progress notes. Not training-eligible. Decays to superseded when resolved.
- **Episodic**: Resolved experiments with full provenance. Training-eligible. The core dataset.
- **Semantic** (future work — no implementation in v1): Distilled lessons/rules extracted from repeated episodic patterns. Retrieval-only artifacts — used for agent guidance and context, NOT as automatic training data unless human-reviewed. Define extraction trigger and format during Lane C iteration.

**Confidence tiers** on experiments (tracked as metadata, not vectors):
- `verified` — has Tier 1a test/assertion evidence
- `observed` — has Tier 1b build/lint evidence or Tier 2 embedding similarity
- `inferred` — Tier 3 Qwen self-judge only
- `stale` — auto-assigned if >90 days since last retrieval AND confidence is not `verified`

**Outcome feedback** (`src/store/feedback.py`):
- When a retrieved experiment helps solve a new task (user confirms or Tier 1a evidence), mark outcome=good → experiment gets priority in future retrieval.
- When a retrieved experiment is irrelevant or misleading, mark outcome=bad → deprioritized in retrieval ranking.
- Simple: just a `retrieval_count` + `positive_outcome_count` per experiment, used as a retrieval boost factor.

**What NOT to do** (per Codex's scoping advice):
- No automatic decay on verified/gold training data
- No automatic merge of episodic experiments into synthetic training targets
- No schema-fit weighted training
- Semantic distilled lessons are retrieval artifacts, not training truth

### A.3 Claude Code skill (`skills/dream-forge/SKILL.md`)
Runs in-context — Claude summarizes its own conversation into the 5 stages + reference solution + artifacts.
- **`status`**: Set to `resolved` if the task has an actual implemented/verified outcome, `unresolved` if it's still in progress or planning-only.
- **`reference_solution` required for resolved only** — for code tasks, this is the **final correct code** (the function, class, or block after the fix), NOT a unified diff. Training on diffs teaches the model to produce patch syntax (+/-/@@ metadata) instead of correct code, wastes tokens on diff overhead (~100-150 tokens), and is fragile (hallucinated line numbers break the patch). Store the raw diff in `git_diff` for provenance/debugging. For non-code tasks, this is a structured summary of the final answer. For unresolved captures, set to None.
- **`task_group_id`**: Auto-assigned by Claude based on semantic similarity to existing experiments, but **user-confirmable** via `uv run python -m src.capture.cli review-groups` before any train/validation/test split is generated. This is a split-integrity field — incorrect grouping causes train/test leakage — so it must not rely on heuristics alone.
- `final_plan` captures the planning prose; `reference_solution` captures what actually solved the problem. Both saved, distinct fields.
- **Provenance capture rules** (how the skill assigns `pre_solution_context` vs `post_solution_artifacts`):
  - **`error_trace`**: Files referenced in error messages/stack traces shown BEFORE the assistant began solving. Captured at task start.
  - **`user_provided`**: Files the user explicitly shared or referenced in the problem statement.
  - **`retrieved_pre`**: Files the assistant actually retrieved/read BEFORE the first modifying action (observed teacher context from the real conversation). Classification is based on git-based provenance: files present before `git diff` shows changes = pre-solution.
  - **`materialized_fallback`**: Files added LATER by the data_prep pipeline's fallback retrieval step (C.5 step 2) when an example becomes SFT-eligible. These are reconstructed training context, NOT observed teacher context. Distinct provenance so auditing can tell what was real vs reconstructed.
  - **`diff` / `test_output` / `retrieved_post`**: Files created, modified, or read AFTER the first Edit/Write/Bash-that-modifies call = post-solution. Goes into `post_solution_artifacts`.
  - **Decision boundary**: Use **git-based provenance** instead of tool-call introspection (Claude Code skills don't expose a structured tool-call API). The skill runs `git diff` and `git status` to identify modified files (post-solution) and `git log` to verify starting state. Pre-solution context = files referenced in error traces + files explicitly provided. Post-solution artifacts = files appearing in `git diff`. This is approximate but reliable and auditable.
- **`repo_dirty`**: Boolean flag. If the working tree had uncommitted changes at task start, set to `true`. When `repo_dirty: true`, fallback repo retrieval (C.5 step 2) is disabled unless the needed files are already in `pre_solution_context` — because `repo_hash` alone cannot reconstruct the actual state the agent worked with.
- Unresolved captures are still useful for retrieval, clustering, and future follow-up, but do NOT feed SFT/calibration pipeline.
- **Unresolved→resolved upgrade**: When a task is later resolved, create a NEW resolved experiment with `resolves_experiment_id` pointing to the original unresolved one. Do NOT update the unresolved record in place (preserves history). The new resolved record inherits the same `task_group_id` and split assignment.
- **Superseded parent policy**: Once a resolved child exists, the unresolved parent is marked `superseded: true` and **hidden from default retrieval/search**. It remains in the database for history/audit views only. Default LanceDB queries filter `WHERE superseded != true AND status == "resolved"` for training candidates, `WHERE superseded != true` for general retrieval.
Triggered by: `/dream-forge` after completing a plan execution or at any natural stopping point.
Saves JSON to `experiments/<timestamp>-<slug>.json`, then runs: `uv run python -m src.capture.cli validate <file>` (Pydantic validation — structural + non-empty required fields check. Max 2 retries. Non-zero exit code on failure. The ~30-field schema is complex enough that Claude occasionally omits fields or uses wrong names).

**Disk budget note**: Activation caching at [32 layers × 12288 neurons × float32] = 1.5MB per sample. 1000 samples = 1.5GB. 5000 pilot = 7.5GB. Document in `data/` directory description.

### A.4 Manual CLI (`src/capture/cli.py`)
`uv run python -m src.capture.cli add` — interactive entry for bootstrapping seed data.

---

## Lane B: H-Neuron Adaptation on QA (validate the science)

This is an adaptation/port of the H-Neurons paper methodology to Qwen3.5 9B on TriviaQA. Key differences from the original:
- `transformers` v5+ for both sampling AND activation extraction (original uses vLLM for fast sampling + transformers for hook access). Local string matching instead of GPT-4o for answer-span extraction.
- Qwen3.5 is a **multimodal hybrid architecture**: text backbone alternates 3x linear_attention (Gated DeltaNet) + 1x full_attention. Load text-only (`Qwen3_5ForCausalLM.from_pretrained(...)`) to skip vision encoder.
- Both layer types have standard `down_proj` in their MLP, so H-neuron hooks work on all 32 layers. Note: full-attention layers also use output gating (`attn_output_gate: true`, making them "Gated Attention", not plain attention). DeltaNet vs Gated Attention layers may distribute H-neurons differently — **stratify analysis by layer type**. Derive groups from `config.text_config.layer_types`: pattern is `[linear_attention, linear_attention, linear_attention, full_attention] × 8` for the 9B model.
The core scientific method (consistency filtering → CETT activation extraction → L1 sparse probe → intervention) is preserved.

### B.0a Day-0 compatibility gate (`src/reproduce/compat_check.py`)
Before any Lane B work, verify the full environment works:
- **PyTorch sm_120 compatibility**: RTX 5080 uses sm_120 (Blackwell). Verify `torch.cuda.is_available()` and that a simple tensor op on GPU works. May need PyTorch nightly or custom build for sm_120 support. Pin PyTorch version once verified.
- `transformers` v5+ installed (Qwen3.5 requires v5 — HF docs only exist in v5.x)
- **bitsandbytes Windows check**: bitsandbytes has no official Windows support. Test 8-bit loading — if it fails, switch to WSL2 for all GPU operations.
- **GGUF DeltaNet conversion check**: Convert the unmodified base model to GGUF via `convert_hf_to_gguf.py` → test-infer → verify coherent output. DeltaNet layer support in llama.cpp is unverified. Pin llama.cpp version.
- `Qwen/Qwen3.5-9B` loads **text-only** at 8-bit via bitsandbytes. Try `Qwen3_5ForCausalLM.from_pretrained(...)` (must explicitly specify this class — default architecture is `Qwen3_5ForConditionalGeneration` which includes the vision encoder. Verify in compat_check that VRAM matches text-only expectations.) to skip vision encoder.
- **RTX 5080 + bitsandbytes 8-bit works** (RTX 50-series is new — real compat risk). If 8-bit fails, fall back to 4-bit detection.
- `tokenizer.apply_chat_template(messages, enable_thinking=False)` works
- `tokenizer.apply_chat_template(messages, enable_thinking=True)` works
- Forward hooks register on `down_proj` layers in both DeltaNet and full-attention layer types
- Single inference completes, VRAM stays under 16GB with batch=1
- **Training VRAM check**: Load 9B 8-bit → add LoRA (with all DeltaNet + full-attention target modules) → run one training step with `use_gradient_checkpointing="unsloth"` at batch_size=1, seq_len=2048 → check peak VRAM < 16GB
- **Gradient checkpointing**: Unsloth's `"unsloth"` mode (activation offloading) documented with 16-bit and 4-bit but not explicitly with 8-bit base + LoRA. PEFT fallback uses `model.gradient_checkpointing_enable()` (standard HF — more VRAM but guaranteed compatible). Test both in compat_check.
- **LoRA merge test**: Run `save_pretrained_merged` early to verify 8-bit → fp16 merge works. Known issues: Unsloth downloads ~18GB fp16 weights for merge (pre-download once to avoid surprise disk usage), and issue #3277 reports LoRA count mismatches from `load_in_8bit=True` (may or may not be resolved)
- **Verify Qwen3.5 8-bit loading works**: Note: #3501/#3806 are Qwen3 (standard transformer), NOT Qwen3.5 (hybrid DeltaNet). The fix (3D tensor support in bitsandbytes 8-bit matmul) is general and may help, but Qwen3.5's issues may be entirely different due to the DeltaNet architecture. Treat 8-bit loading as an independent go/no-go test for Qwen3.5 specifically.
- **CRITICAL — Unsloth Qwen3.5 is actively broken**: #4294 (LoRA merge failures), #4166 (Embedding attribute errors), #4188 (extreme VRAM on 9B even with 32GB). Root cause: Unsloth has NO Qwen3.5-specific code — treats the hybrid DeltaNet architecture as standard transformer.
- **Primary framework: Unsloth** — faster training, native GGUF export. Qwen3.5-specific bugs exist (#4294 merge, #4188 VRAM, #4160 NaN) but may be resolved quickly. Day-0 compat_check tests the full pipeline and fails fast if Unsloth is broken.
- **Framework fallback chain** (compat_check tests in this order, stop at first success):
  1. Unsloth (`FastLanguageModel` + `save_pretrained_merged`) — try first
  2. Standard PEFT + HF Trainer (no Unsloth) — guaranteed compatible, slower, different merge path
  3. DeepSpeed ZeRO-3 + CPU offloading (final fallback — much slower, needs WSL2 for async_io)
- **Pin framework version**: Once compat_check passes, pin in `pyproject.toml`. Do NOT auto-upgrade.
- **DeltaNet LoRA targeting verification**: After loading model + LoRA, **print all trainable parameter names** and verify DeltaNet modules (in_proj_qkv, in_proj_z, out_proj) appear. PEFT silently skips unrecognized modules — if DeltaNet targets don't work, only 8/32 attention layers (25%) get LoRA adapters. This is a **hard gate**: if DeltaNet LoRA targeting fails, the training is fundamentally incomplete.
- `dataset_num_proc=1` set for Windows stability
- Run as: `uv run python -m src.reproduce.compat_check`

### B.0b Qwen 9B fast sanity gate (`src/reproduce/sanity_gate.py`)
Before committing to the full Lane B build, run a quick 100-200 sample QA pilot testing BOTH probe roles:
- Generate 10 samples each on 200 TriviaQA questions
- Filter for consistency, extract activations on a small balanced subset (~50 per class)
- **Detector test** (L2 1-vs-1): check classification accuracy. > 55% = some detection signal, > 60% = useful for confidence/gating.
- **Intervention test** (L1 3-vs-1): check sparsity + quick alpha sweep [0.0, 0.3, 0.5]. Any hallucination reduction without obvious capability collapse?
- **Decision matrix**:
  - Both signals present → proceed with full Lane B
  - Detector works but intervention is weak → proceed, but plan to use H-neurons for confidence scoring and replay priority only (NO gating, NO weight scaling — gating requires a neuron map to suppress)
  - Neither signal → pivot to LoRA-only (no H-neuron), do not invest weeks in full Lane B
- Run as: `uv run python -m src.reproduce.sanity_gate`

### B.1 Dataset (`src/reproduce/dataset.py`)
- Download TriviaQA **rc.nocontext** subset (same as the H-Neurons repo uses for training data)
- Generate 10 samples per question using Qwen3.5-9B (8-bit) with: `temperature=1.0, top_k=50, top_p=0.9, enable_thinking=False, max_tokens=50` (matching H-Neurons repo defaults). **enable_thinking=False is mandatory** — with thinking ON, the model spends all tokens on `<think>` blocks and never produces an answer. With thinking OFF, the empty think block is prefill input and doesn't consume from max_tokens.
- **Resumability**: Cache all generated responses to `data/trivia_samples/` as JSONL (one file per batch). Sampling and activation extraction are long single-GPU jobs (5K questions × 10 samples = 50K inferences). If interrupted, resume from last completed batch. Same for activation extraction — cache per-sample `.npy` files and skip existing ones.
- Filter for consistency: keep only questions where ALL 10 samples are correct OR all are incorrect
- Balance: equal faithful + hallucinated samples
- **Target**: 1,000 balanced samples minimum (500 correct + 500 incorrect)
- **Pilot pass first**: Run 5,000 questions, measure consistency yield. A 9B model may have far fewer "all 10 correct/incorrect" samples than the 70B models the paper tested (bimodal confidence helps, but yield could still be <1%). If strict 10/10 filtering yields <500 per class:
  - **Asymmetric filtering expected**: At ~75% accuracy, "all 10 correct" yields ~5.6% of questions but "all 10 incorrect" yields ~0.0001%. The class imbalance is catastrophic with strict filtering. Use asymmetric thresholds: keep 10/10 for the "correct" class (plenty of samples), but relax the "incorrect" class to ≥7/10 incorrect. This is a methodological decision — document it explicitly.
  - Or use confidence-weighted filtering (retain high-confidence subset from the full 95K TriviaQA question pool)
  - Document all thresholds used — this is an adaptation, not a reproduction
- **Judging**: Normalized alias matching (same as H-Neurons repo) — normalize both response and gold answer (lowercase, strip articles/punctuation), check if any normalized alias is a substring of the normalized response. No API needed.

### B.1.1 Answer-token extraction (`src/reproduce/answer_spans.py`)
The H-Neurons paper uses GPT-4o for answer-span extraction. We do it locally:
Two-pass matching strategy, both on the ORIGINAL response text with `return_offsets_mapping=True`:
- **Pass 1: Raw alias matching** — try each entry in TriviaQA's `aliases` (original casing/punctuation) as a substring of the original response. This preserves exact character offsets.
- **Pass 2: Case-insensitive matching** — if Pass 1 misses, match `normalized_aliases` against `response.lower()`. Offsets usually align because `.lower()` preserves string length for ASCII. **Unicode guard**: add a runtime `assert len(response) == len(response.lower())` check — `.lower()` can change length for certain Unicode characters (e.g., Turkish İ→i̇). If length changes, skip Pass 2 and fall back to all-output-token mode for that example.
- **Fallback: All-output-token mode** — if both passes miss (answer is paraphrased), use all output tokens for CETT. Set `answer_tokens: []` per the repo instructions.
- **Offset alignment step**: `offset_mapping` from `tokenizer(response)` gives positions in the raw response text, BUT the actual model input is a chat-templated sequence with special tokens (`<|im_start|>`, role headers, `<|im_end|>`) that shift all positions. To get correct token indices in the full sequence: (a) compute offsets on raw response to find answer span characters, (b) separately tokenize the full chat-templated text, (c) locate the answer tokens within it by subsequence string matching (similar to how the H-Neurons repo uses LLM-based token identification). Do NOT assume `offset_mapping` from raw text produces globally-correct indices in the templated sequence.
- Track match rate per pass. Estimates are rough — Pass 2 (.lower() matching) still misses punctuation/spacing/article variants that full normalization would absorb. Actual fallback rate to all-output-token mode may be higher than expected. Measure, don't assume.
- **Gate metric**: If total span-match rate (Pass 1 + Pass 2) < 50%, treat all-output-token mode as the primary methodology and report this explicitly. The mode switch is a methodological decision, not a silent fallback.

### B.2 Activation extraction (`src/reproduce/hooks.py` + `cett.py`)
- Register forward hooks on all 32 `down_proj` layers
- Capture `input[0].detach()` + output norms
- CETT = (abs(activations) * weight_norms) / (output_norms + 1e-8)  where weight_norms = per-column norms of down_proj weights (matching the exact H-Neurons code path in extract_activations.py)
- **Aggregation**: mean across the token dimension (matching the repo default). Max is an alternative but not the baseline.
- Aggregate across answer tokens (the actual answer span in the response)
- Output: numpy array [32 layers × 12288 neurons] per sample
- Save to `data/activations/`

### B.3 H-neuron identification (`src/reproduce/detect.py`)
Train **two probes** for different purposes (per the H-Neurons repo's own recommendation):

**Detector probe** (for confidence/calibration scoring):
- **1-vs-1 mode** + **L2 regularization**: hallucinated answer tokens vs faithful answer tokens
- Highest predictive accuracy (paper's recommendation for detection)
- Outputs sigmoid probability = hallucination risk score → feeds into Lane C calibrator
- Save: `models/detector_probe.pkl`

**Intervention probe** (for sparse neuron map + weight scaling):
- **3-vs-1 mode** + **L1 regularization**: hallucinated answer tokens vs (faithful answer + non-answer tokens)
- Isolates neurons specific to factual errors, not general language modeling
- Sparser, more targeted — paper's recommended mode for intervention
- Grid search over C parameter for sparsity/accuracy tradeoff. Defaults: `solver="liblinear"`, `max_iter=1000`. **Note**: `penalty="l1"` is deprecated in scikit-learn 1.8+ (removed in 1.10). Use `l1_ratio=1` instead. Pin scikit-learn version in `pyproject.toml`.
- H-neurons = features with positive weights → map to (layer_idx, neuron_idx) pairs
- Save: `models/h_neuron_map.json`

### B.4 Intervention (`src/reproduce/intervene.py`)
Two approaches (both mathematically equivalent for linear layers):
- **Forward-hook activation scaling (primary, for experimentation)**: Register hooks that scale `input[0][:, :, target_neurons] *= alpha` during forward pass. Non-destructive — original weights untouched, alpha can be changed per-inference. This follows the README recommendation.
- **Permanent weight scaling (for export only)**: Cannot modify 8-bit quantized weights directly (int8 + scale factors). Two cases:
  - **Lane B standalone (no LoRA yet)**: Load base model in fp16 (`torch_dtype=torch.float16`, no quantization), apply `down_proj.weight.data[:, target_neurons] *= alpha`, save as fp16 safetensors. Export to GGUF Q8 via `llama.cpp`'s `convert_hf_to_gguf.py` (Unsloth's `save_pretrained_gguf` won't work on manually-modified weights).
  - **Lane C (after LoRA training)**:
    - **If Unsloth**: `save_pretrained_merged(save_method="merged_16bit")`, then apply weight scaling, then export GGUF Q8 via `convert_hf_to_gguf.py`
    - **If PEFT**: Reload base in fp16 on CPU (`device_map="cpu"`, `torch_dtype=torch.float16`), load adapter, `model.merge_and_unload()`, save. Requires ~20GB free system RAM. Then apply weight scaling, then export GGUF Q8.
- Grid search alpha in [0.0, 0.1, 0.3, 0.5, 0.7] using the hook approach first
- **GGUF thinking template (MANDATORY)**: ALL Qwen3.5 models default to thinking ON — GGUF carries the template verbatim. Before GGUF export, **patch the Jinja2 template itself** to hardcode `enable_thinking=False`. Do NOT rely on runtime kwargs — `enable_thinking` is broken in llama.cpp CLI (issue #20182) and LM Studio has no GUI toggle (issue #1559). Without this patch, users see raw `<think>...</think>` blocks.
- **llama.cpp DeltaNet verification**: Pin llama.cpp version. `convert_hf_to_gguf.py` has explicit Qwen3.5 support (PRs #19435, #19468). Add to compat_check: convert unmodified base to GGUF → test-infer → verify coherent output → **verify KV cache works** (second prompt should NOT re-process full context — llama.cpp #19858 reports KV cache issues with hybrid DeltaNet layers causing massive slowdown). Q8_0 is the safe quantization choice for H-neuron-modified weights — if lower quants are ever considered, use `llama-imatrix` for importance matrix generation.
- **Verification step**: Compare hook-based vs baked intervention on held-out TriviaQA to confirm equivalence before exporting

### B.5 Evaluation + stability checks (`src/reproduce/evaluate.py`)
- TriviaQA accuracy before vs after intervention
- **Stability**: Run classifier with 3+ random seeds / sample splits, measure H-neuron map overlap. If identified neurons are unstable across seeds, the probe is not reliable.
- **Non-regression**: Track retained accuracy on originally-correct examples. Intervention must not tank capability.
- **Abstain/refusal rate**: Measure before vs after — intervention can "improve hallucination rate" by making the model refuse to answer. Track this explicitly.

### B.6 Gate (pilot thresholds — adjust based on evidence)
These are starting targets, not hard scientific criteria. The H-Neurons paper emphasizes tradeoffs around C, sparsity, and capability retention — exact thresholds depend on what the pilot data shows.
- Classifier accuracy > 65%
- Sparse neuron set (< 0.1% of total)
- Stable across 3+ seeds (> 50% neuron overlap)
- Retained accuracy on correct examples drops < 5%
- Abstain rate increase < 10%
- Measurable hallucination reduction on held-out QA

**Three possible outcomes** (aligns with B.0b decision matrix):
- **Full H-neuron path**: Both detector and intervention pass → Lane C uses detection, gating, intervention, weight scaling, and re-detect on promotion.
- **Detector-only path**: Detector works but intervention is weak → Lane C uses detection as a **training/eval aid only, NOT a runtime inference feature**. Specifically:
  - **Training aid**: detector score drives active replay priority (which failures get replayed next)
  - **Eval aid**: detector confidence enables richer evaluation analysis (confidence-stratified accuracy)
  - **NOT runtime**: no forward hooks, no gating, no suppression, no runtime abstention from the probe. Runtime hedging comes solely from learned LoRA behavior.
  - NO weight scaling, NO L1 intervention probe, NO bake step, NO intervention arms in ablation
  - **Still requires L2 detector re-calibration after LoRA merge** — base-model detector is stale on post-LoRA activations (consistent with C.5.2)
  - GGUF gets capability LoRA + learned hedging only (no baked H-neuron scaling)
- **No H-neuron path**: Neither signal → Lane C pivots to LoRA-only (no H-neuron). H-neuron code becomes research-only, not part of the product pipeline.

### VRAM budget for Lane B (8-bit inference)
Qwen3.5-9B is a multimodal model (text + vision encoder). **Load text-only** to save VRAM:
- Load text-only via `Qwen3_5ForCausalLM.from_pretrained(...)` which ignores `model.visual.*` weights automatically
- Day-0 compat check must verify this works

| Component | VRAM |
|-----------|------|
| Qwen3.5-9B text-only at 8-bit | ~13 GB |
| Activation hooks (CPU offload per layer) | ~0.5 GB |
| Generation buffers (batch=1) | ~1 GB |
| **Total** | **~14.5 GB** (tight but fits 16GB with batch=1) |

**Caution**: RTX 5080 compatibility with bitsandbytes 8-bit is a compat-check item (RTX 50-series is new). If 8-bit fails, fall back to 4-bit detection (lower quality activations but fits easily).

---

## Lane C: Sleep/Wake Research (the novel contribution — after Lane B validates)

Combines Lane A data (personal workflows) with Lane B methodology (H-neuron detection).

### C.1 Wake phase (`src/engine/wake.py`)
- Qwen reads a captured problem (from Lane A experiments)
- Temperature: 0.1-0.3, single deterministic solution
- `enable_thinking=False` in `tokenizer.apply_chat_template()` — what the model confidently "knows"
- Never sees Claude's answer

### C.2 Dream phase (`src/engine/dream.py`)
**Structured dream generation** — not just generic high-temp sampling. Generate around specific failure modes:
- **Replay verified failures**: Re-present problems where Qwen previously failed (from Tier 1a data or the **operational trap library** — NOT the frozen trap benchmark). These are the highest-value replay targets.
- **False-premise variants**: Take a problem and subtly corrupt the premise (e.g., wrong error message, outdated API). Tests over-compliance — the core H-neuron mechanism.
- **Counterfactual rewrites**: Take a solved problem, change one constraint, see if Qwen adapts or halluccinates the old answer.
- **Wake/teacher disagreement cases**: Problems where wake output disagreed with `reference_solution`. These are the boundary cases where the model's confidence is malformed.
- **Standard high-temp sampling**: Temperature 0.7-1.0, 5-10 varied solutions per problem. Still useful but not the only dream type.

**Synthetic replay provenance**: All structured dream outputs (false-premise, counterfactual, replay) are NOT the same as real captured experiments. They must carry explicit metadata:
- `synthetic: true`
- `generator: "false_premise" | "counterfactual" | "replay" | "high_temp"`
- `parent_experiment_id: <UUID of the real experiment that spawned this>`
- `generation_depth: 1` (how many synthetic hops from the original real experiment)
- **Excluded from default eval sets** and from core SFT unless later independently verified (e.g., Tier 1a test passes on the synthetic variant). Otherwise the replay loop quietly contaminates the dataset.

**Synthetic depth cap**: Only real resolved experiments (`synthetic: false`) may generate synthetic children. Synthetic items CANNOT spawn further synthetic items (`generation_depth` must stay at 1). If a synthetic variant is independently verified (Tier 1a passes), it can be promoted to real episodic data (`synthetic: false`) and then generate its own children. This prevents the loop from drifting away from the real task distribution.
- **Baseline**: `enable_thinking=False` — use all-output-token mode for CETT, matching Lane B methodology
- **Ablation (later)**: `enable_thinking=True` — test whether thinking tokens improve H-neuron detection. Research hypothesis, not established method.

### C.3 Teacher reference comparison (`src/engine/compare.py`)
**Fully local** — no API calls. Compare Qwen's output against the experiment's `reference_solution` (not `final_plan` — that's planning prose, not the actual resolution). Label quality tiers:
1. **Tier 1a — Executable verification (strongest)**: Tests pass, assertions hold, deterministic checks. Only this tier feeds neuron claims and the calibrator.
2. **Tier 1b — Build/lint/error artifacts**: Build succeeds, lint passes, no error logs. Useful context but not equivalent to Tier 1a — a build can pass while the solution is still wrong.
3. **Tier 2 — Embedding similarity**: cosine distance via all-MiniLM-L6-v2 between Qwen outputs and the experiment's `reference_solution`. Good for triage, not for neuron claims alone.
4. **Tier 3 — Qwen self-judge**: Prompt Qwen to compare its own output against the reference. Weakest signal.

Classification: correct / partially_correct / incorrect per output, tagged with which tier produced the label.

### C.3.0 Verifiable vs non-verifiable data split
Many personal workflow experiments (planning, debugging, research) won't have deterministic tests. Explicitly split Lane C data:
- **Verifiable track** (has Tier 1a labels): feeds neuron claims, calibrator training, abstain labeling, and the "science" path. This subset may be small.
- **Capability-only track** (Tier 1b/2/3 labels only): feeds SFT/LoRA capability training using `reference_solution` as target. No neuron claims, no calibration metrics from this data.

**Science path lifecycle** (one threshold, no ambiguity):
- **0-99 Tier 1a + gold-set resolved examples**: Capability-only mode. SFT/LoRA on `reference_solution` targets. No calibration claims, no ECE/Brier, no neuron claims from workflow data.
- **100+ Tier 1a + gold-set resolved examples**: Science/calibration enabled. Learned calibrator trains, ECE/Brier reportable, H-neuron workflow claims allowed, abstain policy trains.

Report this explicitly. Don't pretend weak labels are strong ones.

### C.3.1 Human-reviewed gold set
Maintain a small (50-100) manually reviewed benchmark of personal workflow experiments with ground-truth correctness labels. Without this, domain-specific calibration claims stay ambiguous. Review labels once during initial capture, not every cycle.

### C.4 H-neuron calibration (`src/engine/calibrate.py`)
- Apply Lane B's activation extraction + CETT to the wake/dream outputs
- **Only use Tier 1a (test-verified) and gold-set labels for neuron claims**. Tier 1b/2/3 labels are too noisy for claiming domain-specific H-neuron effects.
- Compare: do the same neurons fire during coding hallucinations as during QA hallucinations? (Novel research question)

### C.5 Two training targets (`src/engine/tune.py`)
Both serve the self-improvement goal but do different jobs:

**Capability adapter** (SFT/LoRA — improves task performance):
- **If Unsloth (primary)**: `unsloth.FastLanguageModel.from_pretrained("Qwen/Qwen3.5-9B", load_in_8bit=True)`
- **If PEFT (fallback)**: `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", quantization_config=BitsAndBytesConfig(load_in_8bit=True))` + `peft.get_peft_model(model, lora_config)`
- **Lane B and Lane C must use the SAME framework** — H-neuron indices from one model object may not map correctly to another (Unsloth patches model internals). `model_loader.py` must be framework-aware.
- **SFT target**: The experiment's `reference_solution` field — the actual Claude/Codex output that solved the problem. NOT Qwen's own outputs that happened to pass comparison (that would be self-training noise). This is teacher-guided distillation.
- LoRA config (Qwen3.5 hybrid architecture — DeltaNet + full attention have DIFFERENT projection names):
  ```
  target_modules = [
      "q_proj", "k_proj", "v_proj", "o_proj",                              # full attention (8 layers)
      "in_proj_qkv", "in_proj_z", "out_proj",                              # DeltaNet (24 layers)
      "gate_proj", "up_proj", "down_proj",                                  # MLP (all 32 layers)
  ]
  # NOTE: in_proj_a and in_proj_b (DeltaNet decay/update, 4096→32) intentionally skipped —
  # rank-16 LoRA on a 32-output-feature layer is questionable
  # WARNING: The parent GatedDeltaNet class uses fused names (in_proj_qkvz, in_proj_ba).
  # Qwen3.5 deletes these and uses separate projections. Do NOT reference fused names.
  r = 16, lora_alpha = 16  # 1:1 ratio per Unsloth examples
  ```
  PEFT silently skips layers without matching names — no error. Without the DeltaNet names, LoRA only adapts 8/32 attention layers (25%).
  Estimated LoRA params: ~80.3M (~0.16GB), targeting 10 modules (attention + DeltaNet + MLP projections)
- Gradient checkpointing enabled

**SFT data construction policy** (`src/engine/data_prep.py`):
- **Prompt template per resolution_type** — each type gets a structured input format:
  - `code_change`: system prompt + problem + pre_solution_context + error_output + constraints → reference_solution
    **Pre/post-solution split**: Only include artifacts available BEFORE the solution was found. `error_output` and `context_files` are pre-solution context. `commands_run`, `test_results`, `diffs` are POST-solution verification — include them as labels/filtering/eval, NOT in the training prompt. Otherwise you create train/inference mismatch (model sees passing tests at training time but not at inference time).
  - `answer`: system prompt + problem + breakdown → reference_solution
  - `config_change`: system prompt + problem + error_output → reference_solution
  - `research_finding`: system prompt + problem + breakdown → reference_solution
- **Context selection for code tasks** (deterministic, auditable, **PRE-SOLUTION ONLY**):
  1. Files from `pre_solution_context` (provenance: `"error_trace"`, `"user_provided"`, or `"retrieved_pre"` — files the agent retrieved BEFORE attempting to solve, captured at task-start time)
  2. If insufficient AND `repo_dirty == false`, retrieve from repo at `repo_hash` revision using the `problem` text as query. Retrieval config pinned in `data/retrieval_config.json`. **Materialize retrieved files**: when an example becomes SFT-eligible, freeze the actual retrieved file list into `pre_solution_context` with provenance `"materialized_fallback"` (NOT `"retrieved_pre"` — these are reconstructed, not observed). Do NOT re-run retrieval at training time — the materialized list IS the prompt. This eliminates retrieval drift between data prep runs. If `repo_dirty == true`, skip repo retrieval entirely — only use files already captured in `pre_solution_context`.
  **Mechanically enforced**: `post_solution_artifacts` is a separate schema field. The training prompt builder reads ONLY `pre_solution_context` — it never sees the post-solution field. This makes leakage a code bug, not a policy violation.
  **Truncation rule** (deterministic, auditable):
  - Python files: extract the AST node (function/class) around the error/reference line. Fallback: ±30 lines.
  - Non-Python: ±30 lines window around the stack-trace line.

  **Seq_length constraint**: Training `max_seq_length=2048` means EVERYTHING (system prompt + context + reference solution) must fit in 2048 tokens. Anything beyond is silently truncated — the model never learns from truncated content.
  - System prompt: ~150 tokens
  - Pre-solution context: ~600 tokens max (AST-node extraction: function signature + body, ~80-120 tokens per file, fits 3-5 relevant functions)
  - Error output: ~200 tokens max (last 3 stack frames + error message — hard cap)
  - Reference solution (final correct code, not diff): ~600 tokens max (a 30-line function is ~200-300 tokens — more token-efficient than diff format). For multi-file changes, include each modified function/block labeled by file path. If total exceeds 600 tokens, include only the primary change (drop secondary changes to git_diff for the full picture), or split into separate training examples per file.
  - Total: ~1550 tokens (leaves ~500 token buffer — much more comfortable than diff-based budget)

  This is tight. Long reference solutions and deep stack traces WILL be truncated. Alternatives if too restrictive:
  - Increase `max_seq_length=4096` (adds ~4-5GB activation VRAM — **will OOM on 16GB**, only viable with the 4B fallback)
  - Split long solutions into multiple training examples (more complex data_prep)
- **Minimum completeness gates** — resolved experiments must meet these before becoming SFT-eligible:
  - `code_change`: requires `repo_hash` + at least one entry in `pre_solution_context` (NOT `post_solution_artifacts` — the prompt builder only reads pre-solution files, so eligibility must require actual prompt-available grounding) + at least one of `error_output`/`test_results`/`commands_run`
  - `answer`: requires `reference_solution` (non-empty)
  - `config_change`: requires `reference_solution` + `error_output`
  - Incomplete records stay in the store for retrieval but are excluded from `data_prep.py` training data
- **Minimum SFT dataset size**: 300-500 resolved experiments before first meaningful LoRA training run. Below 300, only smoke-test the pipeline mechanically — don't expect real behavior change.
- **Mixing ratios** (adjust as data accumulates):
  - Code sub-domains (Python web, Android, config, etc.): 60% total, cap any single sub-domain at 40%
  - Direct answers: 20%
  - Research/config: 10%
  - Abstention examples: 10% (keep small to avoid over-hedging). **These must be synthetically generated** — real Claude conversations almost never contain abstention. Generate by taking problems where Qwen consistently fails (from replay priority) and creating hedged response examples.
- **Domain balancing**: If one domain (e.g., Python) dominates, cap at 40% of total and upsample underrepresented domains.
- Abstention examples use the SAME prompt templates but with hedged/abstain `reference_solution` text — they are not a separate format.

**Calibration layer** (H-neuron probe/gating — improves reliability):
- Forward-hook H-neuron intervention from Lane B, applied during inference (research runtime only, full H-neuron path only)
- Adaptive gating per C.8: multi-signal gate with MVP starting at `alpha = f(detector_risk)` only. Full multi-signal version (detector + divergence + trap + prompt type) added after validation. Does NOT gate on divergence alone.

**Abstain policy training** (`src/engine/abstain.py`):
- **Labels**: ONLY from verifiable track data:
  - Tier 1a failures (tests fail) where ALL dream samples AND wake answer also fail = "should abstain"
  - Gold-set human-labeled "should abstain" examples
  - Do NOT use Tier 2/3 semantic judgments for abstain labels — those are too noisy and will teach hedging from weak signals
- **Training target**: SFT on natural language abstention responses (e.g., problem → "I'm not confident enough to answer this accurately — here's what I think but you should verify: ..."). NO special tokens — use normal language so the behavior survives GGUF export and works in LM Studio.
- **Trained INTO the capability LoRA** (not a separate adapter). This means abstention behavior carries into the daily GGUF model, which is intended — the user wants LM Studio to benefit from knowing when to hedge.
- **Threshold**: Learned from the calibrator in the research runtime — determines which examples become abstain training data. Tune on **validation set** (NOT the frozen test set) to maximize accuracy-coverage tradeoff (selective prediction curve).
- **Non-regression guard**: Track abstain rate per evaluation cycle. If abstain rate increases >15% without proportional accuracy gain on remaining answers, the abstain policy is too aggressive — roll back.

**Two output artifacts** (path-dependent):
1. **Research runtime** (Python):
   - Full H-neuron: fp16 model + forward hooks + adaptive gating + probe confidence + runtime abstention
   - Detector-only: fp16 model + detector scoring for training data selection and eval analysis (NOT runtime inference)
   - No H-neuron: fp16 model, no probe features
2. **Daily-use GGUF** (LM Studio):
   - Full H-neuron: merge LoRA to fp16 → bake H-neuron weight scaling → export GGUF Q8. Carries: capability + hedging + baked scaling.
   - Detector-only: merge LoRA to fp16 → export GGUF Q8. Carries: capability + hedging. No baked scaling.
   - No H-neuron: same as detector-only.
   All paths: GGUF carries learned hedging/abstention language (via LoRA). Does NOT carry: forward hooks, adaptive alpha, probe-triggered runtime abstention, confidence scoring.

**Canonical model state**: fp16/merged safetensors is the master copy. GGUF is a downstream export, never the source of truth.

### C.5.1 Ablation matrix (required before claiming "both methods help")
Evaluate all five variants on the same held-out set:
| Variant | LoRA (capability) | Abstain training | H-neuron intervention | What it tests |
|---------|-------------------|------------------|-----------------------|---------------|
| Base | — | — | — | Baseline |
| LoRA-only | ✓ | — | — | Pure capability gain from distillation |
| LoRA+abstain | ✓ | ✓ | — | Capability + hedging behavior |
| H-neuron-only | — | — | ✓ | Reliability gain from intervention alone |
| LoRA+H-neuron | ✓ | — | ✓ | Capability + intervention, no abstain |
| Combined | ✓ | ✓ | ✓ | Full system |

The LoRA+H-neuron arm isolates whether combined gains come from intervention or from hedging behavior.

"LoRA-only" means pure capability SFT on `reference_solution` targets — NO hedging/abstention mixed in.

Track accuracy, hallucination rate, calibration (ECE), and response type distribution (see C.7) for each. If combined ≤ LoRA-only, the calibration layer isn't earning its keep.

**Path-specific ablation and export:**

**Full H-neuron path** — all six variants in the matrix above. GGUF-level ablation on: LoRA-only, LoRA+abstain, LoRA+H-neuron, Combined. Re-detect + bake required for every H-neuron variant.

**Detector-only path** — reduced matrix (drop all H-neuron intervention arms):
| Variant | LoRA | Abstain | Detector-prioritized data selection | What it tests |
|---------|------|---------|-------------------------------------|---------------|
| Base | — | — | — | Baseline |
| LoRA (random data) | ✓ | — | — | Capability gain from unprioritized SFT |
| LoRA (detector data) | ✓ | — | ✓ | Does detector-driven replay priority improve the resulting LoRA? |
| LoRA+abstain (random data) | ✓ | ✓ | — | Capability + hedging, unprioritized |
| LoRA+abstain (detector data) | ✓ | ✓ | ✓ | Full detector-only system |
The "detector-prioritized data selection" column means: the detector score drove which experiments were replayed during training. All variants are evaluated identically at inference time — no detector at inference. The test isolates whether smarter data selection (informed by the detector) produces a better LoRA. **Fixed training budget**: random and detector variants must use the same number of examples and training steps — only the selection order/priority differs. Otherwise you're comparing different effective datasets, not data selection quality. GGUF-level ablation on: LoRA (random), LoRA (detector data), LoRA+abstain (random), LoRA+abstain (detector data). All four exported as GGUF Q8 and evaluated — the detector-data benefit must survive quantization to count as a product improvement.

**No H-neuron path** — same as detector-only but also drop detector-based confidence/gating from the research runtime.

### C.5.2 Re-detect AND re-calibrate after LoRA (full H-neuron path only)
Both the L1 intervention map AND the L2 detector probe must be refreshed after LoRA changes internal activations. A base-model detector/probe is not guaranteed valid on a LoRA-merged model. Re-detection must run on the **exact merged adapter variant being promoted**. For each ablation variant that includes H-neuron intervention:
1. Train the specific LoRA variant (capability-only, or capability+abstain)
2. Merge that specific LoRA into fp16
3. **Re-run Lane B detect** (CETT extraction + L1 intervention probe + L2 detector probe) on the merged model. **VRAM note**: fp16 merged model doesn't fit 16GB. Re-quantize by: save merged fp16 to disk (~19GB safetensors) → reload with `load_in_8bit=True, device_map="auto"` (bitsandbytes quantizes on-the-fly). Requires ~40GB total free disk (original + merged). Takes ~30-90 minutes for 1000 samples.
4. Grid search alpha on the updated map
5. Bake the new scaling into the merged fp16 weights
6. Export GGUF Q8

If promoting Combined, re-detect runs on the merged LoRA+abstain model, not the LoRA-only model. This adds ~1 hour per promotion cycle but prevents shipping stale neuron maps.

**Detector-only path**: Still requires re-training the L2 detector probe on the merged model (detector drives replay priority and eval analysis — stale detector on post-LoRA activations is unreliable). Skip L1 intervention probe and weight scaling steps.

**No-H-neuron path**: Skip this entire section. Merge LoRA → export GGUF directly.

### C.5.3 Three-way evaluation split
To prevent evaluation leaking into tuning:
- **Train**: Most data. Feeds SFT, abstain training, H-neuron detection.
- **Validation**: Used for threshold tuning, calibrator fitting, alpha search, abstain sensitivity. Touched repeatedly during development — results are directional, not final.
- **Frozen test (rotating)**: Used only at promotion time. Produces reportable numbers. However, repeated promote/reject decisions against the same frozen set will slowly overfit to it. Treat as trustworthy for ~5-10 promotion cycles, then **refresh**: retire the current frozen test into validation, carve a new frozen test from recent data.
- **Archival benchmark (permanent)**: A separate small set (starts at 20-30 examples, grows over time) that is NEVER rotated, NEVER used for promotion decisions. Evaluated only occasionally (~monthly) as a **coarse health check** — not a source for fine-grained trend lines at this size. Treat as a sanity signal: "is the project directionally improving or regressing?" For trustworthy trends, the archival set needs to grow to 100+ examples over time.
- **Trap benchmark** (regression test for learned lessons): A curated list of known failure modes with targeted categories:
  - **False-premise acceptance**: does the model accept obviously wrong assumptions?
  - **Confident fabrication**: does it invent APIs, functions, or facts?
  - **Bad user-assumption compliance**: does it comply with a subtly wrong user request instead of pushing back?
  - **Tool/API hallucination**: does it generate plausible but nonexistent code patterns?
  - **Repeated past mistakes**: specific failures from Lane B/C history
  Each trap has a trigger query and expected prevention behavior. Evaluated per promotion cycle. Start with 5-10 traps, grow over time. This directly tests what H-neurons are supposed to control: over-compliance behaviors.

- **Over-compliance mini-benchmark** (`src/eval/compliance_bench.py`): A small structured set (20-30 prompts) specifically designed to test over-compliance in coding/workflow contexts. Bridges the gap between TriviaQA-style findings and real-world behavior. Include: false premises about APIs, wrong error messages to debug, subtly incorrect code to review. This is how Lane B neuron findings get validated in Lane C's domain.

**Two-tier trap system** (critical separation):
- **Operational trap library** (`data/trap_library/`): A living collection of known failure patterns. CAN be used for structured dream replay, runtime trap matching (C.8 gating), and active replay priority. Grows over time as new failures are discovered. This is the working tool.
- **Frozen trap benchmark** (`data/frozen_benchmarks/traps/`): A separate, fixed snapshot of trap prompts used ONLY for evaluation. NEVER consumed by replay, dream generation, SFT, prompt generation, OR runtime gating. Read-only. If you need to add new traps to the benchmark, create a new versioned snapshot.
- **Frozen over-compliance benchmark** (`data/frozen_benchmarks/compliance/`): Same rule — eval only, never consumed by any training or runtime path.

The operational library and frozen benchmarks may initially contain similar traps, but they diverge over time: the library grows and changes, the benchmark stays fixed.

**General contamination rule**: If a failure is discovered only because the model missed a prompt from ANY frozen eval asset (frozen trap benchmark, frozen compliance benchmark, rotating frozen test, archival benchmark), that failure CANNOT feed into the operational trap library, replay queue, SFT training, synthetic generation, or runtime heuristics until that eval snapshot is retired/versioned. This prevents all frozen eval assets from indirectly influencing training or runtime.

**Group-aware splitting**: Experiments about the same underlying task/bug/feature must stay in the same split — even across sessions. Split by `task_group_id`, not by individual experiment.

**Split immutability rules**:
- Once a `task_group_id` is assigned to a split (train/validation/frozen test/archival), that assignment is **permanent** and stored in a `data/split_assignments.json` lookup table.
- All future experiments with the same `task_group_id` (including resolved children of unresolved parents) inherit the parent group's split assignment automatically.
- **New task_group_ids** arriving after the initial split is created go to **train only** by default. They only move into validation/frozen test/archival during scheduled benchmark refreshes (when the frozen test rotates).
- This prevents the agent from accidentally contaminating promotion benchmarks during autonomous operation.

If data is small, keep frozen test small (20-30 examples) but never reuse it for tuning.

### C.5.4 Promotion gate (path-specific)
All evaluated on the **frozen test set** (never used for tuning — see C.5.3):

**Full H-neuron path** — three stages:
1. Research runtime check: candidate fp16 (with hooks + gating) beats previous version
2. Re-detect verification: updated H-neuron map is still sparse and stable (same gate criteria as B.6)
3. GGUF regression check: exported GGUF Q8 with baked scaling evaluated on same frozen test

**Detector-only path** — two stages:
1. Research runtime check: candidate fp16 beats previous version (detector used for eval scoring, not runtime gating)
2. GGUF regression check: exported GGUF Q8 (no baked scaling) evaluated on same frozen test

**No H-neuron path** — two stages:
1. Candidate fp16 beats previous version
2. GGUF regression check

Keep base, candidate, and best model versions (both fp16 and GGUF) so you can roll back.

**GGUF evaluation method**: GGUF models can't be loaded via transformers/Unsloth. Use one of:
- **`llama-server` + OpenAI client** (recommended): Start `llama-server -m model.gguf`, call via `openai.OpenAI(base_url="http://localhost:8080/v1")`. Works now with Qwen3.5. Note: `llama-cpp-python` does NOT support Qwen3.5 (issue #2137) — use the server, not the Python bindings.
- LM Studio local API at `localhost:1234` (OpenAI-compatible, if LM Studio is running)
- `lm-evaluation-harness` with GGUF backend (may hang when reconstructing tokenizer from GGUF — always pass explicit `--tokenizer Qwen/Qwen3.5-9B`)
GGUF eval is ~5-10x slower than native HF — acceptable for promotion gates (run once per cycle).

### C.6 Confidence signal (`src/engine/confidence.py`) — path-dependent

**Full H-neuron path:**
- **Primary: Detector probe probability** — L2 1-vs-1 detector sigmoid. `confidence = 1 - P(hallucination)`. Must be recalibrated on Tier 1a + gold-set workflow labels before standalone use.
- **Secondary: Sleep divergence** — embedding distance wake vs dream cloud. `confidence = 1 / (1 + normalized_divergence)`.
- **Supplementary: Token logprobs** — useful as calibrator feature, not standalone.
- **Required: Trained calibrator** — logistic regression on [probe_prob, divergence, avg_logprob, answer_length] → calibrated confidence for ECE/Brier.

**Detector-only path:**
- Same as full H-neuron but detector is used ONLY for eval analysis (confidence-stratified accuracy), not runtime. Calibrator still trainable if enough labeled data exists.

**No H-neuron path:**
- No detector probe available. Confidence signals limited to: sleep divergence + token logprobs only.
- Calibrator trained on [divergence, avg_logprob, answer_length] (no probe feature). Weaker but still usable.
- Active replay uses wake/teacher disagreement + Tier 1a failures only (no detector score).

**All paths — minimum data**: 100 Tier 1a + gold-set labeled examples for calibrator. Fallback: no ECE/Brier, report only accuracy, hallucination rate, coverage, response type distribution.

### C.7 Evaluation metrics (`src/eval/metrics.py`)
Track across sleep cycles:
- **Accuracy rate**: % correct on held-out experiments
- **Hallucination rate**: % containing provably false statements
- **Calibration**: ECE and Brier score using the **learned calibrator** from C.6 (NOT raw logprobs)
- **Per-domain performance**: accuracy by tag (python, android, web, etc.)
- **Response type distribution** — classify each output into one of:
  - **Normal answer**: full confident response
  - **Hedged answer**: answers but flags uncertainty (e.g., "I think X but you should verify...")
  - **Hard abstain**: refuses to answer (e.g., "I'm not confident enough to answer this")
  Track counts + accuracy within each bucket separately. Do NOT combine hedged and hard abstain into one number — they are very different behaviors.
- **Selective prediction metrics** (primary for calibration evaluation):
  - **Coverage**: % of queries the model answers (normal + hedged) vs abstains
  - **Risk at coverage**: error rate on the queries the model chose to answer
  - **Per-bucket accuracy**: accuracy within normal, within hedged, separately
  - A model that hedges constantly can look calibrated but is actually unhelpful. Coverage-risk tradeoff is the real measure.

### C.7.1 Active replay priority — path-dependent

**Full H-neuron and detector-only paths:**
- **Detector probe score**: higher hallucination risk → higher priority (but not sole signal)
- **Wake/teacher disagreement**: Qwen's wake output diverged from `reference_solution` → high priority
- **Tier 1a failure**: tests failed on Qwen's output → highest priority
- **Operational trap library hit**: matches a pattern in the operational trap library (NOT the frozen benchmark) → high priority
- Combine as: `priority = w1*detector + w2*disagreement + w3*tier1a_fail + w4*trap_library_hit`, tune weights on validation set
- **Missing-signal policy**:
  - Verifiable tasks: use all four signals
  - Non-verifiable tasks: detector + disagreement only, set tier1a_fail=0
  - **Frozen benchmark items are NEVER eligible for replay.** Operational trap library examples: trap_library_hit + detector only.
  - Missing signals default to 0 (neutral)

**No H-neuron path:**
- No detector probe available. Priority uses: `priority = w2*disagreement + w3*tier1a_fail + w4*trap_library_hit` (three signals, no detector)
- Same missing-signal policy, minus the detector column
- Low priority (all signals low) → skip or deprioritize
- This is a data selection mechanism, NOT training loss weighting. The model trains normally on selected examples.

### C.8 Core runtime features (FULL H-NEURON PATH ONLY)

1. **Adaptive H-neuron gating (promoted from extension to core)** — Static alpha is too blunt for a 9B model. Smaller models are more likely to get capability damage from always-on suppression. Alpha should be a function of **multiple signals**, not just divergence:
   - `alpha = f(detector_risk, divergence, trap_category, prompt_type)`
   - **Detector risk**: L2 probe hallucination probability (high risk → suppress more)
   - **Sleep divergence**: embedding distance between wake and dream cloud (high divergence can mean genuine uncertainty OR multiple valid solutions — not always bad)
   - **Trap category**: if the prompt matches a known trap pattern, suppress more aggressively
   - **Prompt type**: code tasks with multiple valid implementations should tolerate higher divergence than factual questions
   - **Live inference feature extraction** (what's available at runtime without labels):
     - `detector_risk`: always available — single forward pass through the L2 probe
     - `divergence`: available if dream samples are generated (adds latency — skip for low-stakes queries)
     - `trap_category`: use a lightweight BM25/embedding match against the **operational trap library** (NOT the frozen benchmark — that would contaminate evaluation). If the input is similar to a known trap pattern, flag it. Simple and fast.
     - `prompt_type`: infer from a simple classifier or keyword heuristic (contains code block → code task, question mark → factual, etc.)
   - **Minimum viable gate** (ship first): `alpha = f(detector_risk)` only. Add divergence and trap matching after the base gate is validated. Do NOT block shipping on the full multi-signal version.
   - Tune weights on the validation set. Do NOT suppress on divergence alone — that kills creativity and alternative valid solutions.

### C.8.1 Local proxy runtime (`src/runtime/server.py`) (FULL H-NEURON PATH ONLY)
The research runtime as a **local inference server**:
- OpenAI-compatible API at `localhost:8081` (doesn't conflict with LM Studio's port)
- Loads the fp16 model with forward hooks, adaptive gating, probe confidence scoring
- Pi or other local agents can call it for reliability-sensitive tasks
- Keep GGUF/LM Studio as the portable convenience artifact for low-stakes use
- Build AFTER Lane B passes the full H-neuron gate and adaptive gating is validated
- **Detector-only and no-H-neuron paths skip C.8 and C.8.1 entirely.** Detector-only uses the detector for replay priority and confidence scoring in the evaluation pipeline, not as a runtime inference feature.

### C.9 Further extensions (same infrastructure, no new model methods)

These use the existing wake/dream/hook pipeline:

1. **Boundary-only replay** — Only train on cases where wake, dream, and teacher disagree. Skip easy cases. This is just a data filter on the compare phase output — no new infrastructure, much more data-efficient.

2. **Domain H-neuron atlas** — Run Lane B detection separately on different QA domains (trivia, code, science). Compare neuron maps for overlap. Same pipeline, different input data. Tests: global over-compliance core vs domain-local hallucination bundles.

### Roadmap (future — requires different methods or more complexity)

For reference only. Not in the active build plan.

4. **Pre-answer router** — Predict answer/clarify/abstain/dream before generating. Needs a separate classifier head.
5. **Two-timescale sleep** — Ephemeral steering hooks vs consolidated LoRA. Needs session-tracking architecture.
6. **Premise-resistance training** — Adversarial false-premise data generation. Needs crafted dataset.
7. **Counterfactual invariance** — Prompt perturbation training. Needs systematic perturbation generator.

Sources: H-Neurons (2512.01797), Self-Refine (2303.17651), HalluciBot (2404.12535), Reflexion (2303.11366), SPIN (2401.01335), Intrinsic Self-Correction (2406.15673)

---

### VRAM strategy for Lane C training
**All-9B strategy**: One model end-to-end. 8-bit frozen base + bf16 LoRA adapters — the standard PEFT recipe before QLoRA. Unsloth's warning is specifically about QLoRA 4-bit, NOT 8-bit. No 8-bit warning exists. INT8 quality drop vs bf16 is expected to be small (no Unsloth warning exists for 8-bit, unlike 4-bit). Verify empirically in compat check — do not assume a specific number.

**Training VRAM breakdown** (9B 8-bit base + bf16 LoRA + gradient checkpointing):
| Component | Type | VRAM |
|-----------|------|------|
| Frozen 9B base (8-bit) | weights storage | ~9.5-10 GB |
| LoRA adapters (bf16, ~80.3M params) | weights | ~0.16 GB |
| Optimizer states (8-bit AdamW) | training overhead | ~0.5 GB |
| Activations (grad checkpointing, seq=2048, bs=1) | training overhead | ~1.5-2 GB |
| KV cache + CUDA context + bitsandbytes metadata | runtime overhead | ~1-1.5 GB |
| **Total** | | **~13-14.5 GB** (~1.5-3GB headroom) |

| Phase | Config | VRAM (total incl. overhead) |
|-------|--------|---------------------------|
| Detection (Lane B) | 9B 8-bit text-only inference, batch=1 | ~13 GB (may be ~8-9 GB if vision weights truly excluded — **measure in compat_check**) |
| Training (Lane C) | 9B 8-bit base + bf16 LoRA | ~13-14.5 GB |
| Inference (wake/dream) | 9B 8-bit text-only | ~13 GB |
| **LoRA merge** | fp16 reload + merge — see merge note below | **Does NOT fit 16GB GPU** |
| Deployment | GGUF Q8 via LM Studio | N/A |

**Merge step — fp16 doesn't fit GPU**: Merging requires fp16 weights (~18-19GB). On a 16GB GPU, merge must happen on CPU:
- **Unsloth (primary)**: `save_pretrained_merged(save_method="merged_16bit")` — may handle CPU offloading internally. Test in compat_check. This is a known failure point (#4294, #4166).
- **PEFT (fallback)**: `model = AutoModelForCausalLM.from_pretrained(..., device_map="cpu", torch_dtype=torch.float16)` → `PeftModel.from_pretrained(model, adapter_path)` → `model.merge_and_unload()` → `model.save_pretrained(...)`. Requires ~20GB free system RAM.
- Disk note: all safetensors shards (~19.3GB) downloaded regardless of text-only loading (weights interleaved across shards).

**Hard constraints for 16GB**:
- `batch_size=1`, `max_seq_length=2048` (see seq_length constraint below)
- Gradient checkpointing required: PEFT path uses `model.gradient_checkpointing_enable()`; Unsloth path uses `use_gradient_checkpointing="unsloth"` (activation offloading)
- `dataset_num_proc=1` for Windows stability
- Verify in compat_check that the Qwen3 8-bit bug fix (Unsloth PR #3806, fixing issue #3501) is present

**Day-0 compat check must verify**: Load 9B 8-bit → add LoRA → run one training step with gradient checkpointing → check peak VRAM. If failure:

**Two orthogonal fallback dimensions** (try framework fallback first, then model-size):
1. **Framework fallback** (same 9B model, different tooling): Unsloth → Standard PEFT + HF Trainer → DeepSpeed ZeRO-3 (3 items — merge method alternatives are in B.4, not here)
2. **Model-size fallback** (if VRAM is the bottleneck regardless of framework): 9B → all-4B (Qwen3.5-4B bf16 LoRA, ~11GB)

Try framework fallback first. Only drop to 4B if 9B doesn't fit 16GB in ANY framework. Do NOT use a 9B/4B split — H-neuron maps and adapters cannot transfer between architecturally different models.

**What NOT to use**: QLoRA 4-bit (explicitly flagged by Unsloth for Qwen3.5). bf16 full LoRA on 9B (~22GB, doesn't fit). Generic CPU weight offloading (Unsloth's offloading is activation checkpointing only).

**Disk space note**: `save_pretrained_merged` from 8-bit base downloads ~18GB of fp16 weights (can't merge LoRA into quantized weights losslessly). Pre-download the fp16 base once to avoid surprise disk usage during promotion cycles.

**Merge precision mismatch**: LoRA adapters trained against 8-bit quantized weights learned corrections relative to quantized representations. Merging into fp16 introduces a precision gap. Cannot call `merge_and_unload()` while model is loaded in 8-bit — must reload base in fp16, load adapter, then merge. **Quality comparison step**: run the same eval with (a) 8-bit base + LoRA adapter and (b) merged fp16 model. Report the gap — usually small but must be measured.

**DeepSpeed fallback** (final resort): ZeRO-3 with CPU offloading can fit full bf16 9B in <1GB GPU by offloading parameters to CPU. Requires ~64-128GB system RAM. **On native Windows**: async_io (libaio) is unavailable, making CPU offloading significantly slower (5-10x, not 3-5x). If DeepSpeed is needed, **WSL2 is essentially mandatory** for acceptable training speed. Uses standard HuggingFace Trainer + PEFT. Zero quality compromise.

**Windows note**: Set `dataset_num_proc=1` in all data loading to avoid crashes (per Unsloth Windows docs).

---

## Infrastructure (build AFTER Lane B shows results)

### Agentic Orchestration
**Pi coding agent** is the primary framework (native Windows support). Hermes Agent requires WSL2 — optional.

**Daemon** (`src/orchestrator/daemon.py`):
- Background process watching `experiments/` for new JSON files
- Triggers a sleep cycle when N new experiments accumulate (default: 10)
- Checks GPU availability before loading model
- `uv run python -m src.orchestrator.daemon start`

**Agent API** (`src/orchestrator/agent_api.py`):
- CLI returning structured JSON — any agent can call it
- Commands: `status`, `run-cycle`, `results`, `experiments`
- Pi extension wraps these via its bash tool

**Terminal UI** (`src/ui/dashboard.py`):
- Rich Live display: cycle status, experiment queue, H-neuron counts, accuracy trend

### Distributed Correction Sharing (build AFTER Lane C validates)
Two tiers of shareable output, split by portability:

**Broadly shareable** (any model/framework):
- Labeled experiment data (problem + reference_solution + Tier 1a labels)
- Training recipes (hyperparameters, consistency thresholds, methodology)
- Evaluation results and ablation matrices
- Domain H-neuron atlas findings (which domains overlap, general patterns)

**Model-specific** (exact base model + version only):
```json
{
  "model_id": "Qwen3.5-9B",
  "base_model_revision": "abc123...",
  "tokenizer_version": "...",
  "chat_template_hash": "sha256:...",
  "lane_b_outcome": "full_h_neuron | detector_only | no_h_neuron",
  "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"],
  "training_recipe_hash": "sha256:...",
  "eval_dataset_hash": "sha256:...",
  "lora_adapter": "base64-encoded safetensors",
  "h_neuron_map": null,
  "alpha_values": null,
  "domain_tags": ["python", "android"],
  "metrics": {"accuracy_before": 0.62, "accuracy_after": 0.78}
}
```
**Correction type** — path-dependent:
- Full H-neuron: `h_neuron_map` and `alpha_values` populated, `"applies_to": "base"|"merged_lora"`, `"application_order": ["load_base", "apply_adapter", "merge", "apply_neuron_scaling"]`
- Detector-only: `h_neuron_map` and `alpha_values` are null. Package contains only the LoRA adapter + training recipe. No neuron scaling to apply.
- No H-neuron: same as detector-only.

`lane_b_outcome` field tells collaborators which path produced this package. Without it, someone might try to apply non-existent neuron scaling.

**Two share formats** (path-specific canonical definitions):
- **Canonical layered package** (for collaboration):
  - Full H-neuron: base model hash → adapter → neuron map → alpha values → application order
  - Detector-only: Qwen3.5-9B hash → adapter → training recipe (including detector-prioritized data selection config). No neuron map. Equally canonical — not a second-class artifact.
  - No H-neuron: Qwen3.5-9B hash → adapter → training recipe. Simplest package.
- **Optional merged convenience artifact**: Pre-merged GGUF / fp16 safetensors for people who just want to run it. Easy but hard to compare or improve.

Provenance fields let collaborators verify compatibility before merging. Without matching `base_model_revision` + `tokenizer_version`, corrections are not safely mergeable.

**Default collaboration path** (safe):
- Compare packages (inspect neuron maps, metrics, domain tags)
- Replay locally: apply a community correction to your own model, evaluate on your validation set
- Only adopt if it improves your metrics

**Research-only aggregation** (advanced, not the default):
- Average alphas across contributors (requires exact base model + adapter match)
- Merge LoRA via TIES/DARE (requires matching target modules + base revision)
- Weight by contributor validation metrics
- This is experimental and should be validated per-merge, not applied blindly

Data/recipes are portable across models; weights/neuron maps are not.

---

## Build Order

```
Phase 0: Foundation (pyproject.toml, schema, LanceDB, embeddings)
    │
    ├── Lane A: Capture skill + CLI (parallel with Lane B)
    │
    └── Lane B: H-Neuron adaptation on TriviaQA
            │
            ├── Full H-neuron path → Lane C with intervention + gating + weight scaling
            │
            ├── Detector-only path → Lane C with detector-prioritized data selection + eval analysis (training/eval aid only, NOT runtime)
            │
            └── No H-neuron path → Lane C with LoRA-only (no H-neuron) (no probe features)

    All three paths → Orchestrator + UI + Sharing (after Lane C shows gains)
```

**Estimated order of implementation**:
1. Phase 0: Foundation (~1-2 sessions)
2. Lane A + Lane B in parallel (~3-5 sessions each)
3. Lane B gate check
4. Lane C (~3-5 sessions)
5. Infrastructure (daemon, agent, UI) (~2-3 sessions)
6. Distributed sharing (~2 sessions)

---

## Smoke Test vs Real Validation

- **Smoke test** (5 manual experiments): Verifies the pipeline mechanically works. NOT for scientific conclusions.
- **Real validation** (Lane B, 1000+ TriviaQA samples): Answers whether H-neuron detection works on Qwen3.5 9B.
- **Novel validation** (Lane C): Answers whether the sleep/wake methodology improves coding calibration.

---

## Verification

1. **Phase 0**: `uv sync` succeeds; LanceDB CRUD works; embeddings generate
2. **Day-0 gate**: `uv run python -m src.reproduce.compat_check` passes — Qwen3.5-9B loads at 8-bit, `enable_thinking` works, hooks register, VRAM < 16GB
3. **Lane A**: `/dream-forge` in Claude Code produces valid JSON; artifacts store in separate columns; semantic search works on problem/solution fields
4. **Lane B pilot**: Run 5,000 TriviaQA questions, measure consistency yield. If 10/10 filter yields <500/class, relax to 9/10 or 8/10.
5. **Lane B spans**: Verify case-insensitive alias matching + `offset_mapping` correctly identifies answer token indices. Track match rate.
6. **Lane B full** (path-dependent):
   - Full H-neuron: 1000+ balanced samples, detector accuracy > 65%, intervention probe sparse (< 0.1%), stable across 3+ seeds, intervention reduces hallucinations, retained accuracy drops <5%, abstain rate increase <10%
   - Detector-only: detector accuracy > 55%, useful confidence signal. Skip intervention metrics.
   - No H-neuron: skip Lane B entirely beyond sanity gate.
7. **Lane B export** (full H-neuron path only): Merge to fp16 → apply weight scaling → export GGUF Q8 → verify hook-based and baked intervention produce equivalent results on held-out data. Detector-only/no-H-neuron paths skip this.
8. **Leakage control tests** (core correctness):
   - Prompt builder reads ONLY `pre_solution_context`, never `post_solution_artifacts` — unit test with a mock experiment containing both
   - `repo_dirty == true` disables fallback retrieval — unit test confirms no retrieval call when dirty
   - Materialized fallback files tagged `"materialized_fallback"`, not `"retrieved_pre"` — schema validation test
   - Git-based provenance: `git diff` correctly identifies post-solution files, `git_start_hash` matches recorded starting state
   - Frozen benchmark items never appear in replay queue, operational trap library, or training data — integration test scanning all pipeline outputs
9. **Lane C** (path-specific): Wake/dream outputs generate; artifact-first labeling works; verifiable/non-verifiable split applied; capability LoRA trains on `reference_solution` with pre-solution-only context; path-appropriate ablation matrix shows each component's contribution; promotion gate passes on both fp16 and GGUF
9. **Infrastructure**: Daemon starts and watches experiments/; agent API returns valid JSON; dashboard renders
10. **End-to-end**: Semi-autonomous after bootstrap. Requirements before the science path activates: initial gold set (50-100 human-reviewed examples) + 100 Tier 1a + gold-set labeled examples (the same threshold C.6 requires for the calibrator). After bootstrap, cycles run autonomously: new experiments → full pipeline → GGUF export → metrics logged. Before bootstrap, only capability SFT runs (no calibration claims).
