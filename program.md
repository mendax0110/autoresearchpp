# autoresearch-cpp — Agent Program

## Role

You are an autonomous research agent. Your job is to improve the model trained
by `train.cpp` (built from `src/` and `include/`). The metric is **val_bpb**
(validation bits-per-byte). Lower is better.

## What you may modify

- Any file under `src/` or `include/` **except** `src/main.cpp` and
  `include/Config.h` (those define the harness contract).
- You may add new `.cpp` / `.h` files if useful.

## What you must not modify

- `CMakeLists.txt` or `test/CMakeLists.txt`.
- Files under `test/`.
- `program.md` itself.

## Experiment loop

1. Read `experiments.jsonl` — understand what has been tried and what the
   current best val_bpb is.
2. Form one hypothesis about what might improve val_bpb.
3. Edit the relevant source files.
4. Build:
   ```
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=<path> && cmake --build build -j$(nproc)
   ```
5. Run a timed experiment:
   ```
   ./build/autoresearch --device cpu --budget-secs 300
   ```
6. Compare the new val_bpb to the previous best.
7. If improved: keep changes and log your hypothesis as `note` in the record.
   If not improved: revert and note why it failed.
8. Repeat.

## Ideas to explore (not exhaustive)

- Rotary position embeddings instead of learned positional embeddings.
- RMSNorm instead of LayerNorm.
- SwiGLU activation in the MLP.
- Muon optimizer for weight matrices, AdamW for everything else.
- Adjusting depth, headDim, numHeads for better parameter efficiency.
- Gradient checkpointing to allow larger batches on CPU.
- Learning-rate warmup and cosine decay within the budget.

## Constraints

- The training budget is **fixed at 300 s wall-clock** (excluding build time).
- Do not change `--budget-secs` when running experiments.
- Keep the code compiling cleanly under C++20 with `-Wall -Wextra`.
- Do not add external dependencies beyond LibTorch.

## Output format

After each run, the harness appends one JSON line to `experiments.jsonl`.
Summarise each experiment as a short sentence in the `note` field before the
run completes (pass it via code if you can; otherwise update the last line
manually).