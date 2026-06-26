# TRT FP32 baseline for KernelBench Level 4

## Context

The TRT baseline pipeline (`TRT_Baseline/`) produces a TensorRT FP32 reference time for
each KernelBench problem, written to `results/timing/RTX-4090-D/baseline_time_trt.json`
and consumed by `tool_scripts/fill_summary*.py`. It was originally **hard-scoped to L1+L2**:
the only knob is `levels: [...]` in `TRT_Baseline/config.yaml`, and the export stage
(`convertONNX/convert_all.py`) walks *only* the dirs in that list — so `--filter level4/7`
alone never reaches Level 4. Goal: produce an L2-style `trt_baseline` entry for an L4 file,
e.g. `level4/7_gpt2_bs32_seq256.py`.

No code changes are needed — both stages already support arbitrary levels via the config +
`--filter`. We add a Level-4 config and run the existing two-stage pipeline. Result lands as
a new `level4` key in the same JSON, alongside L1/L2.

L4 differs from L1/L2 in three ways that matter (none block the approach):
- Input is `int64` token IDs (`32×256`), not float tensors — the main numeric-parity risk.
- Output logits are ~1.6 GB (`32×256×50257`) — fits the 24 GB 4090 under the 16 GB workspace cap.
- `Model` is a HuggingFace `AutoModelForCausalLM` (gpt2) — not cached, so stage 2 downloads
  ~500 MB (needs internet) and ONNX export is heavier than the toy L1/L2 modules.

## Step 1 — Add `TRT_Baseline/config_l4.yaml`

Copy `TRT_Baseline/config.yaml` verbatim, changing only `levels: [4]`. Keep
`output_path: results/timing/RTX-4090-D/baseline_time_trt.json` so the L4 entry is
**appended** next to L1/L2 (stage 3 merges into the existing JSON; `build_and_time.py:226`).
A separate config is non-destructive — the committed L1+L2 `config.yaml` is left untouched.

Change the tolerance to `1e-3`.

## Step 2 — Stage 2: export gpt2 → ONNX

```bash
python TRT_Baseline/convertONNX/convert_all.py \
  --config TRT_Baseline/config_l4.yaml \
  --filter level4/7
```
- `--filter` merges into the existing `manifest.json` (does not wipe L1/L2 — `convert_all.py:166-172`).
- Success = `manifest.json` has `level4/7_gpt2_bs32_seq256.py` with `"status": "ok"` and
  `convertONNX/onnx_out/level4/7_gpt2_bs32_seq256.onnx` exists.

## Step 3 — Stage 3: build TRT engine + time + numeric check

```bash
python TRT_Baseline/build_and_time.py \
  --config TRT_Baseline/config_l4.yaml \
  --filter level4/7
```
Drives `trtexec` (TRT 10.16, in `.venv`) to build the FP32 engine and time it (`computeMs`,
kernel-only, matching the torch baseline contract — `build_and_time.py:99-112`), then runs
the Python numeric-parity check vs torch eager (`atol=rtol=5e-2`). Writes
`level4 → 7_gpt2_bs32_seq256.py` into `baseline_time_trt.json` with `mean/std/min/max/num_trials`
+ provenance + `status`.

**Known risk — int64 input binding (`numeric_check`, `build_and_time.py:176`).** The check
binds the torch input tensor's `data_ptr()` straight to the TRT input with no dtype reconcile.
If ONNX export / `onnxsim` narrows the token-id input to int32 while torch holds int64, the
binding is wrong → `numeric_mismatch`. This does **not** affect timing: `trtexec` generates
its own inputs, so `mean/std` stay valid; only the `status` flag / parity guarantee are affected.

## Fallbacks
- **Numeric mismatch but engine built:** timing entry is still produced (`numeric_mismatch`).
  For a clean `ok` timing without the parity gate, re-run stage 3 with `--skip-numeric`
  (`build_and_time.py:214`). Report the mismatch either way.
- **ONNX export fails:** capture `manifest.json`'s `error`/`traceback` (likely opset / HF
  tracing). Options: bump `onnx.opset_version` (18+), disable `onnx.simplify`, or fall back to
  the torch baseline for this problem. Do not silently skip.

## Generalizing to all of Level 4 (optional)
With `config_l4.yaml` in place, dropping `--filter` sweeps every file in `level4/` (20 files)
through both stages — same as the L1/L2 full sweep.

## Verification
1. `manifest.json` → `level4/7_gpt2_bs32_seq256.py` `status: ok`; `.onnx` present.
2. `baseline_time_trt.json` → `level4.7_gpt2_bs32_seq256.py` has positive `mean`,
   `num_trials == 200`.
3. Engine artifacts: `TRT_Baseline/engines/level4/7_gpt2_bs32_seq256.{plan,times.json}`.
4. Sanity replay: `trtexec --loadEngine=.../7_gpt2_bs32_seq256.plan --iterations=200`
   reproduces a similar mean.
5. Report `status` and mean ms.
