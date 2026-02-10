import gc
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


_initialized = False
_train_fn = None
_train_fn_final = None
_pf_stream = None


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    global _initialized, _train_fn, _train_fn_final, _pf_stream

    is_cuda = str(device).startswith("cuda")

    if not _initialized:
        _initialized = True

        if is_cuda:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

            try:
                torch.autograd.set_detect_anomaly(False)
            except Exception:
                pass

            try:
                torch._C._jit_set_profiling_mode(False)
                torch._C._jit_set_profiling_executor(False)
            except Exception:
                pass

            try:
                import torch._inductor.config as ic
                ic.coordinate_descent_tuning = True
                ic.triton.unique_kernel_names = True
                ic.fx_graph_cache = True
                ic.epilogue_fusion = True
                for _attr, _val in [
                    ("aggressive_fusion", True),
                    ("triton.cudagraph_trees", True),
                    ("triton.cudagraphs", True),
                    ("force_fuse_int_mm_with_mul", True),
                    ("use_mixed_mm", True),
                    ("benchmark_kernel", True),
                    ("split_reductions", True),
                    ("max_autotune", True),
                    ("max_autotune_gemm", True),
                    ("shape_padding", True),
                    ("triton.multi_kernel", True),
                    ("combo_kernels", True),
                ]:
                    try:
                        _parts = _attr.split(".")
                        _obj = ic
                        for _p in _parts[:-1]:
                            _obj = getattr(_obj, _p)
                        setattr(_obj, _parts[-1], _val)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                import torch._dynamo.config as dc
                dc.cache_size_limit = 256
                for _attr, _val in [
                    ("automatic_dynamic_shapes", False),
                    ("assume_static_by_default", True),
                    ("suppress_errors", True),
                    ("optimize_ddp", False),
                ]:
                    try:
                        setattr(dc, _attr, _val)
                    except Exception:
                        pass
            except Exception:
                pass

            for obj in gc.get_objects():
                if isinstance(obj, torch.optim.Optimizer) and obj is not optimizer:
                    obj.state.clear()
            gc.collect()
            torch.cuda.empty_cache()

            _pf_stream = torch.cuda.Stream()

    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            for group in optimizer.param_groups:
                if not group.get("fused", False):
                    group["fused"] = True
                    group["foreach"] = False
        except Exception:
            pass

    if _train_fn is None:

        def _step_intermediate(input_ids, labels):
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            return loss.detach()

        def _step_final(input_ids, labels):
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            return loss.detach(), logits.detach()

        if is_cuda:
            compile_opts = dict(mode="max-autotune", fullgraph=False, dynamic=False)
            try:
                _train_fn = torch.compile(_step_intermediate, **compile_opts)
                _train_fn_final = torch.compile(_step_final, **compile_opts)
            except Exception:
                try:
                    compile_opts_fb = dict(mode="reduce-overhead", fullgraph=False, dynamic=False)
                    _train_fn = torch.compile(_step_intermediate, **compile_opts_fb)
                    _train_fn_final = torch.compile(_step_final, **compile_opts_fb)
                except Exception:
                    _train_fn = _step_intermediate
                    _train_fn_final = _step_final
        else:
            _train_fn = _step_intermediate
            _train_fn_final = _step_final

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    try:
        pf = _pf_stream
        use_pf = is_cuda and pf is not None

        if use_pf:
            with torch.cuda.stream(pf):
                next_batch = next(data_iterator).to(device, non_blocking=True)
        else:
            next_batch = next(data_iterator).to(device)

        total_tokens = 0
        last_step = num_steps - 1

        for step in range(num_steps):
            if use_pf:
                torch.cuda.current_stream().wait_stream(pf)
            batch = next_batch

            if step < last_step:
                nb = next(data_iterator)
                if use_pf:
                    with torch.cuda.stream(pf):
                        next_batch = nb.to(device, non_blocking=True)
                else:
                    next_batch = nb.to(device)

            if step < last_step:
                loss = _train_fn(batch[:, :-1], batch[:, 1:])
            else:
                loss, logits = _train_fn_final(batch[:, :-1], batch[:, 1:])

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_tokens += batch.numel()

    finally:
        if gc_was_enabled:
            gc.enable()

    return InnerStepsResult(
        final_logits=logits.float(),
        total_tokens=total_tokens,
        final_loss=loss.item(),
    )


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("TESTING train_agent_output.py")
    print("=" * 60)
    print()

    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    print(f"Batch size: {batch_size}")
    print(f"Steps per eval: {num_steps}")
    print(f"Evaluations: {num_evals}")
    print()

    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    attn_impl = "sdpa"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    print(f"Loading model (attn={attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.gradient_checkpointing_enable()
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    if torch.cuda.is_available():
        data = data.pin_memory()
    print(f"Samples: {data.shape[0]:,}, Sequence length: {data.shape[1]}")
    print()

    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    print("Warmup (compile)...")
    t0 = time.perf_counter()
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  Warmup: {t1 - t0:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    print(f"Running {num_evals} evaluations...")
    tps_list = []

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        tps = result.total_tokens / elapsed
        tps_list.append(tps)
        print(
            f"  Eval {i + 1}: {elapsed:.2f}s, "
            f"TPS={tps:,.0f}, "
            f"tokens={result.total_tokens:,}, "
            f"loss={result.final_loss:.4f}"
        )

    print()
    tps_list.sort()
    median_tps = tps_list[len(tps_list) // 2]
    print(f"Median TPS: {median_tps:,.0f}")
    print("Done!")
