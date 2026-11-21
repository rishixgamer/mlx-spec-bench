"""Work performed inside a single isolated trial process."""

from dataclasses import asdict
from time import perf_counter_ns

import psutil

from .cache import plan_cache
from .config import Config
from .decode import encode_prompts, generate
from .memory import environment
from .metrics import generation_metrics
from .models import (
    PlatformError, PreflightError, artifact_location, cache_root, check_tokenizers, initialize,
    is_allocation_failure, load_artifact, metadata, resolve_artifact,
)
from .quality import evaluation_identity, prepare_quality, score
from .suites import load_evaluation, load_suite

RUNTIME_RESERVE_BYTES = 256 * 1024**2


def memory_allowance(config: Config) -> int:
    physical = psutil.virtual_memory().total
    return min(config.memory_limit_bytes or physical * 85 // 100, physical)


def preflight(config, target, draft, plan, longest: int, root) -> dict:
    models = [target, draft] if draft else [target]
    weights = sum(model.estimate_weights() for model in models)
    conversion = max((
        3 * model.estimate_weights("fp16") + model.estimate_weights()
        for model in models if not artifact_location(model, root)[0].is_dir()
    ), default=0)
    block = max(plan.prefill_chunk, min(config.k, config.max_tokens))
    width = sum(model.bytes_per_token for model in models)
    kv = plan.budget_bytes or 2 * (longest + block) * width * config.batch_size
    vocab = max(model.config["vocab_size"] for model in models)
    heads = max(model.config["num_attention_heads"] for model in models)
    working = 4 * block * (4 * vocab + heads * (longest + block))
    estimate = RUNTIME_RESERVE_BYTES + max(conversion, weights + kv + working)
    allowance = memory_allowance(config)
    if estimate > allowance:
        raise PreflightError(
            f"memory preflight estimates {estimate} bytes against a {allowance}-byte allowance "
            f"(memory_limit_bytes={config.memory_limit_bytes}, target={config.target.id}/"
            f"{config.target.weight_format}, draft={config.draft}, batch_size={config.batch_size}, "
            f"k={config.k}, cache={config.cache}, max_tokens={config.max_tokens}); "
            "use a smaller model, reduce batch_size/k/cache budget, or increase the memory allowance"
        )
    return {
        "estimate_bytes": estimate, "allowance_bytes": allowance, "weight_bytes": weights,
        "cold_conversion_bytes": conversion, "kv_bytes": kv, "verification_workspace_bytes": working,
        "runtime_reserve_bytes": RUNTIME_RESERVE_BYTES,
    }


def execute(config: Config, warmups: int, *, require_metal: bool = True) -> dict:
    identities = dict.fromkeys(("target", "draft", "suite", "evaluation"))
    result = {
        "status": "failure", "error": None, "load_ns": None, "warmup_ns": None, "quality_ns": None,
        "metrics": None, "identities": identities, "environment": environment(), "data": None,
    }
    try:
        suite = load_suite(config.suite)
        identities["suite"] = suite.identity
        evaluation = load_evaluation() if config.quality else None
        if evaluation:
            identities["evaluation"] = evaluation_identity(evaluation, config.max_tokens)

        mx = initialize(require_metal=require_metal)
        result["environment"] = environment(mx)
        start = perf_counter_ns()
        root = cache_root()
        target_meta = metadata(config.target, root)
        draft_meta = metadata(config.draft, root) if config.draft else None
        if draft_meta:
            check_tokenizers(target_meta, draft_meta)
        prompts = encode_prompts(suite, target_meta, config.max_tokens, draft_meta)
        passages, tasks = prepare_quality(evaluation, target_meta, config) if evaluation else ([], [])
        longest = max(
            [len(prompt.input_ids) + config.max_tokens for prompt in [*prompts, *tasks]]
            + [len(passage.input_ids) for passage in passages]
        )
        plan = plan_cache(config, target_meta, draft_meta)
        estimate = preflight(config, target_meta, draft_meta, plan, longest, root)
        target_path, target_sha = resolve_artifact(mx, target_meta, root)
        if draft_meta:
            draft_path, draft_sha = resolve_artifact(mx, draft_meta, root)
        target = load_artifact(mx, target_meta, target_path, target_sha)
        identities["target"] = target.identity
        draft = None
        if draft_meta:
            draft = load_artifact(mx, draft_meta, draft_path, draft_sha)
            identities["draft"] = draft.identity
        result["load_ns"] = perf_counter_ns() - start

        start = perf_counter_ns()
        for _ in range(warmups):
            generate(mx, target, draft, config, plan, prompts)
        mx.synchronize()
        mx.clear_cache()
        result["warmup_ns"] = perf_counter_ns() - start if warmups else 0

        mx.reset_peak_memory()
        resident = mx.get_active_memory()
        generated = generate(mx, target, draft, config, plan, prompts)
        mx.synchronize()
        peak = max(resident, mx.get_peak_memory())

        start = perf_counter_ns()
        quality, quality_data = score(mx, target, config, plan, evaluation, passages, tasks) if evaluation else (None, None)
        result["quality_ns"] = perf_counter_ns() - start if evaluation else 0
        result["metrics"] = generation_metrics(
            generated, resident=resident, peak=peak, quality=quality, window=plan.target_window
        )
        result["data"] = {
            "schema_version": 1, "suite": identities["suite"], "evaluation": identities["evaluation"],
            "cache_plan": asdict(plan), "preflight": estimate,
            "requests": [asdict(trace) for trace in generated.traces], "quality": quality_data,
        }
        result["status"] = "success"
    except Exception as error:
        if isinstance(error, PlatformError):
            kind = "platform"
        elif isinstance(error, PreflightError):
            kind = "preflight"
        elif is_allocation_failure(error):
            kind = "allocation_failure"
        elif isinstance(error, (ValueError, OSError)):
            kind = "config"
        else:
            kind = "runtime"
        result["error"] = {"kind": kind, "message": str(error) or type(error).__name__}
    return result
