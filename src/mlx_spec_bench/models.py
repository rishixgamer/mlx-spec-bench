"""Pinned Qwen2 model sources, offline fixtures, and verified MLX artifacts."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .config import ModelSpec
from .io import file_fingerprint, fingerprint, read_json, write_json


class PlatformError(RuntimeError):
    pass


class PreflightError(RuntimeError):
    pass


class RuntimeAllocationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    license: str


SOURCES = {
    "fixture": Source("local:fixture", "fixture-generator-v1", "MIT"),
    "qwen2.5-1.5b": Source(
        "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", "Apache-2.0"
    ),
    "qwen2.5-0.5b": Source(
        "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", "Apache-2.0"
    ),
}
FIXTURE_SEED = 20261120
FIXTURE_WORDS = {
    "[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "red": 3, "blue": 4, "green": 5,
    "gold": 6, "black": 7, "white": 8, "orange": 9, "purple": 10,
}
FIXTURE_CONFIG = {
    "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 128,
    "intermediate_size": 192, "num_attention_heads": 4, "num_key_value_heads": 2,
    "num_hidden_layers": 2, "max_position_embeddings": 128, "vocab_size": 64,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False,
    "eos_token_id": 2, "pad_token_id": 0,
}
BITS = {"q4": 4, "q6": 6, "q8": 8}
GROUP_SIZE = 64
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt")
METADATA_FILES = (*TOKENIZER_FILES, "config.json", "generation_config.json")
TOKENIZER_PROBES = ("", "Hello, world!", " a\n\tb ", "café 東京 🌱", "[EOS]", "<|endoftext|>")


def initialize(*, require_metal: bool = True):
    import platform

    if require_metal and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise PlatformError(
            f"Metal benchmarks require Apple Silicon macOS 15+ and arm64 Python; "
            f"found {platform.system()} {platform.machine()}. Run schema validation and reports on this host."
        )
    if require_metal and int(platform.mac_ver()[0].split(".")[0]) < 15:
        raise PlatformError("Metal benchmarks require macOS 15 or later; upgrade macOS and retry.")
    try:
        import mlx.core as mx
    except ImportError as error:
        raise PlatformError("MLX unavailable; install 'mlx-spec-bench[metal]' in the active arm64 environment.") from error
    if require_metal and not mx.metal.is_available():
        raise PlatformError("MLX could not initialize Metal; use native arm64 Python on an Apple Silicon Mac.")
    mx.set_default_device(mx.gpu if require_metal else mx.cpu)
    return mx


@contextmanager
def _locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def cache_root() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mlx-spec-bench"
    return Path(os.environ.get("MLX_SPEC_BENCH_CACHE", default)).expanduser().resolve()


def _fixture_metadata(root: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    path = root / SOURCES["fixture"].revision / "metadata"
    with _locked(path.with_suffix(".lock")):
        if not path.is_dir():
            with TemporaryDirectory(dir=path.parent) as temporary:
                staged = Path(temporary) / "metadata"
                staged.mkdir()
                tokenizer = Tokenizer(models.WordLevel(FIXTURE_WORDS, unk_token="[UNK]"))
                tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
                tokenizer.save(str(staged / "tokenizer.json"))
                write_json(staged / "tokenizer_config.json", {
                    "tokenizer_class": "TokenizersBackend",
                    "unk_token": "[UNK]", "pad_token": "[PAD]", "eos_token": "[EOS]",
                    "model_max_length": 128,
                })
                write_json(staged / "config.json", FIXTURE_CONFIG)
                os.replace(staged, path)
    return path


def _qwen_weight_shapes(config: dict):
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    kv_width = config["num_key_value_heads"] * (hidden // config["num_attention_heads"])
    yield "model.embed_tokens.weight", (config["vocab_size"], hidden)
    yield "model.norm.weight", (hidden,)
    if not config["tie_word_embeddings"]:
        yield "lm_head.weight", (config["vocab_size"], hidden)
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for projection, width in (("q", hidden), ("k", kv_width), ("v", kv_width), ("o", hidden)):
            yield prefix + f"self_attn.{projection}_proj.weight", (width, hidden)
            if projection != "o":
                yield prefix + f"self_attn.{projection}_proj.bias", (width,)
        for projection in ("gate", "up"):
            yield prefix + f"mlp.{projection}_proj.weight", (intermediate, hidden)
        yield prefix + "mlp.down_proj.weight", (hidden, intermediate)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield prefix + f"{norm}.weight", (hidden,)


def _fixture_weights(path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    weights = {}
    for name, shape in _qwen_weight_shapes(FIXTURE_CONFIG):
        if "norm.weight" in name:
            weights[name] = np.ones(shape, dtype=np.float16)
        else:
            weights[name] = (rng.integers(-8, 9, shape) / 4096).astype(np.float16)
    embedding = weights["model.embed_tokens.weight"]
    head = weights["lm_head.weight"]
    transitions = {4: 5, 5: 6, 6: 2, 7: 2, 8: 8}
    for token in range(FIXTURE_CONFIG["vocab_size"]):
        embedding[token, token] = 1
        head[transitions.get(token, 2), token] = 1
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "mlx"})


def _download(source: Source, *, weights: bool) -> Path:
    from huggingface_hub import snapshot_download

    patterns = list(METADATA_FILES)
    if weights:
        patterns += ["model*.safetensors", "LICENSE*", "README.md"]
    return Path(snapshot_download(repo_id=source.repo, revision=source.revision, allow_patterns=patterns))


def metadata(spec: ModelSpec, root: Path) -> "ModelMetadata":
    from mlx_lm.utils import load_tokenizer

    source = SOURCES[spec.id]
    try:
        path = _fixture_metadata(root) if spec.id == "fixture" else _download(source, weights=False)
        config = read_json(path / "config.json")
        if spec.id == "fixture":
            from mlx_lm.tokenizer_utils import TokenizerWrapper
            from transformers.tokenization_utils_tokenizers import TokenizersBackend

            tokenizer = TokenizerWrapper(TokenizersBackend(
                tokenizer_file=str(path / "tokenizer.json"), unk_token="[UNK]",
                pad_token="[PAD]", eos_token="[EOS]", model_max_length=128,
            ), eos_token_ids=[FIXTURE_WORDS["[EOS]"]])
        else:
            tokenizer = load_tokenizer(
                path, {"trust_remote_code": False}, eos_token_ids=config["eos_token_id"]
            )
    except Exception as error:
        raise ValueError(
            f"{spec.id}/{spec.weight_format}: model metadata or tokenizer unavailable at "
            f"{source.repo}@{source.revision}. Check network access or populate the pinned cache. {error}"
        ) from error
    return ModelMetadata(spec, source, path, config, tokenizer, _tokenizer_signature(tokenizer))


def _tokenizer_signature(tokenizer) -> dict:
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # The graph is necessary for behavior outside any finite set of probes.
    behavior = {name: value for name, value in backend.items() if name not in ("padding", "truncation")}
    return {
        "vocabulary": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map,
        "special_ids": sorted(tokenizer.all_special_ids),
        "eos_ids": sorted(tokenizer.eos_token_ids),
        "behavior": behavior,
        "probes": [tokenizer.encode(text, add_special_tokens=False) for text in TOKENIZER_PROBES],
    }


@dataclass(frozen=True)
class ModelMetadata:
    spec: ModelSpec
    source: Source
    path: Path
    config: dict
    tokenizer: object
    signature: dict

    @property
    def bytes_per_token(self) -> int:
        c = self.config
        return 4 * c["num_hidden_layers"] * c["num_key_value_heads"] * (
            c["hidden_size"] // c["num_attention_heads"]
        )

    def estimate_weights(self, weight_format: str | None = None) -> int:
        import math

        bits = BITS.get(weight_format or self.spec.weight_format)
        total = 0
        for _, shape in _qwen_weight_shapes(self.config):
            count = math.prod(shape)
            if bits and len(shape) == 2 and shape[-1] % GROUP_SIZE == 0:
                total += count * bits // 8 + count // GROUP_SIZE * 4
            else:
                total += count * 2
        return total


def check_tokenizers(target: ModelMetadata, draft: ModelMetadata) -> None:
    for key in target.signature:
        if target.signature[key] != draft.signature[key]:
            reason = {
                "vocabulary": "vocabulary-to-ID mapping differs",
                "behavior": "tokenization graph differs",
                "probes": "encoded probe tokens differ",
            }.get(key, "special tokens or IDs differ")
            raise ValueError(
                f"tokenizer mismatch: {target.spec.id} ({fingerprint(target.signature)}) versus "
                f"{draft.spec.id} ({fingerprint(draft.signature)}); {reason}"
            )


def _artifact_files(path: Path) -> dict[str, str]:
    return {
        item.name: file_fingerprint(item) for item in sorted(path.iterdir())
        if item.is_file() and item.name != "verification.json"
    }


def _recipe(spec: ModelSpec) -> str:
    cast = "float16; mlx-lm " + version("mlx-lm")
    bits = BITS.get(spec.weight_format)
    return f"{cast}; affine group={GROUP_SIZE}, bits={bits}" if bits else cast


def _convert(mx, meta: ModelMetadata, path: Path, temporary: Path) -> None:
    from mlx.utils import tree_map
    from mlx_lm.utils import load_model, quantize_model, save_config, save_model

    if meta.spec.id == "fixture":
        original = temporary / "original"
        shutil.copytree(meta.path, original)
        _fixture_weights(original)
    else:
        original = _download(meta.source, weights=True)
    path.mkdir()
    model, config = load_model(original, lazy=True)
    model.update(tree_map(lambda array: array.astype(mx.float16), model.parameters()))
    if bits := BITS.get(meta.spec.weight_format):
        model, config = quantize_model(model, config, GROUP_SIZE, bits, mode="affine")
    save_model(path, model, donate_model=True)
    save_config(config, path / "config.json")
    for name in (*TOKENIZER_FILES, "generation_config.json", "LICENSE"):
        if (original / name).is_file():
            shutil.copyfile(original / name, path / name)
    mx.synchronize()
    mx.clear_cache()


def artifact_location(meta: ModelMetadata, root: Path) -> tuple[Path, dict]:
    identity = {"source": meta.source.repo, "revision": meta.source.revision, "recipe": _recipe(meta.spec)}
    return root / meta.source.revision / fingerprint(identity)[:20], identity


def resolve_artifact(mx, meta: ModelMetadata, root: Path) -> tuple[Path, str]:
    path, identity = artifact_location(meta, root)
    with _locked(path.with_suffix(".lock")):
        try:
            if path.is_dir():
                manifest = read_json(path / "verification.json")
                if manifest != {"identity": identity, "files": _artifact_files(path)}:
                    raise ValueError(f"cached file hashes disagree; remove {path} and regenerate")
            else:
                with TemporaryDirectory(dir=path.parent) as temporary:
                    staged = Path(temporary) / "artifact"
                    _convert(mx, meta, staged, Path(temporary))
                    manifest = {"identity": identity, "files": _artifact_files(staged)}
                    write_json(staged / "verification.json", manifest)
                    os.replace(staged, path)
        except (MemoryError, RuntimeAllocationError):
            raise
        except Exception as error:
            if is_allocation_failure(error):
                raise RuntimeAllocationError(str(error)) from error
            raise ValueError(
                f"{meta.spec.id}/{meta.spec.weight_format}: cannot use or convert the pinned weights. "
                f"Check the source cache and free disk space, then retry. {error}"
            ) from error
    return path, fingerprint(manifest)


def is_allocation_failure(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(term in message for term in (
        "out of memory", "failed to allocate", "cannot allocate memory", "bad_alloc", "metal error: oom",
        "exceeds the maximum allowed buffer size",
    ))


@dataclass
class LoadedModel:
    model: object
    metadata: ModelMetadata
    identity: dict


def load_artifact(mx, meta: ModelMetadata, path: Path, sha: str) -> LoadedModel:
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    model, config = load_model(path)
    requested = meta.spec.weight_format
    bits = BITS.get(requested)
    quantized = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    quant_config = config.get("quantization")
    expected_modules = {
        name.removesuffix(".weight") for name, shape in _qwen_weight_shapes(meta.config)
        if bits and len(shape) == 2 and shape[-1] % GROUP_SIZE == 0
    }
    matches = set(quantized) == expected_modules
    if bits:
        matches = matches and quant_config == {"bits": bits, "group_size": GROUP_SIZE, "mode": "affine"}
        matches = matches and all(
            (module.bits, module.group_size, module.mode, module.weight.dtype) ==
            (bits, GROUP_SIZE, "affine", mx.uint32) for module in quantized.values()
        )
    else:
        matches = matches and quant_config is None
    params = tree_flatten(model.parameters())
    packed_names = {name + ".weight" for name in quantized}
    fp16 = []
    for name, array in params:
        if name not in packed_names:
            matches = matches and array.dtype == mx.float16
            if not any(name.endswith(suffix) for suffix in (".scales", ".biases")):
                fp16.append(name)
    size = sum(array.nbytes for _, array in params)
    matches = matches and size == meta.estimate_weights()
    if not matches:
        raise ValueError(
            f"{meta.spec.id}/{requested}: actual tensor dtypes, packed modules, or size disagree "
            f"with the requested format; remove {path} and regenerate"
        )
    mx.synchronize()
    return LoadedModel(model, meta, {
        "id": meta.spec.id, "source": meta.source.repo, "revision": meta.source.revision,
        "license": meta.source.license, "artifact_sha256": sha,
        "tokenizer_sha256": fingerprint(meta.signature), "weight_format": requested,
        "weight_bytes": size, "recipe": _recipe(meta.spec),
        "group_size": GROUP_SIZE if bits else None,
        "quantized_modules": len(quantized),
        "retained_fp16_parameters": sorted(fp16) if bits else [],
    })
