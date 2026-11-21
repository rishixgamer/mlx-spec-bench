"""Transactional sliding-window KV caches for the supported Qwen2 architecture."""

from dataclasses import dataclass

from .config import Config
from .models import ModelMetadata, PreflightError

PREFILL_CHUNK = 32


@dataclass(frozen=True)
class CachePlan:
    target_window: int | None
    draft_window: int | None
    prefill_chunk: int
    budget_bytes: int | None


def plan_cache(config: Config, target: ModelMetadata, draft: ModelMetadata | None) -> CachePlan:
    if config.cache.policy == "unbounded":
        return CachePlan(None, None, PREFILL_CHUNK, None)
    budget = config.cache.budget_bytes
    reserve = min(64, config.max_tokens)
    # Equal target/draft slots keep the target window invariant across K and
    # the presence of a draft. Two copies cover append and compaction overlap.
    slots = 2 * config.batch_size
    width = max(target.bytes_per_token, draft.bytes_per_token if draft else 0)
    minimum = slots * 2 * width * (reserve + 1)
    if budget < minimum:
        raise PreflightError(
            f"cache.budget_bytes: requested {budget} bytes, minimum {minimum} bytes "
            f"for batch_size={config.batch_size}, max_tokens={config.max_tokens}; "
            "increase the budget or reduce batch_size/max_tokens"
        )
    per_slot = budget // slots // 2
    return CachePlan(
        per_slot // target.bytes_per_token - reserve,
        per_slot // draft.bytes_per_token - reserve if draft else None,
        min(PREFILL_CHUNK, reserve),
        budget,
    )


class LayerCache:
    """Keep staged tokens until the decoder chooses a committed prefix."""

    def __init__(self, mx, window: int | None):
        self.mx = mx
        self.window = window
        self.offset = 0
        self.start = 0
        self.keys = self.values = None

    def make_mask(self, length: int, *, return_array=False, window_size=None):
        window = self.window if window_size is None else window_size
        if window is None and not return_array:
            return None if length == 1 else "causal"
        rows = self.mx.arange(self.offset, self.offset + length)[:, None]
        columns = self.mx.arange(self.start, self.offset + length)[None, :]
        allowed = rows >= columns
        return allowed if window is None else allowed & (rows - columns < window)

    def update_and_fetch(self, keys, values):
        length = keys.shape[-2]
        if self.keys is not None:
            keys = self.mx.concatenate((self.keys, keys), axis=-2)
            values = self.mx.concatenate((self.values, values), axis=-2)
        self.keys, self.values = keys, values
        self.offset += length
        return keys, values

    @property
    def arrays(self):
        return () if self.keys is None else (self.keys, self.values)

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays)

    def commit(self, end: int) -> None:
        start = self.start if self.window is None else max(self.start, end - self.window)
        if not self.start <= start <= end <= self.offset:
            raise AssertionError(f"invalid cache commit [{start}, {end}) from [{self.start}, {self.offset})")
        if start == end:
            self.keys = self.values = None
        elif (start, end) != (self.start, self.offset):
            section = slice(start - self.start, end - self.start)
            self.keys = self.mx.contiguous(self.keys[..., section, :])
            self.values = self.mx.contiguous(self.values[..., section, :])
        self.start, self.offset = start, end
