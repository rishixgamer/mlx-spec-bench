"""Cohort scheduling, greedy verification, and generation evidence."""

from dataclasses import dataclass, field
from random import Random
from time import perf_counter_ns

from .cache import CachePlan, LayerCache
from .config import Config
from .models import LoadedModel, ModelMetadata


@dataclass(frozen=True)
class Prompt:
    id: str
    input_ids: list[int]


@dataclass
class Trace:
    prompt_id: str
    input_ids: list[int]
    submit_ns: int
    output_ids: list[int] = field(default_factory=list)
    emissions: list[dict] = field(default_factory=list)
    cache_events: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    proposed: int = 0

    def emit(self, tokens: list[int], kind: str, when: int) -> None:
        if tokens:
            self.output_ids.extend(tokens)
            self.emissions.append({"at_ns": when, "kind": kind, "count": len(tokens)})


@dataclass
class CacheStats:
    committed_peak: int = 0
    working_peak: int = 0
    evictions: int = 0


class CohortMemory:
    def __init__(self, budget_bytes: int | None, stats: CacheStats):
        self.budget_bytes = budget_bytes
        self.stats = stats
        self.committed = 0
        self.working = 0

    def update(self, committed_delta: int, working_delta: int) -> None:
        self.committed += committed_delta
        self.working += working_delta
        self.stats.committed_peak = max(self.stats.committed_peak, self.committed)
        # Observe live array bytes and charge two copies under a bounded plan.
        charge = self.working * (2 if self.budget_bytes is not None else 1)
        self.stats.working_peak = max(self.stats.working_peak, charge)
        if self.budget_bytes is not None and charge > self.budget_bytes:
            raise AssertionError(f"KV charge {charge} exceeds configured bound {self.budget_bytes}")


class Session:
    def __init__(self, mx, loaded: LoadedModel, window: int | None, role: str,
                 trace: Trace, memory: CohortMemory):
        self.mx = mx
        self.model = loaded.model
        self.layers = [LayerCache(mx, window) for _ in self.model.layers]
        self.window = window
        self.role = role
        self.trace = trace
        self.memory = memory
        self.committed = 0
        self.committed_bytes = 0
        self.working_bytes = 0
        self.evictions = 0
        self.next = None

    @property
    def offset(self) -> int:
        return self.layers[0].offset

    def advance(self, tokens: list[int], *, all_predictions=False, logits=False):
        mx = self.mx
        raw = self.model(mx.array([tokens], dtype=mx.int32), cache=self.layers)[0]
        output = raw if all_predictions else raw[-1]
        if not logits:
            output = mx.argmax(output, axis=-1)
        arrays = [array for layer in self.layers for array in layer.arrays]
        mx.eval(output, *arrays)
        mx.synchronize()
        size = sum(array.nbytes for array in arrays)
        self.memory.update(0, size - self.working_bytes)
        self.working_bytes = size
        return output if logits else output.tolist()

    def commit(self, end: int) -> None:
        start_before = self.layers[0].start
        for layer in self.layers:
            layer.commit(end)
        arrays = [array for layer in self.layers for array in layer.arrays]
        if arrays:
            self.mx.eval(*arrays)
            self.mx.synchronize()
        start = self.layers[0].start
        evictions = start - start_before
        self.memory.stats.evictions += evictions
        self.evictions += evictions
        size = sum(array.nbytes for array in arrays)
        self.memory.update(size - self.committed_bytes, size - self.working_bytes)
        self.committed = end
        self.trace.cache_events.append({
            "role": self.role, "start": start, "end": end, "bytes": size,
            "evictions": self.evictions, "staged_bytes": self.working_bytes,
        })
        self.committed_bytes = self.working_bytes = size

    def append(self, tokens: list[int], *, logits=False, all_predictions=False):
        self.next = self.advance(tokens, logits=logits, all_predictions=all_predictions)
        self.commit(self.offset)
        return self.next

    def release(self) -> None:
        self.memory.update(-self.committed_bytes, -self.working_bytes)
        self.layers.clear()
        self.next = None


def _sample(mx, logits, temperature: float, rng: Random) -> int:
    cumulative = mx.cumsum(mx.softmax(logits.astype(mx.float32) / temperature))
    selected = mx.minimum(mx.sum(cumulative < rng.random()), logits.size - 1)
    mx.eval(selected)
    mx.synchronize()
    return selected.item()


def speculative_step(target, draft, maximum: int, eos: set[int], *, proposals: int):
    """Verify one draft block and leave both sessions aligned at its committed end."""
    start = target.committed
    assert start == draft.committed
    proposed = []
    for _ in range(min(proposals, maximum)):
        token = draft.next
        proposed.append(token)
        draft.next = draft.advance([token])
        if token in eos:
            break
    target_after = target.advance(proposed, all_predictions=True)
    predictions = [target.next, *target_after]
    accepted_count = 0
    for suggestion, expected in zip(proposed, predictions):
        if suggestion != expected:
            break
        accepted_count += 1

    accepted = proposed[:accepted_count]
    target.commit(start + accepted_count)
    draft.commit(start + accepted_count)
    if accepted_count == maximum or (accepted and accepted[-1] in eos):
        return accepted, None, None, len(proposed)
    next_token = predictions[accepted_count]
    corrected = next_token if accepted_count < len(proposed) else None
    bonus = next_token if corrected is None else None
    assert target.committed == draft.committed == start + accepted_count
    return accepted, corrected, bonus, len(proposed)


def encode_prompts(suite, target: ModelMetadata, maximum: int, draft=None) -> list[Prompt]:
    tokenizer = target.tokenizer
    limit = target.config["max_position_embeddings"]
    if draft:
        limit = min(limit, draft.config["max_position_embeddings"])
    prompts = []
    for entry in suite.data["prompts"]:
        ids = tokenizer.encode(suite.render(entry["text"]), add_special_tokens=False)
        observed = len(ids)
        if not observed or observed + maximum > limit:
            raise ValueError(
                f"prompt {entry['id']!r}: observed {observed} input tokens, context limit {limit}, "
                f"requested up to {maximum} new tokens; supply a nonempty prompt within the limit"
            )
        prompts.append(Prompt(entry["id"], ids))
    return prompts


def _mark_stop(trace: Trace, eos: set[int], maximum: int) -> None:
    if trace.output_ids[-1] in eos:
        trace.stop_reason = "eos"
    elif len(trace.output_ids) == maximum:
        trace.stop_reason = "max_tokens"


@dataclass
class Generation:
    traces: list[Trace]
    prefill_ns: int
    generation_ns: int
    cache: CacheStats


def generate(mx, target: LoadedModel, draft: LoadedModel | None,
             config: Config, plan: CachePlan, prompts: list[Prompt]) -> Generation:
    rng = Random(config.seed)
    eos = target.metadata.tokenizer.eos_token_ids
    prefill_ns = 0
    traces = []
    stats = CacheStats()
    origin = perf_counter_ns()

    for batch_start in range(0, len(prompts), config.batch_size):
        memory = CohortMemory(plan.budget_bytes, stats)
        submitted = perf_counter_ns() - origin
        active = []
        for prompt in prompts[batch_start:batch_start + config.batch_size]:
            trace = Trace(prompt.id, prompt.input_ids, submitted)
            t = Session(mx, target, plan.target_window, "target", trace, memory)
            d = Session(mx, draft, plan.draft_window, "draft", trace, memory) if draft else None
            started = perf_counter_ns()
            for index in range(0, len(prompt.input_ids), plan.prefill_chunk):
                chunk = prompt.input_ids[index:index + plan.prefill_chunk]
                t.append(chunk, logits=bool(config.temperature))
                if d:
                    d.append(chunk)
            prefill_ns += perf_counter_ns() - started
            first = _sample(mx, t.next, config.temperature, rng) if config.temperature else t.next
            trace.emit([first], "autoregressive", perf_counter_ns() - origin)
            _mark_stop(trace, eos, config.max_tokens)
            if trace.stop_reason is None:
                active.append((t, d, trace))
            else:
                t.release()
                if d:
                    d.release()
            traces.append(trace)
        while active:
            unfinished = []
            for t, d, trace in active:
                previous = trace.output_ids[-1]
                if d:
                    # The first token and each correction/bonus are emitted
                    # before their cache entries are needed by the next round.
                    if t.committed < len(trace.input_ids) + len(trace.output_ids):
                        t.append([previous])
                        d.append([previous])
                    accepted, corrected, bonus, proposed = speculative_step(
                        t, d, config.max_tokens - len(trace.output_ids), eos, proposals=config.k
                    )
                    trace.proposed += proposed
                    now = perf_counter_ns() - origin
                    trace.emit(accepted, "accepted", now)
                    trace.emit([] if corrected is None else [corrected], "corrections", now)
                    trace.emit([] if bonus is None else [bonus], "bonus", now)
                else:
                    next_token = t.append([previous], logits=bool(config.temperature))
                    if config.temperature:
                        next_token = _sample(mx, next_token, config.temperature, rng)
                    trace.emit([next_token], "autoregressive", perf_counter_ns() - origin)
                _mark_stop(trace, eos, config.max_tokens)
                if trace.stop_reason is None:
                    unfinished.append((t, d, trace))
                else:
                    t.release()
                    if d:
                        d.release()
            active = unfinished
    return Generation(traces, prefill_ns, perf_counter_ns() - origin, stats)
