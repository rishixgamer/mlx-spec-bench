"""Matched corpus perplexity and exact-match task scoring."""

from dataclasses import dataclass
import math
import unicodedata

from .decode import CacheStats, CohortMemory, Prompt, Session, Trace, encode_prompts, generate
from .io import fingerprint

RULES = (
    "quality-v1: score tokens fully inside the continuation with a predecessor; "
    "ignore padding and boundary-crossing tokens; use corpus exp(total_nll / tokens); "
    "greedy target-only tasks, decode without specials and compare NFKC casefolded "
    "whitespace-collapsed text by exact match"
)


@dataclass(frozen=True)
class Passage:
    id: str
    input_ids: list[int]
    scored: list[bool]


def prepare_quality(evaluation, target, config) -> tuple[list[Passage], list[Prompt]]:
    passages = []
    limit = target.config["max_position_embeddings"]
    backend = target.tokenizer.backend_tokenizer
    pad_id = target.tokenizer.pad_token_id
    for item in evaluation.data["text"]:
        prefix = evaluation.render(item["prefix"])
        encoded = backend.encode(prefix + item["continuation"], add_special_tokens=False)
        if len(encoded.ids) > limit:
            raise ValueError(
                f"evaluation passage {item['id']!r}: observed {len(encoded.ids)} tokens, context limit {limit}"
            )
        mask = [
            index > 0 and token != pad_id and start >= len(prefix) and end > start
            for index, (token, (start, end)) in enumerate(zip(encoded.ids, encoded.offsets))
        ]
        passages.append(Passage(item["id"], encoded.ids, mask))
    # The prompt encoder only uses text/ID and the format template.
    from .suites import Suite

    task_suite = Suite({
        "template": evaluation.data["template"], "prompts": evaluation.data["tasks"],
    }, evaluation.sha256)
    return passages, encode_prompts(task_suite, target, config.max_tokens)


def evaluation_identity(evaluation, max_tokens: int) -> dict:
    return {
        "id": evaluation.data["id"],
        "sha256": fingerprint({"data": evaluation.sha256, "rules": RULES, "max_tokens": max_tokens}),
    }


def masked_nll(mx, logits, labels: list[int], mask: list[bool]) -> tuple[float, int]:
    indices = [index for index, scored in enumerate(mask) if scored]
    if not indices:
        return 0.0, 0
    rows = logits[mx.array(indices)].astype(mx.float32)
    expected = mx.array([labels[index] for index in indices])
    chosen = mx.take_along_axis(rows, expected[:, None], axis=-1).squeeze(-1)
    losses = mx.maximum(mx.logsumexp(rows, axis=-1) - chosen, 0)
    total = mx.sum(losses)
    mx.eval(total)
    mx.synchronize()
    return total.item(), len(indices)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def task_matches(actual: str, expected: str) -> bool:
    return normalize(actual) == normalize(expected)


def corpus_perplexity(total_nll: float, count: int) -> float | None:
    return math.exp(total_nll / count) if count else None


def score(mx, target, config, plan, evaluation, passages, tasks) -> tuple[dict, dict]:
    from dataclasses import replace

    nll = []
    count = 0
    details = []
    for passage in passages:
        session = Session(
            mx, target, plan.target_window, "target", Trace(passage.id, passage.input_ids, 0),
            CohortMemory(plan.budget_bytes, CacheStats()),
        )
        passage_nll = []
        passage_count = 0
        for start in range(0, len(passage.input_ids) - 1, plan.prefill_chunk):
            end = min(start + plan.prefill_chunk, len(passage.input_ids) - 1)
            logits = session.append(passage.input_ids[start:end], logits=True, all_predictions=True)
            loss, tokens = masked_nll(mx, logits, passage.input_ids[start + 1:end + 1], passage.scored[start + 1:end + 1])
            passage_nll.append(loss)
            passage_count += tokens
        session.release()
        total = math.fsum(passage_nll)
        nll.append(total)
        count += passage_count
        details.append({
            "id": passage.id, "input_ids": passage.input_ids, "scored_mask": passage.scored,
            "total_nll": total, "scored_tokens": passage_count,
        })
    greedy = replace(config, draft=None, k=0, temperature=0)
    generation = generate(mx, target, None, greedy, plan, tasks)
    task_details = []
    correct = 0
    for trace, entry in zip(generation.traces, evaluation.data["tasks"], strict=True):
        actual = target.metadata.tokenizer.decode(trace.output_ids, skip_special_tokens=True)
        matched = task_matches(actual, entry["answer"])
        correct += matched
        task_details.append({
            "id": trace.prompt_id, "input_ids": trace.input_ids, "output_ids": trace.output_ids,
            "stop_reason": trace.stop_reason, "actual": actual, "expected": entry["answer"],
            "correct": matched,
        })
    return {
        "total_nll": math.fsum(nll), "scored_tokens": count,
        "task_correct": correct, "task_examples": len(tasks), "rules": RULES,
    }, {"text": details, "tasks": task_details}
