"""Aggregate downstream scores without mixing model and split variability."""

from __future__ import annotations

import statistics
from collections.abc import Iterable


def summarize_seeded_results(results: Iterable[dict]) -> dict:
    """Summarize one task/architecture's evaluator JSONs.

    Each JSON is one pretrained checkpoint containing downstream split-seed
    scores. The central score gives each pretrained checkpoint equal weight
    after averaging its split scores. With multiple pretrained seeds, ``std``
    is the sample SD across those per-checkpoint means. With one checkpoint,
    ``std`` is the sample SD across that checkpoint's split seeds.
    """
    scores_by_pretraining_seed: dict[str, dict[str, float]] = {}
    for result in results:
        raw_seed = result.get("pretraining_seed")
        seed = "unspecified" if raw_seed is None else str(int(raw_seed))
        if seed in scores_by_pretraining_seed:
            raise ValueError(f"Duplicate downstream result for pretraining seed {seed}.")
        scores = {
            str(int(split_seed)): float(score)
            for split_seed, score in result.get("seeds", {}).items()
        }
        if not scores:
            raise ValueError(f"No downstream split scores for pretraining seed {seed}.")
        scores_by_pretraining_seed[seed] = scores

    if not scores_by_pretraining_seed:
        raise ValueError("No downstream results were supplied.")

    split_sets = {
        tuple(sorted(scores, key=int))
        for scores in scores_by_pretraining_seed.values()
    }
    if len(split_sets) != 1:
        raise ValueError(f"Pretraining checkpoints use different split seeds: {sorted(split_sets)}")

    seed_means = {
        seed: statistics.mean(scores.values())
        for seed, scores in scores_by_pretraining_seed.items()
    }
    split_stds = {
        seed: statistics.stdev(scores.values()) if len(scores) > 1 else None
        for seed, scores in scores_by_pretraining_seed.items()
    }
    pretraining_seed_std = statistics.stdev(seed_means.values()) if len(seed_means) > 1 else None
    available_split_stds = [value for value in split_stds.values() if value is not None]
    mean_split_seed_std = statistics.mean(available_split_stds) if available_split_stds else None

    if pretraining_seed_std is not None:
        reported_std = pretraining_seed_std
        std_basis = "pretraining_seed_means"
    elif mean_split_seed_std is not None:
        reported_std = mean_split_seed_std
        std_basis = "downstream_split_seeds_for_one_pretraining_seed"
    else:
        reported_std = None
        std_basis = "not_available"

    flat_scores = {
        f"pretrain_{seed}/split_{split_seed}": score
        for seed, split_scores in scores_by_pretraining_seed.items()
        for split_seed, score in split_scores.items()
    }
    return {
        "mean": statistics.mean(seed_means.values()),
        "std": reported_std,
        "std_basis": std_basis,
        "pretraining_seed_std": pretraining_seed_std,
        "mean_split_seed_std": mean_split_seed_std,
        "n_pretraining_seeds": len(seed_means),
        "n_split_seeds_per_pretraining_seed": len(next(iter(split_sets))),
        "pretraining_seeds": sorted(scores_by_pretraining_seed),
        "split_seeds": list(next(iter(split_sets))),
        "pretraining_seed_means": seed_means,
        "split_std_by_pretraining_seed": split_stds,
        "scores": flat_scores,
    }
