"""Population initialization, selection, crossover, and mutation."""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from .grammar import (Node, all_paths, evaluate_expression, has_nested_partial,
                      random_tree, replace_at_path, validate_expression)


def create_random_expression(rng, primitive_features, max_depth=5, temporal_windows=(2, 4, 8, 16)):
    return random_tree(rng, tuple(primitive_features), max_depth, tuple(temporal_windows))


def create_initial_population(rng, primitive_features, population_size=32, max_depth=5, temporal_windows=(2, 4, 8, 16)):
    return initialize_population(rng, tuple(primitive_features), population_size, max_depth, tuple(temporal_windows))


def deduplicate_population(population):
    return list({str(node): node for node in population}.values())


def filter_invalid_features(population, primitive_features=None, max_depth=10):
    return [node for node in population if validate_expression(node, primitive_features, max_depth)]


def filter_constant_features(population, frame, min_unique=2, sample_mask=None):
    kept = []
    for node in population:
        try:
            values = evaluate_expression(node, frame)
            if sample_mask is not None:
                mask = np.asarray(sample_mask, dtype=bool)
                if len(mask) != len(frame):
                    raise ValueError("sample_mask must match the frame rows")
                values = values.iloc[np.flatnonzero(mask)]
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if values.nunique() >= min_unique:
                kept.append(node)
        except (ValueError, KeyError, TypeError, FloatingPointError, AttributeError):
            continue
    return kept


def filter_redundant_features(population, frame, threshold=0.98, min_samples=30, sample_mask=None):
    """Keep the first representative from each highly correlated expression group."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    kept, values = [], []
    for node in tqdm(population, desc="Filtering Redundant Expressions", leave=False):
        try:
            series = evaluate_expression(node, frame).to_numpy(dtype=float)
        except (ValueError, KeyError, TypeError, FloatingPointError, AttributeError):
            continue
        mask = np.ones(len(series), dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool)
        if len(mask) != len(series):
            raise ValueError("sample_mask must match the frame rows")
        series = series[mask]
        duplicate = False
        for prior in values:
            valid = np.isfinite(series) & np.isfinite(prior)
            if valid.sum() >= min_samples and np.std(series[valid]) > 1e-12 and np.std(prior[valid]) > 1e-12:
                corr = np.corrcoef(series[valid], prior[valid])[0, 1]
                if np.isfinite(corr) and abs(corr) >= threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(node)
            values.append(series)
    return kept


def initialize_population(
    rng: np.random.Generator,
    variables: tuple[str, ...],
    population_size: int,
    max_depth: int,
    temporal_windows: tuple[int, ...],
) -> list[Node]:
    """Create a mixed grow/full population with varied initial complexity."""
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    trees = []
    for index in range(population_size):
        depth = int(rng.integers(2, max_depth + 1)) if max_depth > 2 else max_depth
        trees.append(random_tree(
            rng,
            variables,
            depth,
            temporal_windows=temporal_windows,
            grow=(index % 2 == 0),
        ))
    return trees


def tournament_select(
    rng: np.random.Generator,
    scored_population: list[tuple[Node, float]],
    tournament_size: int,
) -> Node:
    """Select the highest-fitness individual in a random tournament."""
    if not scored_population:
        raise ValueError("cannot select from an empty population")
    size = min(max(1, tournament_size), len(scored_population))
    indices = rng.choice(len(scored_population), size=size, replace=False)
    winner = max((scored_population[int(index)] for index in indices), key=lambda item: item[1])
    return winner[0]


def crossover(
    rng: np.random.Generator,
    left: Node,
    right: Node,
    max_depth: int,
) -> Node:
    """Replace one random subtree, retrying until the height bound holds."""
    left_paths = all_paths(left)
    right_paths = all_paths(right)
    for _ in range(12):
        left_path = left_paths[int(rng.integers(len(left_paths)))]
        right_path = right_paths[int(rng.integers(len(right_paths)))]
        child = replace_at_path(left, left_path, _node_at(right, right_path))
        if child.height <= max_depth and not has_nested_partial(child) and validate_expression(child, max_depth=max_depth):
            return child
    return left


def mutate(
    rng: np.random.Generator,
    node: Node,
    variables: tuple[str, ...],
    max_depth: int,
    temporal_windows: tuple[int, ...],
) -> Node:
    """Replace one subtree with a random bounded subtree."""
    paths = list(all_paths(node))
    rng.shuffle(paths)
    for path in paths:
        remaining_depth = max_depth - len(path)
        if remaining_depth < 1:
            continue
        current = node
        inside_partial = False
        for index in path:
            if current.op == "partial":
                inside_partial = True
            current = current.children[index]
        replacement = random_tree(
            rng,
            variables,
            remaining_depth,
            temporal_windows=temporal_windows,
            grow=True,
            allow_partial=not inside_partial,
        )
        child = replace_at_path(node, path, replacement)
        if child.height <= max_depth and not has_nested_partial(child) and validate_expression(child, variables, max_depth=max_depth):
            return child
    return node


def _node_at(node: Node, path: tuple[int, ...]) -> Node:
    current = node
    for index in path:
        current = current.children[index]
    return current
