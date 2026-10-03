"""Population initialization, selection, crossover, and mutation."""

from __future__ import annotations

import numpy as np

from .grammar import Node, all_paths, has_nested_partial, random_tree, replace_at_path


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
        if child.height <= max_depth and not has_nested_partial(child):
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
        if child.height <= max_depth and not has_nested_partial(child):
            return child
    return node


def _node_at(node: Node, path: tuple[int, ...]) -> Node:
    current = node
    for index in path:
        current = current.children[index]
    return current
