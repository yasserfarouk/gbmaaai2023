from math import sqrt
from typing import Iterable


def dist(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    if x is None or y is None:
        return float("nan")
    return sqrt(sum((a - b) * (a - b) for a, b in zip(x, y)))


def nash_dist(w: tuple[float, ...], n: tuple[float, ...]) -> float:
    return dist(w, n)


def pareto_dist(w: tuple[float, ...], p: Iterable[tuple[float, ...]]) -> float:
    return min(dist(w, x) for x in p)
