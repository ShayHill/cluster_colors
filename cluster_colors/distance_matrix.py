#!/usr/bin/env python3
# last modified: 221031 13:01:48
"""An elegant distance matrix?

:author: Shay Hill
:created: 2022-10-27
"""
import math
from contextlib import suppress
from typing import Callable, Generic, Hashable, TypeVar

_T = TypeVar("_T", bound=Hashable)


class DistanceMatrix(Generic[_T]):
    """A complete function matrix for a commutative function.

    Keeps matrix up to date so min and argmin will never miss a change.
    """

    def __init__(self, func: Callable[[_T, _T], float]):
        self.func = func
        self.cache: dict[tuple[_T, _T], float] = {}
        self._items: set[_T] = set()

    def __call__(self, a: _T, b: _T) -> float:
        with suppress(KeyError):
            return self.cache[a, b]
        with suppress(KeyError):
            return self.cache[b, a]
        raise KeyError(f"({a}, {b}) not in cache")

    def remove(self, item: _T):
        self._items.remove(item)
        for key in tuple(self.cache.keys()):
            if item in key:
                del self.cache[key]

    def add(self, item: _T):
        """Add a new item to the cache so min and argmin will see it."""
        for other in self._items:
            self.cache[item, other] = self.func(item, other)
        self._items.add(item)

    def min(self, a: _T) -> float:
        others = self._items - {a}
        return min(self(a, b) for b in others)

    def argmin(self, a: _T) -> _T:
        others = self._items - {a}
        return min(others, key=lambda b: self(a, b))

    def keymin(self) -> tuple[_T, _T]:
        keymin = min(self.cache, key=self.cache.__getitem__)
        return keymin

    def valmin(self) -> float:
        if not self.cache:
            return math.inf
        return self.cache[self.keymin()]
