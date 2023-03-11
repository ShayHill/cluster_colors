"""A distance matrix.

:author: Shay Hill
:created: 2022-10-27
"""
import math
from collections.abc import Callable, Hashable
from contextlib import suppress
from typing import Generic, TypeVar

_T = TypeVar("_T", bound=Hashable)


class DistanceMatrix(Generic[_T]):
    """A complete function matrix for a commutative function.

    Keeps matrix up to date so min and argmin will never miss a change.
    """

    def __init__(self, func: Callable[[_T, _T], float]) -> None:
        """Initialize with a function.

        :param func: a commutative function
        """
        self.func = func
        self.cache: dict[tuple[_T, _T], float] = {}
        self._items: set[_T] = set()

    def __call__(self, a: _T, b: _T) -> float:
        """Return the cached value or compute it.

        :param a: hashable argument to a cummutative function
        :param b: hashable argument to a cummutative function
        :return: self.func(a, b)
        :raise KeyError: if a or b are not in the cache
        """
        with suppress(KeyError):
            return self.cache[a, b]
        with suppress(KeyError):
            return self.cache[b, a]
        msg = f"({a}, {b}) not in cache"
        raise KeyError(msg)

    def remove(self, item: _T):
        """Remove an item from the cache so min and argmin will not see it.

        :param item: item to remove
        """
        self._items.remove(item)
        for key in tuple(self.cache.keys()):
            if item in key:
                del self.cache[key]

    def add(self, item: _T):
        """Add a new item to the cache so min and argmin will see it.

        :param item: item to add
        """
        for other in self._items:
            self.cache[item, other] = self.func(item, other)
        self._items.add(item)

    def min(self, a: _T) -> float:
        """Return the minimum value of the function for a given item.

        :param a: item to find the minimum for
        :return: minimum value of the function for a
        """
        others = self._items - {a}
        return min(self(a, b) for b in others)

    def argmin(self, a: _T) -> _T:
        """Return the item that minimizes the function for a given item.

        :param a: item to find the minimum for
        :return: item that minimizes the function for a
        """
        others = self._items - {a}
        return min(others, key=lambda b: self(a, b))

    def keymin(self) -> tuple[_T, _T]:
        """Return the pair of items that minimizes the function.

        :return: pair of items that minimizes the function, in arbitrary order
        """
        return min(self.cache, key=self.cache.__getitem__)

    def valmin(self) -> float:
        """Return the minimum value of the function.

        :return: minimum value of the function
        """
        if not self.cache:
            return math.inf
        return self.cache[self.keymin()]
