"""Exceptions where clusters can no longer be split or merged.

:author: Shay Hill
:created: 2024-09-16
"""


class FailedToSplitError(Exception):
    """Exception raised when no clusters can be split."""

    def __init__(self, message: str | None = None) -> None:
        """Create a new AllClustersAreSingletonsError instance."""
        message_ = message or "Cannot split any cluster. All clusters are singletons."
        self.message = message_
        super().__init__(self.message)


class FailedToMergeError(Exception):
    """Exception raised when no clusters can be merged."""

    def __init__(self, message: str | None = None) -> None:
        """Create a new CannotMergeSingleCluster instance."""
        message_ = message or "Cannot merge any cluster. All members in one cluster."
        self.message = message_
        super().__init__(self.message)
