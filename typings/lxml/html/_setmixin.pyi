"""
This type stub file was generated by pyright.
"""

class SetMixin(MutableSet):
    """
    Mix-in for sets.  You must define __iter__, add, remove
    """
    def __len__(self): # -> int:
        ...
    
    def __contains__(self, item): # -> bool:
        ...
    
    issubset = ...
    issuperset = ...
    union = ...
    intersection = ...
    difference = ...
    symmetric_difference = ...
    def copy(self): # -> set[Unknown]:
        ...
    
    def update(self, other): # -> None:
        ...
    
    def intersection_update(self, other): # -> None:
        ...
    
    def difference_update(self, other): # -> None:
        ...
    
    def symmetric_difference_update(self, other): # -> None:
        ...
    
    def discard(self, item): # -> None:
        ...
    


