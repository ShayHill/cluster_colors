
import random

_top = 50000
_mid = _top // 2

RANDOM_INTS = [random.randint(1, _top) for i in range(100000)] 

def through_three_times() -> tuple[set[int], set[int], set[int]]:
    """Return a tuple of sets of integers that are in the list RANDOM_INTS
    exactly three times, exactly two times, and exactly one time, respectively.
    """
    random_ints = set(RANDOM_INTS)
    lowers = {x for x in random_ints if x < _mid}
    random_ints -= lowers
    uppers = {x for x in random_ints if x > _mid}
    random_ints -= uppers
    # middle = {x for x in random_ints if x == _mid}
    return lowers, uppers, random_ints

def cmp(a, b):
    return (a > b) - (a < b)

def partition_list():
    """Return a tuple of three lists of integers that are in the list RANDOM_INTS
    exactly three times, exactly two times, and exactly one time, respectively.
    """
    random_ints = set(RANDOM_INTS)
    parted = {x: set() for x in (-1, 0, 1)}
    for x in random_ints:
        parted[cmp(x, _mid)].add(x)
    return parted[-1], parted[0], parted[1]

# def create_list_then_return_set():
    # """Return a set of 10000 random integers between 1 and 50."""
    # return set([x for x in RANDOM_INTS])

# def add_ints_one_at_a_time_to_set():
    # """Add 10000 random integers between 1 and 50 to a set one at a time."""
    # return {x for x in RANDOM_INTS}

def race_functions():
    """Run through_three_times() and partition_list() and compare the time it
    takes to run each function.
    """
    import timeit
    print(timeit.timeit(partition_list, number=100))
    print(timeit.timeit(through_three_times, number=100))


    # from timeit import timeit
    # print("create_list_then_return_set took: ", timeit(create_list_then_return_set, number=100))
    # print("add_ints_one_at_a_time_to_set took: ", timeit(add_ints_one_at_a_time_to_set, number=100))    

if __name__ == "__main__":
    race_functions()

