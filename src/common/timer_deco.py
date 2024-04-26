from __future__ import unicode_literals, print_function, division
from contextlib import ContextDecorator
import time

class timerdeco(ContextDecorator):
    def __init__(self, desc="", verbose=True):
        self.desc = desc
        self.elapsed = 0.0
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(f"(timerdeco) {self.desc} starts...")
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            print(f"(timerdeco) {self.desc} ends with elapsed time: {self.elapsed}")
        return False

if __name__ == "__main__":
    @timerdeco()
    def test():
        print("Blabla")
    test()
