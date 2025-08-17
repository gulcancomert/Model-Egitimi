import time
from contextlib import contextmanager

@contextmanager
def timer():
    t0 = time.perf_counter()
    yield lambda: (time.perf_counter() - t0) * 1000.0
