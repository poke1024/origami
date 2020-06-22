from contextlib import contextmanager
from timeit import default_timer


# https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
