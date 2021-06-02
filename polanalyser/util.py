import importlib
is_numba_available = importlib.util.find_spec("numba") is not None
if is_numba_available:
    from numba import njit

def njit_if_available(*args, **keywords):
    def _njit_if_available(func):
        if is_numba_available:
            return njit(*args, **keywords)(func)
        else:
            return func

    return _njit_if_available
