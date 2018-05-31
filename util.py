from functools import wraps
import errno
import os
import signal
import numpy as np

from progressbar import ProgressBar, Percentage, Bar, ETA
import itertools

class TimeoutError(Exception):
    pass

class timeout(object):
    def __init__(self, seconds):
        self.seconds = seconds

    def __call__(self, f):
        def _handle_timeout(signum, fname):
            raise TimeoutError()

        def wrapped_f(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(self.seconds)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapped_f

def make_progressbar(val):
    return ProgressBar(widgets=[Percentage(),
                                Bar(left=" |", right="| "),
                                ETA()],
                       maxval=val)

def list_to_idx_dict(lst):
    res = {}
    for i, x in enumerate(lst):
        res[i] = x
    return res

def dict_mean(d):
    lst = []
    for key in d:
        lst.append(d[key])
    return np.mean(lst)

def filter_dict(f, d):
    return { k: v for k, v in d.items() if f(k, v) }

def map_dict(f, d):
    return { k: f(v) for k, v in d.items() }

def filter_dataframe_by_column(f, data):
    return data[[col for col in data.columns if
                 len(list(filter(lambda x: f(x), data[col].values))) == 0]]

def merge_dicts(d1, d2):
    if d1 == {}:
        return d2
    if d1.keys() != d2.keys():
        raise Exception("Keys must match: {} != {}".format(d1.keys(), d2.keys()))
    for k in d1:
        d1[k] += d2[k]
    return d1

def product_dict(d):
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def average(f, n_times):
    res = []
    for n in range(n_times):
        res.append(f())
    return np.mean(res)
