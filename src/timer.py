#!/usr/bin/env python
# coding: utf-8

# Imports
import time
import timeit

# Python Timer Class
class Timer:
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start

    def __float__(self):
        return self.elapsed

    def __str__(self):
        return self.verbose()

    def verbose(self):
        if self.elapsed is None:
            return '<not-measured>'
        return time.strftime('%H:%M:%S', time.gmtime(self.elapsed))