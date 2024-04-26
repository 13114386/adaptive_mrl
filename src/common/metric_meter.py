from __future__ import absolute_import, division, print_function, unicode_literals


class MetricMeter():
    def __init__(self, ratio=1.0):
        self.history = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.ratio = ratio

    def __call__(self, value):
        self.history += [value]
        if 0. < self.ratio < 1.0:
            self._apply_ratio(value)
        else:
            self._apply_sum_avg(value)
            
    def history(self):
        return self.history

    def average(self):
        return self.avg

    def total(self):
        return self.sum

    def _apply_sum_avg(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def _apply_ratio(self, value):
        self.sum += value
        if self.count == 0:
            self.avg = value
        else:
            self.avg = self.ratio * self.avg + (1. - self.ratio) * value
        self.count += 1


class ProgressCounter():
    def __init__(self, init_count=0):
        self.counter = init_count

    @property
    def count(self):
        return self.counter

    def __iadd__(self, value):
        self.counter += value
        return self
