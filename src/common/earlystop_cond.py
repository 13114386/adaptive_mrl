from __future__ import unicode_literals, print_function, division


class EarlyStopRougeCondition():
    '''
        Stop if ROUGE scores flatten consecutively.
    '''
    def __init__(
        self,
        stop_count,
        init_score = None
    ):
        self.score = init_score
        self.stop_count = stop_count
        self.count = 0

    def __call__(self, score, logger):
        criterion = score == self.score
        if criterion:
            self.count += 1
        else:
            self.score = score
            self.count = 0
        early_stop = self.count >= self.stop_count
        if early_stop and logger is not None:
            logger.info(f'Early stopping condition is met.')
        return early_stop


class EarlyStopConditionByCount():
    def __init__(self, stop_steps, verbose=False):
        self.stop_steps = stop_steps
        self.step_counter = 0
        self.verbose = verbose

    def __call__(self):
        stop = self.step_counter >= self.stop_steps
        if stop and self.verbose:
            print("Early stop by count!")
        return stop

    def incr(self):
        self.step_counter += 1

    def reset(self):
        self.step_counter = 0


from common.metric_meter import MetricMeter
class EarlyStopTimeLimitCondition():
    def __init__(self, time_limit):
        '''
            time_limit: in seconds. less than or equal to 0 means no limit.
        '''
        self.time_limit = time_limit
        self.meter = MetricMeter()

    def __call__(self, elapsed):
        if self.time_limit <= 0:
            return False
        self.meter(elapsed)
        # Trigger stop if (the elapsed time + average time) exceeds the time limit.
        return self.meter.average() + self.meter.total() >= self.time_limit
