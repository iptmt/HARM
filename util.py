import os
import time
import logging


logger = logging.getLogger(__name__)

class LossClock:
    def __init__(self, loss_names, interval, silence=False):
        self.losses = {name: 0. for name in loss_names}
        self.interval = interval
        self.silence = silence
        self.last_time = time.time()
        self.cnt = 0
    
    def update(self, named_loss):
        assert type(named_loss) is dict
        for name in named_loss:
            assert name in self.losses
            self.losses[name] += named_loss[name]
        self.cnt += 1
        if self.cnt % self.interval == 0:
            now = time.time()
            time_cost = int(now - self.last_time)
            for name in self.losses:
                self.losses[name] /= self.interval
            loss_repr = '; '.join(["%s: %.4f" % (name, self.losses[name]) for name in self.losses])
            info = "[Steps] => %d. [Losses] => %s. [Time cost] => %d min %d s." % (self.cnt, loss_repr, time_cost // 60, time_cost % 60)
            if not self.silence:
                logger.info(info)
            for name in self.losses:
                self.losses[name] = 0.
            self.last_time = now

def create_logger(log_path, log_name, debug=False):
    logFormatter = logging.Formatter("[%(levelname)s] [%(asctime)s] -- %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    if not debug:
        logger_file = os.path.join(log_path, log_name)
        fileHandler = logging.FileHandler(logger_file)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    return rootLogger

def mrr(preds, labels):
    sum_query, sum_rank_score = 0, 0.
    for pred, label in zip(preds, labels):
        if not isinstance(pred, list):
            pred = pred.cpu().numpy().tolist()
        if not isinstance(label, list):
            label = label.cpu().numpy().tolist()
        tuples = list(zip(pred, label))
        tuples.sort(key=lambda x: x[0], reverse=True)
        for idx, (_, l) in enumerate(tuples):
            if l == 1:
                sum_rank_score += 1/(idx + 1)
        sum_query += 1
    return sum_rank_score / sum_query

# p.shape=n.shape=(B,)
def max_margin_loss(p, n, anchor=1.0):
    margin = anchor - p + n
    return (margin * (margin > 0).float()).sum()