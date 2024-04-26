from __future__ import unicode_literals, print_function, division
import os
from collections import Counter
from itertools import groupby


class ReplicationStatistic():
    @staticmethod
    def compute(folder, text_triples, srcname):
        summaries = [tpl["summary"][0] if isinstance(tpl["summary"], list) else tpl["summary"] \
                        for tpl in text_triples]
        ReplicationStatistic.compute_stats(folder,
                                           summaries,
                                           consecutive=True,
                                           outputname=f"{srcname}.consecutive.dup.stat")
        ReplicationStatistic.compute_stats(folder,
                                           summaries,
                                           consecutive=False,
                                           outputname=f"{srcname}.dup.stat")

    @staticmethod
    def compute_stats(folder, summaries, consecutive=False, outputname=None, decimal_n=4):
        dup_counter = Counter()
        total_count = 0
        for l in summaries:
            tokens = l.split()
            if tokens[-1] == '.':
                tokens = tokens[:-1]
            if consecutive:
                for key, group in groupby(tokens):
                    group = list(group)
                    dup_counter.update({key: len(group)} if len(group) > 1 else {})
            else:
                dup_counter.update({k: c for k, c in Counter(tokens).items() if c > 1})
            total_count += len(tokens)
        repeat_sum = sum(dup_counter.values())
        agg_stat = {"#total_word_count#": total_count,
                    "#sum_dup_count#": repeat_sum,
                    "#repeat_%#": round(float(repeat_sum)/float(total_count), decimal_n)}
        stat = {**agg_stat, **dup_counter}
        ofpath = os.path.join(folder, outputname)
        with open(ofpath, "w", encoding="utf-8") as f:
            print(stat, file=f)


if __name__ == "__main__":
    def load_specific(fpath, key):
        import json
        from itertools import chain
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            keyed_data = list(chain.from_iterable([d[key] for d in data]))
            return keyed_data

    consecutive = False
    folder = "D:\\Dev\\Projects\\Data\\ats\\CA2\\convolutional\\bart\\baseline\\eval_result"
    inputname = "eval.text.result.json"
    consec = ".consecutive" if consecutive else ""
    outputname = f"eval.summary{consec}.dup.stat"
    summaries = load_specific(os.path.join(folder, inputname), key="summary")
    ReplicationStatistic.compute_stats(folder,
                        summaries,
                        consecutive=consecutive,
                        outputname=outputname)
