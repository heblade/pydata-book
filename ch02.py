import json
from collections import defaultdict
from collections import Counter
import time
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "./datasets/bitly_usagov/example.txt"
def readfile():
    print(open(path).readline())

def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    #截取倒数第n位到结尾
    return value_key_pairs[-n:]

def top_counts2(sequence):
    counts = Counter(sequence)
    return counts.most_common(10)

def readjson(records):
    #print(records[0])
    #print(records[0]["tz"])
    time_zones = [rec["tz"] for rec in records if "tz" in rec]
    #print(time_zones[:10])
    print("Total record count: %d" % (len(time_zones)))
    time_start = time.time()
    counts1 = get_counts(time_zones)
    print("get counts1: %d" % (counts1["America/New_York"]))
    time_end = time.time()
    print('totally cost for count1: ', time_end - time_start)

    time_start = time.time()
    counts2 = get_counts2(time_zones)
    print("get counts2: %d" % (counts2["America/New_York"]))
    time_end = time.time()
    print('totally cost for count2: ', time_end - time_start)

    print(top_counts(counts2))
    print(top_counts2(time_zones))

def practicepandas(records):
    frame = DataFrame(records)
    #print(frame["tz"][:10])
    #tz_counts = frame["tz"].value_counts()
    #print(tz_counts[:10])

    clean_tz = frame["tz"].fillna("Missing")
    clean_tz[clean_tz == ""] = "Unknown"
    tz_counts = clean_tz.value_counts()
    print(tz_counts[:10])
    tz_counts[:10].plot(kind="barh", rot=0, figsize=(10,5))
    plt.show()

def filterdata(records):
    frame = DataFrame(records)
    results = Series([x.split()[0] for x in frame.a.dropna()])
    print(results[:10])
    print(results.value_counts()[:8])

def whetherwindow(records):
    frame = DataFrame(records)
    cframe = frame[frame.a.notnull()]
    operating_system = np.where(cframe["a"].str.contains("Windows"),
                                "Windows",
                                "Not Windows")
    #print(cframe[:5])
    #print(operating_system[:10])
    by_tz_os = cframe.groupby(["tz", operating_system])
    agg_counts = by_tz_os.size().unstack().fillna(0)
    print(agg_counts[:10])
    indexer = agg_counts.sum(1).argsort()
    print(indexer[:10])
    count_subset = agg_counts.take(indexer)[-10:]
    print(count_subset)
    #count_subset.plot(kind="barh", stacked=True, figsize=(10,5))
    print(count_subset.sum(1))
    normed_subset = count_subset.div(count_subset.sum(1), axis=0)
    normed_subset.plot(kind="barh", stacked=True, figsize=(10,5))
    plt.show()

if __name__ == "__main__":
    records = [json.loads(line, encoding="utf-8") for line in open(path)]
    #readfile(records)
    #readjson(records)
    #practicepandas(records)
    #filterdata(records)
    whetherwindow(records)
