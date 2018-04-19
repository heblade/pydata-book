from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def studyseries():
    # obj = Series([4, 7, -5, 3])
    # print("obj: \n",obj, " index: \n", obj.index)

    # obj2 = Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
    # print("obj2: \n", obj2)
    # print(obj2["a"])
    # obj2["d"] = 6
    # print(obj2[["c","a","d"]])
    # print(obj2)
    # print(obj2[obj2 > 0])
    # print(obj2*2)
    # print(np.exp(obj2))
    # print("b" in obj2)
    # print("e" in obj2)

    sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
    obj3 = Series(sdata)
    print(obj3)

    states = ["California", "Ohio", "Oregon", "Texas"]
    obj4 = Series(sdata, index=states)
    print(obj4)
    # print(pd.isnull(obj4), obj4.isnull())
    # print(pd.notnull(obj4), obj4.notnull())
    obj5 = obj3 + obj4

    obj5.name = "population"
    obj5.index.name = "state"
    print(obj5)
    obj5.index = ["Anhui", "Guangdong", "Shanghai", "Beijing", "Henan"]
    print(obj5)

def studydataframe():
    data = {"state":["Ohio","Ohio","Ohio","Nevada","Nevada","Nevada"],
            "year":[2000,2001,2002,2001,2002,2003],
            "pop":[1.5,1.7,3.6,2.4,2.9,3.2]}
    frame = DataFrame(data)
    # print(frame)
    frame2 = DataFrame(data, columns=["year", "state", "pop"])
    # print(frame2)
    frame2 = DataFrame(data,
                       columns=["year", "state", "pop", "debt"],
                       index=["one","two","three","four", "five","six"])
    # print(frame2)
    # print(frame2["state"])
    # print(frame2["year"])
    # print(frame2.ix["three"])
    frame2["debt"] = 16.5
    # print(frame2)
    frame2["debt"] = np.arange(6)
    # print(frame2)
    val = Series([-1.2, -1.5, -1.7], index = ["two", "four", "five"])
    frame2["debt"] = val
    # print(frame2)
    frame2["eastern"] = (frame2.state == "Ohio")
    # print(frame2)
    del frame2["eastern"]
    # print(frame2.columns)
    data3 = {"Nevada":{2001:2.4,2002:2.9},"Ohio":{2000:1.5,2001:1.7,2002:3.6}}
    frame3 = DataFrame(data3, index=[2000, 2001, 2002])
    # print(frame3)
    # frame3.plot()
    # plt.show()
    # print(frame3.T)
    pdata = {"Ohio":frame3["Ohio"][:-1], "Nevada":frame3["Nevada"][:2]}
    # print(DataFrame(pdata))
    frame3.index.name = "year"
    frame3.columns.name = "state"
    print(frame3)
    print(frame3.values)
    print(frame2.values)

def studyindexobj():
    obj = Series(range(3), index=["a", "b", "c"])
    index = obj.index
    print(index)
    print(index[1:])
    # index[1] = "d"
    # print(index)
    index = pd.Index(np.arange(3))
    obj2 = Series([1.5, -2.5, 0], index = index)
    print(obj)
    print(obj2)
    print(obj2.index is index)


def studybasefunc():
    obj = Series([4.5, 7.2, -5.3, 3.6], index = ["d", "b", "a", "c"])
    # print(obj)
    obj2 = obj.reindex(["a","b","c","d","e"])
    # print(obj2)
    obj2 = obj.reindex(["a", "b", "c", "d", "e"], fill_value=0)
    # print(obj2)

    obj3 = Series(["blue","purple","yellow"], index=[0,2,4])
    print(obj3)
    #ffill 向前填充索引
    obj3 = obj3.reindex(range(6), method="ffill")
    print(obj3)

if __name__ == "__main__":
    # studyseries()
    # studydataframe()
    # studyindexobj()
    studybasefunc()