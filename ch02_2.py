import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def handlenames():
    years = range(1880, 2011)
    pieces = []
    columns = ["name", "sex", "births"]
    for year in years:
        path = "./datasets/babynames/yob%d.txt" % (year)
        frame = pd.read_csv(path,
                            names=columns)
        frame["year"] = year
        pieces.append(frame)
    names = pd.concat(pieces, ignore_index=True)
    #print(names.groupby("name").births.sum().sort_values(ascending=False)[:100])
    #print(names.groupby("sex").births.sum())

    total_births = names.pivot_table(index="year",
                                    values = "births",
                                    columns="sex",
                                    aggfunc="sum")
    #print(total_births.tail())
    #total_births.plot(title="Total births by sex and year")
    #plt.show()

    #按照year, sex进行分组，并加入year, sex分组后的姓名比例列
    names = names.groupby(["year","sex"]).apply(add_prop)
    #print(names[:100])
    #print(names[(names["year"]==1880) & (names["sex"]=="F")].prop.sum())
    #print(np.allclose(names.groupby(["year","sex"]).prop.sum(), 1))
    grouped = names.groupby(["year", "sex"])
    top1000 = grouped.apply(get_top1000)

    total_births = top1000.pivot_table(index="year",
                                       values = "births",
                                       columns="name",
                                       aggfunc="sum")
    #print(total_births[:10])

    #统计几个典型名字在各年份的变化情况
    # subset = total_births[["John", "Harry", "Mary", "Marilyn", "David", "Isabela"]]
    # subset.plot(subplots=True,
    #             figsize=(12,10),
    #             grid=False,
    #             title="Number of births per year")
    # plt.show()

    #统计男生，女生最流行前1000姓名在男女总姓名中的比例
    # table = top1000.pivot_table(index="year",
    #                             values="prop",
    #                             columns="sex",
    #                             aggfunc="sum")
    # print(table[:100])
    # table.plot(title="Sum of table1000.prop by year and sex",
    #            yticks=np.linspace(0, 1.2, 13),
    #            xticks=range(1880, 2020, 10))
    # plt.show()

    boys = top1000[top1000.sex=="M"]
    df = boys[boys.year==2010]
    #print(df.sort_index(by="prop", ascending=False)[:100])
    prop_cumsum = df.sort_index(by="prop", ascending=False).prop.cumsum()
    #print(prop_cumsum[:10])
    #print(prop_cumsum.searchsorted(0.5)+1)

    df1900 = boys[boys.year == 1900]
    in1900 = df1900.sort_index(by="prop",
                               ascending=False).prop.cumsum()
    #print(in1900.searchsorted(0.5)+1)

    # diversity = top1000.groupby(["year", "sex"]).apply(get_quantile_count)
    # diversity = diversity.unstack("sex")
    # print(diversity.head())
    # diversity.plot(title="Number of popular name in top 50%")
    # plt.show()
    get_last_letter = lambda x: x[-1]
    last_letters = names.name.map(get_last_letter)
    #print(last_letters)
    last_letters.name = "last_letter"
    table = names.pivot_table(index=last_letters,
                              values = "births",
                              columns=["sex","year"],
                              aggfunc="sum")
    #print(table[:10])
    subtable = table.reindex(columns=[1910, 1960, 2010], level="year")
    #print(subtable.head())
    #print(subtable.sum())
    # letter_prop = subtable / subtable.sum().astype(float)
    # print(letter_prop[:10])
    # fig, axes = plt.subplots(2, 1, figsize=(10,8))
    # letter_prop["M"].plot(kind="bar", rot=0, ax=axes[0], title="Male")
    # letter_prop["F"].plot(kind="bar", rot=0, ax=axes[1], title="Female", legend=False)
    # plt.show()

    # letter_prop = table / table.sum().astype(float)
    # dny_ts = letter_prop.ix[["d", "n", "y"], "M"].T
    # print(dny_ts.head())
    # dny_ts.plot()
    # plt.show()

    all_names = top1000.name.unique()
    #print(all_names[:10])
    mask = np.array(["lesl" in x.lower() for x in all_names])
    # print(mask)
    lesley_like = all_names[mask]
    # print(lesley_like[:10])
    filtered = top1000[top1000.name.isin(lesley_like)]
    #print(filtered.groupby("name").births.sum())
    table = filtered.pivot_table(index="year",
                                 values = "births",
                                 columns="sex",
                                 aggfunc="sum")
    #print(table.sum(1))
    table = table.div(table.sum(1), axis=0)
    #print(table.tail())
    table.plot(style={"M":"k-", "F":"k--"})
    plt.show()

def add_prop(group):
    births = group.births.astype(float)
    group["prop"]=births/births.sum()
    return group

def get_top1000(group):
    return group.sort_index(by="births", ascending=False)[:1000]

def get_quantile_count(group, q=0.5):
    group = group.sort_index(by="prop", ascending=False)
    position = group.prop.cumsum().searchsorted(q) + 1
    return position[0]

if __name__ == "__main__":
        handlenames()