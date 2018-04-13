import pandas as pd

def handlenames():
    years = range(1880, 2011)
    pieces = []
    columns = ["name", "sex", "births"]
    for year in years:
        path = "./datasets/babynames/yob%d.txt" % (year)
        frame = pd.read_csv(path,
                            names=columns)
        pieces.append(frame)
    names = pd.concat(pieces, ignore_index=True)
    print(names.groupby("name").births.sum().sort_values(ascending=False)[:100])
    print(names.groupby("sex").births.sum())

if __name__ == "__main__":
        handlenames()