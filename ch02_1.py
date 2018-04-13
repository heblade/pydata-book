import pandas as pd

def readmovielensdata():
    unames = ["user_id", "gender", "age", "occupation", "zip"]
    users = pd.read_table("./datasets/movielens/users.dat", sep="::", header=None, names=unames)
    #print(users[:10])
    rnames = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_table("./datasets/movielens/ratings.dat", sep="::", header=None, names=rnames)
    #print(ratings[:10])
    mnames = ["movie_id", "title", "genres"]
    movies = pd.read_table("./datasets/movielens/movies.dat", sep="::", header=None, names=mnames)
    #print(movies[:10])
    data = pd.merge(pd.merge(ratings, users), movies)
    #print(data[:5])
    #print(data.ix[0])
    mean_ratings = data.pivot_table(index="title",
                                    values = "rating",
                                    columns="gender",
                                    aggfunc="mean")
    #print(mean_ratings[:10])

    ratings_by_title = data.groupby("title").size()
    #print(ratings_by_title[:10])
    active_titles = ratings_by_title.index[ratings_by_title >= 250]
    print(active_titles[:10])

    mean_ratings = mean_ratings.ix[active_titles]
    #print(mean_ratings[:10])
    top_female_ratings = mean_ratings.sort_index(by="F", ascending=False)
    #print(top_female_ratings[:10])

    mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]
    sorted_by_diff = mean_ratings.sort_index(by="diff")
    #分歧最大，且女性观众更喜欢的电影
    #print(sorted_by_diff[:15])
    #分歧最大，且男性观众更喜欢的电影
    #print(sorted_by_diff[::-1][:15])

    #得出男女分歧最大的电影标准差
    ratings_std_by_title = data.groupby("title")["rating"].std()
    ratings_std_by_title = ratings_std_by_title.ix[active_titles]
    print(ratings_std_by_title.sort_values(ascending=False)[:10])

if __name__ == "__main__":
    readmovielensdata()