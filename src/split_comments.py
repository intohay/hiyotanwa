
import pandas as pd

df = pd.read_csv("data/comments_parsed.csv")


kawaii_df = df[df["kawaii"].notnull()]
like_df = df[df["like"].notnull()]
food_df = df[df["food"].notnull()]
other_df = df[df["other"].notnull()]

# keep id and comment field

kawaii_df = kawaii_df[["id", "name", "kawaii"]]
like_df = like_df[["id", "name", "like"]]
food_df = food_df[["id", "name", "food"]]
other_df = other_df[["id", "name", "other"]]

kawaii_df.to_csv("data/kawaii.csv", index=False)
like_df.to_csv("data/like.csv", index=False)
food_df.to_csv("data/food.csv", index=False)
other_df.to_csv("data/other.csv", index=False)

