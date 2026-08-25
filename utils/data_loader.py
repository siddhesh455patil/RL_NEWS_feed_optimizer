import pandas as pd

def load_data(train_path):
    news = pd.read_csv(f"{train_path}/news.tsv", sep="\t", header=None)
    behaviors = pd.read_csv(f"{train_path}/behaviors.tsv", sep="\t", header=None)

    # IMPORTANT: MIND dataset has MORE columns
    news.columns = [
        "id", "category", "subcategory",
        "title", "abstract", "url",
        "title_entities", "abstract_entities"
    ]

    behaviors.columns = [
        "imp_id", "user", "time",
        "history", "impressions"
    ]

    return news, behaviors