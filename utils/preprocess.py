def parse_impressions(impressions):
    pairs = []

    for item in impressions.split():
        news_id, label = item.split("-")
        pairs.append((news_id, int(label)))

    return pairs

def build_news_category_map(news_df):
    news_df["category"] = news_df["category"].str.lower()
    return dict(zip(news_df["id"], news_df["category"]))

import numpy as np

CATEGORIES = [
    "sports", "news", "finance", "entertainment",
    "lifestyle", "health", "autos", "travel",
    "foodanddrink", "video"
]

CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}
INDEX_TO_CATEGORY = {i: cat for cat, i in CATEGORY_TO_INDEX.items()}

def history_to_vector(history, news_map):
    vector = np.zeros(len(CATEGORIES))

    if isinstance(history, float):  # handle NaN
        return vector

    for news_id in history.split():
        if news_id in news_map:
            cat = news_map[news_id]
            if cat in CATEGORIES:
                idx = CATEGORIES.index(cat)
                vector[idx] += 1

    return vector

def create_rl_dataset(behaviors_df, news_map):

    rl_data = []

    for _, row in behaviors_df.iterrows():

        history = row["history"]
        impressions = row["impressions"]

        if isinstance(history, float):
            continue

        state = history_to_vector(history, news_map)

        parsed = parse_impressions(impressions)

        for news_id, label in parsed:

            if news_id not in news_map:
                continue

            action = news_map[news_id]

            reward = 1 if label == 1 else 0

            rl_data.append({
                "state": state,
                "action": action,
                "reward": reward
            })

    return rl_data