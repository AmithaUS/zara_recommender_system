import pickle

similarity_matrix = pickle.load(open("models/similarity.pkl", "rb"))
train_df = pickle.load(open("models/train_df.pkl", "rb"))


def recommend(product_name, top_n=6):

    # Find index safely
    matches = train_df.index[train_df["name"] == product_name].tolist()

    if len(matches) == 0:
        return []

    idx = matches[0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    recommendations = []

    for i in similarity_scores:
        product = train_df.iloc[i[0]]

        recommendations.append({
            "name": product["name"],
            "price": product["price"],
            "category": product["Product Category"],
            "url": product["url"],
            "score": round(i[1], 3)
        })

    return recommendations

