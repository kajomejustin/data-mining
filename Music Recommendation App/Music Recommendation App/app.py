from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np
import random

app = Flask(__name__)

# -------------------------
# SONG DATASET
# -------------------------
music_data = [
    {"Title": "Katerina", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Ikinya", "Year": 2019},
    {"Title": "Saa Moya", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Ikinya", "Year": 2021},
    {"Title": "Katapilla", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Mixed", "Year": 2020},
    {"Title": "Slowly", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2017},
    {"Title": "My Vow", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2021},
    {"Title": "Ntawamusimbura", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2016},
    {"Title": "Habibi", "Artist": "The Ben", "Genre": "Afropop", "Album": "Kigali Love", "Year": 2022},
    {"Title": "Ndaje", "Artist": "The Ben", "Genre": "Afropop", "Album": "Kigali Love", "Year": 2017},
    {"Title": "Bad", "Artist": "Ariel Wayz", "Genre": "RnB", "Album": "Self Love", "Year": 2022},
    {"Title": "10 Days", "Artist": "Ariel Wayz", "Genre": "RnB", "Album": "Self Love", "Year": 2023},
    {"Title": "Ready", "Artist": "Bwiza", "Genre": "Afropop", "Album": "Bwiza Season", "Year": 2022},
    {"Title": "Ubudodo", "Artist": "Bwiza", "Genre": "Afropop", "Album": "Bwiza Season", "Year": 2023},
    {"Title": "Anytime", "Artist": "Mike Kayihura", "Genre": "RnB", "Album": "Zuba", "Year": 2021},
    {"Title": "Sabrina", "Artist": "Mike Kayihura", "Genre": "RnB", "Album": "Zuba", "Year": 2022},
    {"Title": "Madiba", "Artist": "Kivumbi King", "Genre": "Hip-hop", "Album": "Igikwe", "Year": 2021},
    {"Title": "Pasta", "Artist": "Kivumbi King", "Genre": "Hip-hop", "Album": "Igikwe", "Year": 2023},
    {"Title": "Amata", "Artist": "Social Mula", "Genre": "Traditional", "Album": "Mula Mix", "Year": 2018},
    {"Title": "Superstar", "Artist": "Social Mula", "Genre": "RnB", "Album": "Mula Mix", "Year": 2019}
]

music_df = pd.DataFrame(music_data)
music_df['TrackID'] = ['T{:04d}'.format(i+1) for i in range(len(music_df))]

# -------------------------
# SIMULATED LISTENING HISTORY
# -------------------------
users = [f"U{str(i).zfill(3)}" for i in range(1, 31)]
track_ids = music_df['TrackID'].tolist()
listens = []

for user in users:
    sampled = random.sample(track_ids, random.randint(4, 8))
    for track in sampled:
        listens.append({"UserID": user, "TrackID": track, "Rating": random.choice([4, 5])})

listens_df = pd.DataFrame(listens)

# -------------------------
# FEATURE ENCODING & COSINE SIMILARITY
# -------------------------
features = music_df.copy()
features['GenreCode'] = LabelEncoder().fit_transform(features['Genre'])
features['ArtistCode'] = LabelEncoder().fit_transform(features['Artist'])
X = features[['GenreCode', 'ArtistCode', 'Year']]
X_scaled = StandardScaler().fit_transform(X)

def get_best_k(X):
    distortions = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(km.inertia_)
    kl = KneeLocator(range(1, 10), distortions, curve="convex", direction="decreasing")
    return kl.elbow or 3

k = get_best_k(X_scaled)
KMeans(n_clusters=k, random_state=42).fit(X_scaled)
cos_sim = cosine_similarity(X_scaled)
trackid_to_index = dict(zip(features['TrackID'], features.index))

# -------------------------
# USER-ITEM MATRIX
# -------------------------
user_item_matrix = listens_df.pivot_table(index="UserID", columns="TrackID", values="Rating", fill_value=0)

# -------------------------
# RECOMMENDATION FUNCTION (Hybrid)
# -------------------------
def get_recommendations(track_id, top_n=3):
    if track_id not in trackid_to_index:
        return []

    # Content-based
    idx = trackid_to_index[track_id]
    content_scores = list(enumerate(cos_sim[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    content_recs = [features.iloc[i]["TrackID"] for i, _ in content_scores]

    # Collaborative-based
    if track_id in user_item_matrix.columns:
        users = user_item_matrix[user_item_matrix[track_id] > 0].index
        subset = user_item_matrix.loc[users]
        co_scores = subset.drop(columns=track_id).sum().sort_values(ascending=False)
        collab_recs = co_scores.head(top_n).index.tolist()
    else:
        collab_recs = []

    # Merge (prioritize intersection, then union)
    hybrid = list(dict.fromkeys(content_recs + collab_recs))[:top_n]
    return music_df[music_df["TrackID"].isin(hybrid)].to_dict(orient="records")

# -------------------------
# FLASK ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    selected_title = None
    recommendations = []

    if request.method == "POST":
        selected_title = request.form.get("song")
        track_row = music_df[music_df["Title"] == selected_title].iloc[0]
        track_id = track_row["TrackID"]
        recommendations = get_recommendations(track_id)

    return render_template("index.html", songs=music_df["Title"].tolist(), selected_title=selected_title, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)