import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CPU_COLS = [
    "recency_days", "n_valid", "total_spend", "avg_price",
    "cancel_share", "avg_duration_days",
]

PCA_COLS = CPU_COLS + ["active_span_days", "premium_share"]


def prepare_matrix(feats):
    feats = feats.copy()
    feats[CPU_COLS] = feats[CPU_COLS].fillna(0)
    X = StandardScaler().fit_transform(feats[CPU_COLS])
    return feats, X


def elbow_and_silhouette(X, kmax=9):
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, random_state=7, n_init=10).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(rows)


def run_clustering(feats, k):
    feats, X = prepare_matrix(feats)
    km = KMeans(n_clusters=k, random_state=7, n_init=10).fit(X)
    labels = km.labels_
    pca = PCA(n_components=2, random_state=7).fit(X)
    xy = pca.transform(X)
    return {
        "labels": labels,
        "X": X,
        "km": km,
        "pca": pca,
        "xy": xy,
        "explained": pca.explained_variance_ratio_.cumsum().tolist(),
    }


def profile_clusters(feats, labels, k):
    feats = feats.copy()
    feats["cluster"] = labels
    rows = []
    for cid in range(k):
        sub = feats[feats["cluster"] == cid]
        rows.append({
            "cluster_id": cid,
            "n": len(sub),
            "mean_recency_days": round(sub["recency_days"].mean(), 1) if sub["recency_days"].notna().any() else None,
            "mean_freq": round(sub["n_valid"].mean(), 2),
            "mean_spend": round(sub["total_spend"].mean(), 0),
            "avg_price": round(sub["avg_price"].mean(), 0) if sub["avg_price"].notna().any() else None,
            "cancel_share": round(sub["cancel_share"].mean(), 3),
            "churn_ever_share": round(sub["churn_ever"].mean(), 3),
            "premium_share": round(sub["premium_share"].mean(), 3),
            "mean_duration": round(sub["avg_duration_days"].mean(), 1) if sub["avg_duration_days"].notna().any() else None,
            "top_channel": sub["preferred_channel"].mode().iloc[0] if len(sub) else None,
            "dominant_platform": sub["device_platform"].dropna().mode().iloc[0]
                                 if sub["device_platform"].notna().any() else None,
        })
    return pd.DataFrame(rows)


def compute_rfm(feats):
    f = feats.copy()
    f["r_rank"] = f["recency_days"].fillna(f["recency_days"].max()).rank(method="first")
    f["f_rank"] = f["n_valid"].rank(method="first")
    f["m_rank"] = f["total_spend"].rank(method="first")
    f["r_score"] = (6 - pd.qcut(f["r_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int))
    f["f_score"] = pd.qcut(f["f_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    f["m_score"] = pd.qcut(f["m_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    f["rfm_score"] = (f["r_score"] * 100 + f["f_score"] * 10 + f["m_score"])
    return f[["customer_id", "r_score", "f_score", "m_score", "rfm_score"]].fillna(0)


def save_results(db_path, feats, labels, profiles, rfm):
    con = sqlite3.connect(db_path)
    for _, p in profiles.iterrows():
        con.execute("INSERT INTO seg_cluster (cluster_id, n_customers, mean_recency_days, "
                    "mean_spend, mean_freq, cancel_share) VALUES (?,?,?,?,?,?)",
                    (int(p["cluster_id"]), int(p["n"]), p["mean_recency_days"],
                     p["mean_spend"], p["mean_freq"], p["cancel_share"]))
    rows = []
    for cid, cust in zip(labels, feats["customer_id"]):
        r = rfm[rfm["customer_id"] == cust].iloc[0]
        rows.append((cust, int(cid), int(r["r_score"]), int(r["f_score"]),
                     int(r["m_score"]), int(r["rfm_score"])))
    con.executemany("INSERT INTO seg_assignment (customer_id, cluster_id, r_score, f_score, "
                    "m_score, rfm_score) VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()