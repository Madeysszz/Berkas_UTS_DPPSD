import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charts as ch  # noqa: E402
import clustering as cl  # noqa: E402
import etl_features as fx  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "segmentasi.db")
DATA = os.path.join(BASE, "data")
SCHEMA = os.path.join(BASE, "sql", "segment_schema.sql")
OUT = os.path.join(BASE, "assets")


def main():
    print("=== [1] Feature engineering per pelanggan ===")
    feats = fx.build_features(DB, DATA, SCHEMA)
    print("  pelanggan:", len(feats))
    print(feats[["customer_id", "n_valid", "total_spend", "recency_days",
                 "cancel_share"]].head(8).to_string(index=False))

    print("\n=== [2] RFM ===")
    rfm = cl.compute_rfm(feats)
    print(rfm.describe().to_string())

    print("\n=== [3] Elbow + silhouette ===")
    _, X = cl.prepare_matrix(feats)
    m = cl.elbow_and_silhouette(X, kmax=9)
    print(m.to_string(index=False))
    sil_best = int(m.loc[m["silhouette"].idxmax(), "k"])
    print("  k terbaik murni (silhouette):", sil_best)
    print("  catatan: k=2 hanya memisahkan segmen premium kecil; untuk segmen actionable")
    print("  dipilih k=4 (tekuk elbow, siluet stabil, granularitas cukup untuk strategi)")
    best_k = 4

    print("\n=== [4] K-Means k=", best_k, "===")
    res = cl.run_clustering(feats, best_k)
    labels = res["labels"]

    print("\n=== [5] Profil segmen ===")
    prof = cl.profile_clusters(feats, labels, best_k)
    print(prof.to_string(index=False))

    print("\n=== [6] Simpan hasil + grafik ===")
    cl.save_results(DB, feats, labels, prof, rfm)
    m2 = pd.DataFrame({"k": m["k"], "elbow": m["inertia"], "silhouette": m["silhouette"]})
    paths = ch.charts(m2, res["xy"], labels, feats, prof, rfm, OUT)
    for k, v in paths.items():
        print("  ", k, "->", v)


if __name__ == "__main__":
    main()