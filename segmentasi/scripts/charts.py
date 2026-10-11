import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLT = {"figsize": (9, 4.5), "dpi": 110}


def _save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    p = os.path.join(outdir, name)
    fig.savefig(p)
    plt.close(fig)
    return p


def charts(metrics, xy, labels, feats, profiles, rfm, outdir):
    paths = {}

    fig, ax = plt.subplots(**PLT)
    ax.plot(metrics["k"], metrics["elbow"], marker="o", label="inertia")
    ax.set_title("Elbow: inertia per jumlah klaster")
    ax.set_xlabel("k"); ax.set_ylabel("inertia"); ax.grid(alpha=0.3)
    paths["elbow"] = _save(fig, "elbow.png", outdir)

    fig, ax = plt.subplots(**PLT)
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", color="#8a4a2f")
    ax.set_title("Silhouette score per jumlah klaster")
    ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.grid(alpha=0.3)
    paths["silhouette"] = _save(fig, "silhouette.png", outdir)

    fig, ax = plt.subplots(**PLT)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=18, alpha=0.7)
    ax.set_title("Peta pelanggan PCA 2 komponen, diwarnai klaster")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    paths["pca"] = _save(fig, "pca.png", outdir)

    fig, ax = plt.subplots(**PLT)
    ax.bar(profiles["cluster_id"], profiles["n"], color="#3f6b8a")
    ax.set_title("Ukuran tiap segmen (jumlah pelanggan)")
    ax.set_xlabel("cluster"); ax.set_ylabel("pelanggan")
    for i, v in enumerate(profiles["n"]):
        ax.text(i, v + 10, str(v), ha="center", fontsize=9)
    paths["size"] = _save(fig, "cluster_size.png", outdir)

    fig, ax = plt.subplots(**PLT)
    x = np.arange(len(profiles))
    w = 0.27
    ax.bar(x - w, profiles["mean_freq"], w, label="freq (transaksi)", color="#5a7a8a")
    ax.bar(x, profiles["mean_spend"] / 1000, w, label="spend (ribu)", color="#8a6b2f")
    ax.bar(x + w, profiles["cancel_share"] * 100, w, label="share cancel (%)", color="#8a4a2f")
    ax.set_xticks(x); ax.set_xticklabels(profiles["cluster_id"])
    ax.set_title("Profil antar segmen")
    ax.legend()
    paths["profile"] = _save(fig, "cluster_profile.png", outdir)

    fig, ax = plt.subplots(**PLT)
    glob = rfm.merge(feats, on="customer_id")
    ax.scatter(glob["n_valid"], glob["total_spend"], c=labels, cmap="tab10",
               s=20, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Scatter frekuensi vs pengeluaran (skala log)")
    ax.set_xlabel("n transaksi"); ax.set_ylabel("total spend")
    paths["rf_scatter"] = _save(fig, "rfm_scatter.png", outdir)

    return paths