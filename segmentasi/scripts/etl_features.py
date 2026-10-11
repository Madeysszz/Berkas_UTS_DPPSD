import csv
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    if header and header[0] == "":
        header[0] = "id"
    return header, rows[1:]


def parse_dt(value):
    if not value:
        return None
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})", value)
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)),
                        int(m.group(4)), int(m.group(5)), int(m.group(6)))
    except ValueError:
        return None


def fmt_dt(value):
    d = parse_dt(value)
    return d.strftime("%Y-%m-%d %H:%M:%S") if d else None


def load_logs(path, detail_col):
    header, data = read_csv_rows(path)
    df = pd.DataFrame(data, columns=header)
    if "id" in df.columns and "transaction_id" not in df.columns:
        df = df.drop(columns=["id"])
    for c in ["bill_number", "status_activated", "price", "pricing_id",
              "subscription_start_date", "subscription_expired_at",
              "subscription_cancel_date", "device_data", "camp_id"]:
        if c not in df.columns:
            df[c] = None
    df["detail_id"] = df[detail_col]
    df["channel_id"] = df["bill_number"].fillna("").map(
        lambda b: b.split("-")[0].upper() if b else "UNKNOWN")
    df["is_test"] = df["bill_number"].fillna("").str.startswith("TESTLOCAL").astype(int)
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype("int64")
    df["is_revenue_valid"] = (df["price_num"] > 0).astype(int)
    df["start_clean"] = df["subscription_start_date"].map(fmt_dt)
    df["expired_clean"] = df["subscription_expired_at"].map(fmt_dt)
    df["cancel_clean"] = df["subscription_cancel_date"].map(fmt_dt)

    def is_blank(x):
        return x is None or (isinstance(x, float) and pd.isna(x)) or x == ""

    def duration(r):
        if is_blank(r["start_clean"]) or is_blank(r["expired_clean"]):
            return None
        try:
            d = datetime.fromisoformat(str(r["expired_clean"])) - datetime.fromisoformat(str(r["start_clean"]))
            return d.days if d.days > 0 else None
        except Exception:
            return None
    df["duration_days"] = df.apply(duration, axis=1)

    def churn(r):
        if r["status_activated"] != "CANCEL":
            return 0
        if is_blank(r["expired_clean"]) or is_blank(r["cancel_clean"]):
            return 1
        return int(str(r["cancel_clean"]) < str(r["expired_clean"]))
    df["is_churn"] = df.apply(churn, axis=1)
    other = {}
    if "device_data" in df.columns:
        d = df["device_data"].map(lambda v: None if not v or v == "null" else v)
        try:
            import json
            def plat(v):
                try:
                    return json.loads(v).get("platform") or None
                except Exception:
                    return None
            df["platform"] = d.map(plat)
        except Exception:
            df["platform"] = None
    else:
        df["platform"] = None
    return df


def build_features(db_path, data_dir, schema_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(open(schema_path).read())

    retail = load_logs(os.path.join(data_dir, "retail_transaction_log.csv"),
                       "retail_transaction_log_detail_id")
    cobrand = load_logs(os.path.join(data_dir, "cobrandgen_transaction_log.csv"),
                        "cobrandgen_transaction_log_detail_id")
    all_logs = pd.concat([retail, cobrand], ignore_index=True)
    all_logs = all_logs[all_logs["is_test"] == 0]
    valid = all_logs[all_logs["customer_id"].notna() & (all_logs["customer_id"] != "")]

    ref_max = valid["start_clean"].dropna().max()

    rows = []
    for cid, g in valid.groupby("customer_id"):
        last = g["start_clean"].dropna().max()
        first = g["start_clean"].dropna().min()
        def _blank(x):
            return x is None or (isinstance(x, float) and pd.isna(x))
        rec = (datetime.fromisoformat(ref_max) - datetime.fromisoformat(last)).days \
            if (ref_max and not _blank(last)) else None
        span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days \
            if (not _blank(first) and not _blank(last)) else 0
        prices = g["price_num"]
        v = g[g["is_revenue_valid"] == 1]
        total = int(v["price_num"].sum()) if len(v) else 0
        n_cancel = int((g["status_activated"] == "CANCEL").sum())
        n = len(g)
        band = pd.cut(prices if len(prices) else pd.Series([0]), bins=[-1, 9999, 15000, 10**10],
                      labels=["low", "mid", "premium"]).value_counts()
        rows.append({
            "customer_id": cid,
            "n_rows": n,
            "n_valid": int(v.shape[0]),
            "total_spend": total,
            "avg_price": float(prices.mean()) if len(prices) else None,
            "max_price": int(prices.max()) if len(prices) else None,
            "recency_days": rec,
            "active_span_days": span,
            "first_seen": first,
            "last_seen": last,
            "preferred_channel": g["channel_id"].mode().iloc[0] if len(g) else None,
            "low_share": float(band.get("low", 0)) / n,
            "mid_share": float(band.get("mid", 0)) / n,
            "premium_share": float(band.get("premium", 0)) / n,
            "avg_duration_days": float(g["duration_days"].dropna().mean()) if g["duration_days"].notna().any() else None,
            "n_cancel": n_cancel,
            "cancel_share": n_cancel / n,
            "churn_ever": int(g["is_churn"].sum() > 0),
            "device_platform": g["platform"].dropna().mode().iloc[0] if g["platform"].notna().any() else None,
            "is_mobile": None,
        })
    feats = pd.DataFrame(rows)
    feats.to_sql("dim_customer_features", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    return feats