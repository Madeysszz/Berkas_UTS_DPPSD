PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_customer_features (
    customer_id      TEXT PRIMARY KEY,
    n_rows           INTEGER NOT NULL,
    n_valid          INTEGER NOT NULL,
    total_spend      INTEGER NOT NULL,
    avg_price        REAL,
    max_price        INTEGER,
    recency_days     INTEGER,
    active_span_days INTEGER,
    first_seen       TEXT,
    last_seen        TEXT,
    preferred_channel TEXT,
    low_share        REAL,
    mid_share        REAL,
    premium_share    REAL,
    avg_duration_days REAL,
    n_cancel         INTEGER,
    cancel_share     REAL,
    churn_ever       INTEGER NOT NULL DEFAULT 0,
    device_platform  TEXT,
    is_mobile        INTEGER
);

CREATE TABLE IF NOT EXISTS seg_cluster (
    cluster_id  INTEGER PRIMARY KEY,
    label       TEXT,
    n_customers INTEGER,
    mean_recency_days REAL,
    mean_spend   REAL,
    mean_freq    REAL,
    cancel_share REAL,
    interpretation TEXT
);

CREATE TABLE IF NOT EXISTS seg_assignment (
    customer_id TEXT PRIMARY KEY,
    cluster_id  INTEGER REFERENCES seg_cluster(cluster_id),
    r_score     INTEGER,
    f_score     INTEGER,
    m_score     INTEGER,
    rfm_score   INTEGER
);