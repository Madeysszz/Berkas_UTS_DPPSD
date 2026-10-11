# Laporan Ringkas: Analisis Segmentasi Pelanggan Distribusi Norton

Dokumen penuh ada di **`segmentasi.ipynb`**.

## Konsep dan keputusan metode

- Empat tipe segmentasi (demografi, geografis, perilaku, psikografis) ditinjau terhadap data: yang tersedia hanya **perilaku** (9.324 pelanggan dari `customer_id` anonim); demografi dan geografis tidak ada karena master pelanggan tidak disertakan. Batasan ini dituliskan jujur di laporan.
- Basis segmen: **RFM** (Recency, Frequency, Monetary, skor 1-5 per komponen) + fitur perilaku (harga, durasi, kanal, perangkat, riwayat batal).
- Metode: **K-Means** dengan StandardScaler; alternatif rule-based tiering disajikan sebagai pembanding.
- Jumlah klaster: silhouette maksimum di k=2 (memisahkan segmen premium 146 vs 9.178, tidak actionable). Keputusan tertulis: **k=4** (tekuk elbow, siluet stabil 0,597 vs 0,600 pada k=5, granularitas actionable).

## Hasil segmen (angka nyata)

| Segmen | n | Frek | Spend rata | Harga rata | Share cancel | Durasi |
|---|---|---|---|---|---|---|
| 0. Pembeli Baru Aktif | 4.954 | 1,0 | Rp 13.945 | Rp 13.671 | 0% | 30 hari |
| 1. Pembeli Rentan Churn | 4.228 | 2,1 | Rp 28.617 | Rp 13.430 | 54% | 46 hari |
| 2. Pembeli Premium | 140 | 1,0 | Rp 166.486 | Rp 161.029 | 1% | 348 hari |
| 3. Pelaku Berulang Ekstrem | 2 | 62,5 | Rp 937.500 | Rp 15.000 | 90% | 632 hari |

Total pelanggan tersegmen: 4.954 + 4.228 + 140 + 2 = **9.324**.

## Rekomendasi bisnis

1. **Pembeli Baru Aktif**: dorong loyalitas, tawarkan durasi 60/90 hari dan Norton 360, segmen bebas churn.
2. **Pembeli Rentan Churn**: prioritas retensi, pengingat kedaluwarsa, program perpanjangan, survei; frekuensi 2x membuat nilai churn tinggi.
3. **Pembeli Premium**: layanan prioritas, cross-sell paket keluarga/multi-device, jaga pengalaman.
4. **Pelaku Berulang Ekstrem**: validasi akun dan kontrol anti-fraud (pola bulk di Digiflazz).

## Workflow

Tujuan bisnis → basis & variabel → pembersihan data → feature engineering (RFM) → normalisasi → evaluasi klaster (elbow+silhouette) → K-Means k=4 → profiling & penamaan → validasi (jumlah segmen = jumlah pelanggan) → rekomendasi strategi.

## Keterbatasan

1. Segmentasi hanya perilaku; tanpa demografi/geografis.
2. Rentang data pendek (mayoritas 2026), recency relatif terhadap akhir data.
3. `device_data` hanya 1.628 baris, dominasi Android bisa bias.
4. Segmen 3 (2 pelanggan) berpotensi anomali/otomatisasi.

## Isi folder

```
segmentasi/
├── segmentasi.ipynb
├── segmentasi.db             (dim_customer_features, seg_cluster, seg_assignment)
├── sql/segment_schema.sql
├── scripts/ (etl_features.py, clustering.py, charts.py, run_segment.py, build_notebook.py)
├── data/                     salinan CSV
├── assets/*.png              elbow, silhouette, pca, size, profile, rf_scatter
└── laporan/segmentasi.md
```