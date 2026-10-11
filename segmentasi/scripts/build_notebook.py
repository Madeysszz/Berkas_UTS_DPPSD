import os

import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "segmentasi.ipynb")

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}
cells = []


def M(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def C(code):
    cells.append(nbf.v4.new_code_cell(code))


M("""# Proyek Analisis Segmentasi Pelanggan: Distribusi Norton

**Studi kasus:** memetakan basis pelanggan subscription Norton Indonesia dari data transaksi (cobrand Telkomsel, Digiflazz, retail) menjadi segmen yang actionable.

Dokumen ini memenuhi 4 kompetensi mata kuliah: (1) konsep analisis segmentasi dan identifikasi proses bisnis; (2) pemilihan metode dan pendekatan; (3) penggunaan bahasa dan tools; (4) perancangan workflow segmentasi.""")

M("""## Bab 1. Konsep Dasar Analisis Segmentasi Pelanggan

### 1.1 Definisi dan tipe segmentasi

**Segmentasi pelanggan** adalah proses membagi pasar atau basis pelanggan menjadi kelompok yang relatif homogen berdasarkan karakteristik tertentu, agar strategi pemasaran dapat diarahkan per kelompok. Empat tipe umum:

| Tipe | Basis | Bisa diterapkan pada data ini? |
|---|---|---|
| **Demografi** | usia, gender, pendapatan, pendidikan | Tidak. Master pelanggan tidak ada, hanya `customer_id` anonim |
| **Geografis** | kota, provinsi, negara | Tidak. Tidak ada atribut lokasi pelanggan (hanya IP sebagian kecil) |
| **Perilaku** | riwayat transaksi, frekuensi, nilai, churn, perangkat | **Ya**. Kolom lengkap: harga, durasi, status, perangkat, kanal |
| **Psikografis** | gaya hidup, nilai | Sebagian. Hanya proksi lemah lewat preferensi perangkat dan paket |

Keputusan jujur: proyek ini menerapkan **segmentasi berbasis perilaku**, karena itu satu-satunya dimensi yang tersedia dan terbilang kuat di data.

### 1.2 Identifikasi proses bisnis

Dari analisis proyek Basis Data: pelanggan membeli paket Norton melalui kanal (Telkomsel, Digiflazz, retail), melakukan redem, dan melewati lifecycle langganan (mulai, aktif, berakhir, batal). Karakteristik per perilaku yang bisa diukur per pelanggan: jumlah transaksi, total pengeluaran, selang transaksi terakhir (recency), durasi langganan rata-rata, riwayat pembatalan, preferensi harga, kanal, dan perangkat.

### 1.3 Tahapan segmentasi dan pemanfaatannya

Tahapan: (1) tujuan bisnis, (2) pilih basis dan variabel, (3) pengumpulan dan pembersihan data, (4) memberdayakan variabel (feature engineering), (5) segmentasi (clustering), (6) profiling dan validasi, (7) evaluasi dan strategi. Pemanfaatan segmen: targeting, positioning, personalisasi promosi, strategi retensi dan upsell.""")

M("""## Bab 2. Metode dan Pendekatan

### 2.1 Kerangka RFM

RFM memotret nilai pelanggan dari tiga sudut:

| Komponen | Pertanyaan | Variabel di sini |
|---|---|---|
| Recency (R) | Berapa lama sejak transaksi terakhir? | `recency_days` |
| Frequency (F) | Seberapa sering bertransaksi? | `n_valid` (order yang valid) |
| Monetary (M) | Berapa total pengeluaran? | `total_spend` (harga order valid) |

RFM diskalakan ke skor 1-5 tiap komponen (kuintil), dan skor gabungan `rfm_score = R*100 + F*10 + M`.

### 2.2 Clustering K-Means

**Mengapa K-Means:** fitur perilaku numerik kontinu, jumlah klaster tidak diketahui sebelumnya, interpretasi lewat pusat klaster mudah, dan sudah menjadi standar untuk segmentasi RFM. Perbandingan dengan pendekatan rule-based (tiering nilai manual) disajikan sebagai alternatif; K-Means dipilih karena menemukan pola kombinasi fitur (misal nilai rendah tapi frekuensi tinggi) yang luput dari tiering tunggal.

**Jumlah klaster:** dievaluasi lewat kurva elbow (inertia) dan silhouette. Hasil nyata: silhouette maksimum di **k=2** yang hanya memisahkan segmen premium kecil (146 vs 9.178), tidak actionable. Keputusan metode: dipilih **k=4** dengan alasan diameter: kurva elbow melandai setelah k=4-5 dan silhouette k=4 (0,597) hampir setinggi k=5 (0,600), empat segmen lebih mudah diinterpretasi untuk strategi. Keputusan ini dicatat transparan, bukan menyembunyikan bahwa k=2 adalah optimum statistik.

### 2.3 Fitur yang dipakai (ternormalisasi)

`recency_days, n_valid, total_spend, avg_price, cancel_share, avg_duration_days` — diskalakan dengan StandardScaler agar bobot antarfitur setara.""")

C("""import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.abspath("."), "scripts"))
import etl_features as fx
import clustering as cl
import charts as ch

BASE = os.path.abspath(".")
DB = os.path.join(BASE, "segmentasi.db")
DATA = os.path.join(BASE, "data")
SCHEMA = os.path.join(BASE, "sql", "segment_schema.sql")
OUT = os.path.join(BASE, "assets")""")

M("""### 2.4 Feature engineering per pelanggan

Input: 9.326 pelanggan dari log transaksi (14.261 baris, baris uji dikeluarkan).""")

C("""feats = fx.build_features(DB, DATA, SCHEMA)
print("jumlah pelanggan:", len(feats))
feats[["customer_id", "n_valid", "total_spend", "recency_days",
       "cancel_share", "avg_duration_days"]].head(8).to_string(index=False)""")

C("""feats[["n_valid", "total_spend", "recency_days", "avg_price",
       "cancel_share", "avg_duration_days"]].describe().round(2).to_string()""")

M("""## Bab 3. Implementasi

### 3.1 RFM scoring""")

C("""rfm = cl.compute_rfm(feats)
rfm.describe().round(2).to_string()""")

C("""print("Distribusi skor RFM gabungan (rfm_score):")
rfm["rfm_score"].value_counts().sort_index().to_string()""")

M("""### 3.2 Evaluasi jumlah klaster (elbow + silhouette)""")

C("""_, X = cl.prepare_matrix(feats)
m = cl.elbow_and_silhouette(X, kmax=9)
print("Keputusan: k=4 (alasan pada Bab 2.2; k=2 optimum statistik tapi tidak actionable)")
m.to_string(index=False)""")

C("""K = 4
res = cl.run_clustering(feats, K)
labels = res["labels"]
print("PCA 2 komponen: varians tercakup", [round(v, 3) for v in res["explained"]])
print("banyak pelanggan per klaster:")
pd.Series(labels).value_counts().sort_index().to_string()""")

M("""### 3.3 Profil segmen hasil clustering""")

C("""prof = cl.profile_clusters(feats, labels, K)
prof.to_string(index=False)""")

C("""cl.save_results(DB, feats, labels, prof, rfm)
print("hasil segmen disimpan ke segmentasi.db (seg_cluster, seg_assignment)")""")

M("""### 3.4 Interpretasi segmen (penamaan)

| Segmen | Deskripsi dari angka |
|---|---|
| **0. Pembeli Baru Aktif** | 4.954 pelanggan, 1 transaksi bersih, tanpa riwayat batal (churn 0%), harga menengah, durasi 30 hari |
| **1. Pembeli Rentan Churn** | 4.228 pelanggan, frekuensi 2x, setengah riwayatnya CANCEL (54%), nilai Rp 28,6 rb |
| **2. Pembeli Premium** | 140 pelanggan, spend rata-rata Rp 166 rb, 98,6% kelas premium, durasi 348 hari |
| **3. Pelaku Berulang Ekstrem** | 2 pelanggan, 60-65 transaksi, kanal Digiflazz, windows; pola reseller/bulk, perlu kontrol anomali |

Rekon numerik: 4.954 + 4.228 + 140 + 2 = **9.324** pelanggan tersegmen.""")

C("""from IPython.display import Image
for name in ["elbow.png", "silhouette.png", "pca.png", "cluster_size.png",
             "cluster_profile.png", "rfm_scatter.png"]:
    print("###", name)
    display(Image(filename=os.path.join(OUT, name), width=620))""")

M("""## Bab 4. Workflow Analisis Segmentasi Pelanggan

```mermaid
flowchart TD
    A["1. Tujuan bisnis: target, retensi, upsell"] --> B["2. Identifikasi basis dan variabel"]
    B --> C["3. Pembersihan data transaksi"]
    C --> D["4. Feature engineering per pelanggan (RFM + perilaku)"]
    D --> E["5. Normalisasi fitur"]
    E --> F["6. Evaluasi jumlah klaster (elbow, silhouette)"]
    F --> G["7. Clustering K-Means k=4"]
    G --> H["8. Profiling dan penamaan segmen"]
    H --> I["9. Validasi dan evaluasi"]
    I --> J["10. Rekomendasi strategi per segmen"]
    I -- "profil ambigu" --> E
```

**Checkpoint:**

- Setelah 3: 14.261 baris masuk, baris TESTLOCAL dikeluarkan.
- Setelah 4: 9.324 pelanggan dengan RFM dan fitur perilaku.
- Setelah 6: tabel elbow+silhouette tercatat, keputusan k=4 berdasar alasan.
- Setelah 8: jumlah segmen tervalidasi (jumlah pelanggan = 9.324).

## Bab 5. Kesimpulan, Rekomendasi, dan Keterbatasan

### Rekomendasi bisnis per segmen

1. **Pembeli Baru Aktif (4.954)**: dorong loyalitas dan perpanjangan; tawarkan paket durasi 60/90 hari dan cross-sell Norton 360; segmen belum pernah churn sehingga biaya akuisisi sudah terbayar.
2. **Pembeli Rentan Churn (4.228)**: prioritas retensi, pengingat kedaluwarsa, program perpanjangan dini, survei kepuasan; menahan 1% churn segmen ini nilainya besar karena frekuensi 2x.
3. **Pembeli Premium (140)**: layanan prioritas, paket keluarga/multi-device, jaga experience agar tidak pindah; nilai sangat tinggi per pelanggan.
4. **Pelaku Berulang Ekstrem (2)**: validasi akun dan kontrol anti-fraud (pola bulk di Digiflazz).

### Keterbatasan (disampaikan jujur)

1. Tidak ada data demografi dan geografis, sehingga segmen hanya berbasis perilaku.
2. Rentang waktu data terbatas (kebanyakan 2026), recency relatif terhadap akhir data, bukan waktu nyata.
3. `device_data` hanya terekam pada sebagian baris (1.628), dominasi Android bisa bias.
4. Dua pelanggan di segmen 3 berpotensi anomali/otomatisasi, bukan pelanggan individu.

### Isi folder

```
segmentasi/
├── segmentasi.ipynb          dokumen ini
├── segmentasi.db             fitur pelanggan + hasil segmen
├── sql/segment_schema.sql
├── scripts/                  etl_features, clustering, charts, run_segment
├── data/                     salinan CSV
├── assets/*.png              grafik
└── laporan/segmentasi.md     ringkasan
```""")

nb["cells"] = cells
nbf.write(nb, OUT)
print("notebook ditulis:", OUT, "| sel:", len(cells))