# Segmentasi Perilaku Pelanggan Norton Indonesia

Proyek segmentasi akun (`customer_id`) dari log transaksi Norton memakai fitur RFM dan K-Means.
Status saat ini: **baseline eksploratif** (K-Means k=4), belum tervalidasi. Detail lengkap ada di laporan UTS.

## Struktur proyek

```
basis-data/
  data/                  # 3 CSV sumber (read-only): package, retail log, cobrand log
  basis_data.ipynb       # eksplorasi + ETL ke database
  scripts/etl.py         # gabung log, turunkan kanal/status, muat ke DB
  norton_bisnis.db       # transaction_log, fact_transaction, pricing, dll.
segmentasi/
  segmentasi.ipynb       # fitur per akun, pemilihan k, profil klaster
  scripts/
    etl_features.py      # bentuk fitur per customer_id
    clustering.py        # scaling + K-Means + silhouette
    run_segment.py       # jalankan seluruh alur segmentasi
  segmentasi.db          # fitur + assignment klaster
```

## Workflow

```
CSV mentah ──► ETL ──► norton_bisnis.db ──► fitur per akun ──► K-Means ──► profil segmen ──► hipotesis & eksperimen
 (3 file)     (1)          (2)                  (3)              (4)           (5)                 (6)
```

1. **ETL** (`basis-data/scripts/etl.py`): normalisasi header, gabung retail + cobrand, parse tanggal/harga, turunkan kanal dari awalan `bill_number`.
2. **Database**: simpan `transaction_log` (14.261 baris) dan `fact_transaction`; baris `TESTLOCAL` dikeluarkan.
3. **Fitur** (`etl_features.py`): satu baris per `customer_id` (9.324 profil) berisi `recency_days`, `n_valid`, `total_spend`, `avg_price`, `cancel_share`, `avg_duration_days`.
4. **Clustering** (`clustering.py`): `StandardScaler` + K-Means (`random_state=7`, `n_init=10`), uji k=2–9 dengan inertia dan silhouette.
5. **Profil segmen**: ringkas rata-rata per klaster dan beri nama deskriptif yang netral.
6. **Tindak lanjut**: ubah profil menjadi hipotesis, lalu uji lewat eksperimen dengan kelompok kontrol.

## Cara menjalankan ulang

1. Taruh CSV sumber di `basis-data/data/` (jangan diubah) dan catat hash-nya.
2. Jalankan ETL, lalu `segmentasi/scripts/run_segment.py` (atau buka notebook berurutan).
3. Catat tanggal acuan, versi library, dan seed setiap kali menjalankan.

## Yang masih harus dilakukan

- [ ] Sepakati definisi transaksi valid, makna `ACTIVE`/`CANCEL`, tanggal acuan, dan zona waktu
- [ ] Cek duplikasi dan bangun pemetaan katalog paket ke transaksi
- [ ] Perbaiki fitur: kurangi redundansi `total_spend`/`n_valid`/`avg_price`, imputasi missing selain nol
- [ ] Uji stabilitas (multi-seed, bootstrap) dan alternatif (GMM, HDBSCAN, k-prototypes)
- [ ] Rancang eksperimen retensi/penawaran dengan kelompok kontrol

## Catatan data

Data bersifat internal dan mengandung identifier serta kredensial (email, IP, kode penukaran, PIN).
Jangan commit CSV/DB mentah ke repositori publik dan jangan bagikan `customer_id` mentah.
`CANCEL` bukan sama dengan churn.
