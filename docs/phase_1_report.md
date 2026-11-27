# Laporan PHASE 1

Status: preprocessing selesai; seluruh 411 ID disertakan sesuai arahan klien. Pada PHASE 1 belum ada training.

## Hasil aktual

Sebanyak 1.091 sampel dari 411 ID membentuk 380 komponen pasien/salinan. Enam belas pasangan tanpa Pulse Rate dikeluarkan dari eksperimen utama, tanpa mengubah sumber.

| Subset | Sampel | ID | Komponen |
|---|---:|---:|---:|
| Training | 755 | 287 | 265 |
| Validation | 157 | 59 | 58 |
| Testing | 179 | 65 | 57 |

## Keputusan dan validasi

- GroupShuffleSplit, seed 42, rasio awal 70/15/15 pada komponen. Rasio aktual dipengaruhi ukuran kelompok dan pembulatan.
- ID dengan fitur identik tiga sensor tetap satu subset, termasuk hubungan transitif. Tidak ada penghapusan salinan atau averaging AHI.
- Scaler persentil 1/99 hanya fit pada complete-case training. Tidak ada fitur dengan rentang persentil nol. Rata-rata proporsi nilai fitur di luar batas per sensor/subset sekitar 1,65% sampai 2,53%.
- Matriks 50 x 11 x 3 menjadi citra uint8 224 x 224 x 3 melalui nearest-neighbor. Padding 0, fitur nyata 1..255; mask dan fitur float32 disimpan. AHI asli tetap float64, termasuk 600.
- Nol overlap ID, komponen, fingerprint mentah, dan input hasil encoding antar-subset. Semua array finite, selaras dengan manifest, dan identik setelah reload. Hash sumber tetap sama.
- Notebook dijalankan dari kernel baru: 10 sel kode berhasil. Lima belas tes lulus; pip check lulus. Tiga belas visualisasi disimpan pada 300 DPI.

## File utama

- `src/preprocessing.py`, `notebooks/01_Preprocessing.ipynb`, `tests/test_preprocessing.py`.
- Perubahan: `src/audit.py` memakai loader bersama; config, requirements, README, dan AGENTS diperbarui.
- Dataset, manifest, scaler, metadata, audit, laporan clipping, versi runtime, dan QA: `artifacts/processed_data/`.
- Notebook beroutput: `artifacts/notebooks/01_Preprocessing.executed.ipynb`; grafik: `artifacts/figures/preprocessing/`. Keduanya tetap lokal.

## Batasan dan langkah berikutnya

Seluruh 411 ID, termasuk fitur identik dan AHI ekstrem, dipertahankan sesuai arahan klien. Grouping menjaga salinan persis dalam subset yang sama; tidak memvalidasi identitas biologis atau label. Jika sumber dikoreksi, preprocessing harus diregenerasi.

Environment `.venv` memakai paket ilmiah global dan tooling notebook lokal. GPU/TensorFlow belum diuji. PHASE 2 adalah implementasi dan smoke test ResNet50 setelah persetujuan; full training memerlukan persetujuan tersendiri.

Antislop: PASS untuk kode/dokumentasi, angka dari output aktual, tanpa ID pasien dalam laporan; grafik memakai label dan skala eksplisit. Tidak ada UI atau klaim performa model.
