# Brief penelitian regresi AHI

Prediksi AHI kontinu dari fitur IMF sensor Flow, PulseRate, dan Pleth. Bandingkan ResNet50, VGG16, dan VGG19 untuk menilai kontribusi transfer learning. Setiap arsitektur memakai satu model untuk seluruh posisi tidur.

Acuan pelaksanaan: [AGENTS.md](AGENTS.md). Rincian metodologi: [desain eksperimen](docs/desain_eksperimen.md) dan [search space](docs/search_space.md).

## Data dan input

- Gunakan hanya `data/Flow Patient.xlsx`, `data/PulseRate.xlsx`, dan `data/Pleth.xlsx`, tanpa SMOTE atau perubahan workbook asli.
- Satu sampel adalah pasangan `(id, Posisi)` dengan tiga sensor lengkap. Pertahankan seluruh ID yang disepakati, target ekstrem, prediksi negatif, dan error besar.
- Gunakan preprocessing serta manifest tersimpan. Split seed 42 menjaga seluruh posisi satu ID dan kelompok fitur identik dalam subset yang sama. Transformasi dipelajari hanya dari train.
- Input berupa pseudo-image tiga sensor berukuran 224 x 224 x 3 dan one-hot posisi Supine, Left, Right. Rincian encoding, scaling, serta normalisasi ada pada [preprocessing](docs/preprocessing.md) dan desain eksperimen.

## Eksperimen

Implementasi memakai PyTorch/TorchVision dengan CUDA native Windows. Semua model memakai pooling global, penggabungan posisi, head 128/64, dan output regresi linear. Bobot pretrained dipatok ke `IMAGENET1K_V1`.

| Skema | Inisialisasi dan parameter yang dilatih |
|---|---|
| Scratch | Backbone acak dan head baru; seluruh bobot dilatih |
| Feature extraction (FE) | Backbone ImageNet beku; hanya head baru dilatih |
| FT langsung | Backbone ImageNet dan head baru; head serta blok akhir dilatih sejak awal, tanpa checkpoint FE |

Setiap kandidat diinisialisasi mandiri. Grid yang sama berlaku untuk sembilan kondisi: learning rate awal `1e-6, 1e-5, 1e-4, 1e-3` dikalikan dropout `0.1, 0.3, 0.5`, menghasilkan 108 kandidat.

Protokol tetap: seed 42, Adam/MSE, batch 8, weight decay 0, maksimal 50 epoch, early stopping patience 10, scheduler ReduceLROnPlateau, dan pemulihan checkpoint terbaik dari validation. AMP/GradScaler aktif pada CUDA.

## Pemilihan dan evaluasi

- Pilih konfigurasi tiap kondisi, lalu CNN utama, dari RMSE validation terkecil; gunakan MAE jika seri. Bekukan konfigurasi dan checkpoint sebelum test, tanpa refit train+validation.
- Bandingkan dengan median training dan Ridge pada sampel/split yang sama. Laporkan MAE, MSE, RMSE, R², dan jumlah sampel secara keseluruhan serta per posisi.
- Test telah terpapar pada PHASE 5. Evaluasi terbaru adalah evaluasi lanjutan; hasil test tidak mengubah pilihan atau grid. Kesimpulan terbatas pada satu split/seed dan keterbatasan data, tanpa klaim validasi klinis.

## Status dan hasil

PHASE 0 sampai PHASE 5, smoke, full grid 108 kandidat, dan evaluasi test lanjutan selesai. CNN utama tetap ResNet50/scratch berdasarkan validation. Hasil terbaru tersedia pada [tabel eksperimen](docs/tabel_hasil_eksperimen.md), [ringkasan perbandingan](docs/perbandingan_model.md), dan [laporan evaluasi tuning](docs/tuning_evaluation_report.md).

| Artefak lokal | Lokasi |
|---|---|
| Dataset olahan, scaler, manifest | `artifacts/processed_data/` |
| Kandidat, konfigurasi, checkpoint, history | `artifacts/tuning/grid_v1/{ResNet50,VGG16,VGG19}/full/` |
| Metrik, prediksi, protokol beku, grafik evaluasi | `artifacts/tuning/grid_v1/evaluation_v1/` |
| EDA dan audit | `artifacts/figures/preprocessing/`, `artifacts/audit/` |

## Pelaksanaan

Notebook 01 menangani preprocessing; 02 sampai 04 memakai runner tuning bersama; 05 membaca hasil evaluasi tanpa training ulang CNN. Untuk membaca hasil saja, nonaktifkan flag eksekusi terkait atau jalankan sel hasil tersimpan. Flag aktif yang tersimpan bukan persetujuan baru untuk mengulang training.

Pertahankan hasil historis dan hash protokol. Data pasien serta artifacts tetap lokal; notebook sumber harus tanpa output/execution count. Laporkan hasil dan pemeriksaan setiap tahap, lalu tunggu persetujuan sebelum tahap berikutnya. Gunakan bahasa Indonesia formal untuk dokumentasi dan bahasa Inggris untuk nama kode.
