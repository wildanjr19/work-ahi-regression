# Evaluasi test lanjutan hasil tuning

Selesai pada 27 November 2026 setelah persetujuan pengguna. Sembilan kondisi CNN terpilih dari full grid dan dua baseline dievaluasi pada test setelah pilihan, konfigurasi, checkpoint, data, serta kode dibekukan. Tidak ada training ulang CNN atau refit train+validation.

Test telah terpapar pada PHASE 5. Evaluasi ini merupakan evaluasi lanjutan pada test yang sama, bukan evaluasi independen baru. Hasil historis dan seluruh hasil grid dipertahankan.

## Pemilihan sebelum test

CNN utama: **ResNet50/scratch**, LR awal 1e-04, dropout 0.1; RMSE validation 18.1728, MAE 11.7013. Aturan tetap RMSE validation terkecil, lalu MAE jika seri. Baseline terpilih dari validation: DummyMedian.

Test memuat 179 sampel, 65 ID, dan 57 komponen. Posisi: Supine 65, Left 58, Right 56.

## Hasil keseluruhan

| Model | Strategi | RMSE validation | n test | MAE test | MSE test | RMSE test | R2 test | Prediksi negatif test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | Scratch | 18.1728 | 179 | 17.8816 | 864.8428 | 29.4082 | 0.0835 | 0 |
| ResNet50 | FE | 21.7359 | 179 | 21.2716 | 925.6658 | 30.4248 | 0.0190 | 0 |
| ResNet50 | FT langsung | 18.6765 | 179 | 16.8037 | 742.1816 | 27.2430 | 0.2135 | 0 |
| VGG16 | Scratch | 20.1821 | 179 | 19.7819 | 868.6875 | 29.4735 | 0.0794 | 0 |
| VGG16 | FE | 22.4419 | 179 | 20.3990 | 945.8905 | 30.7553 | -0.0024 | 0 |
| VGG16 | FT langsung | 19.2202 | 179 | 18.4123 | 833.2189 | 28.8655 | 0.1170 | 0 |
| VGG19 | Scratch | 20.6907 | 179 | 20.7198 | 917.1390 | 30.2843 | 0.0281 | 0 |
| VGG19 | FE | 22.3300 | 179 | 20.6683 | 896.1513 | 29.9358 | 0.0503 | 0 |
| VGG19 | FT langsung | 19.3368 | 179 | 18.7888 | 801.5255 | 28.3112 | 0.1506 | 0 |
| DummyMedian | Baseline | 27.1269 | 179 | 22.6362 | 1158.8185 | 34.0414 | -0.2280 | 0 |
| Ridge | Baseline | 459.1794 | 179 | 21.2306 | 1190.3152 | 34.5009 | -0.2614 | 9 |

ResNet50/FT langsung memperoleh RMSE test lebih rendah daripada CNN utama, tetapi hasil test tidak mengubah pilihan ResNet50/scratch. Sembilan kondisi hanya memakai konfigurasi terbaik masing-masing dari validation; seluruh 108 kandidat tidak dicoba pada test.

## CNN utama per posisi

| Posisi | n | MAE | MSE | RMSE | R2 | Prediksi negatif |
|---|---:|---:|---:|---:|---:|---:|
| Supine | 65 | 19.8182 | 688.0785 | 26.2313 | -0.1297 | 0 |
| Left | 58 | 18.2484 | 1169.5222 | 34.1983 | 0.0959 | 0 |
| Right | 56 | 15.2538 | 754.4547 | 27.4673 | 0.0838 | 0 |

MAE, MSE, RMSE, R2, dan jumlah sampel seluruh kondisi per posisi tersedia dalam metrics_validation.csv, metrics_test.csv, serta notebook laporan lokal. R2 subset yang tidak memadai tetap dinyatakan tidak terdefinisi. R2 Supine CNN utama negatif dan R2 keseluruhan rendah; hasil tidak mendukung klaim validasi klinis.

## Biaya kandidat terpilih

| Model | Strategi | Epoch selesai | Epoch terbaik | Parameter trainable | Detik training | Skipped AMP steps |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | Scratch | 21 | 11 | 23779009 | 74.18 | 6 |
| ResNet50 | FE | 47 | 37 | 270977 | 60.38 | 5 |
| ResNet50 | FT langsung | 19 | 9 | 15213185 | 34.75 | 6 |
| VGG16 | Scratch | 50 | 50 | 14789057 | 232.98 | 5 |
| VGG16 | FE | 35 | 25 | 74369 | 64.90 | 4 |
| VGG16 | FT langsung | 15 | 5 | 7153793 | 32.27 | 7 |
| VGG19 | Scratch | 46 | 36 | 20098753 | 258.43 | 7 |
| VGG19 | FE | 48 | 38 | 74369 | 101.98 | 4 |
| VGG19 | FT langsung | 19 | 9 | 9513601 | 49.54 | 6 |

Waktu berasal dari kandidat terpilih yang sudah dilatih; FT langsung tidak menambahkan biaya FE. Biaya seluruh pencarian tercatat pada [laporan full grid](tuning_full_report.md). Baseline dipasang ulang hanya pada train dengan fitur, split, scaler, dan grid alpha yang sama. Ridge memilih alpha 1000 dari validation; StandardScaler hanya fit pada train. Median training tetap dipakai tanpa clipping.

## Pemeriksaan aktual dan perubahan

- Baru: src/tuning_evaluation.py dan tests/test_tuning_evaluation.py. Collector membaca hasil tuning tanpa menulis ulang grid, memeriksa seluruh konfigurasi dan pilihan per kondisi, metadata inisialisasi mandiri, fingerprint preprocessing, prediksi validation, serta hash checkpoint.
- Diperbarui: src/comparison.py memakai alur pembekuan/evaluasi bersama dengan dukungan tuning; src/comparison_plots.py membedakan biaya FT langsung dan historis. Hash pada protokol lama tidak diubah.
- Notebook 05 terhubung ke hasil tuning, tetap RUN_FINAL_EVALUATION=False, tanpa output/execution count. Tiga notebook training dan flag pengguna dipertahankan.
- Unit test: 53 lulus; pip check bersih. Tes baru mencakup penolakan smoke, fingerprint berbeda, checkpoint rusak, pilihan yang tidak sesuai validation, akses tanpa persetujuan, output yang sudah ada, dan pembekuan sebelum test.
- Preflight aktual memuat sembilan checkpoint dengan weights_only=True dan hanya membaca train/validation. Pemeriksaan awal sintetis menemukan tipe path string pada pemeriksaan hash; diperbaiki menjadi Path sebelum data test diakses.
- Eksekusi aktual dari kernel baru: test.npz dimuat 1 kali setelah pembekuan; 18 pembacaan checkpoint memakai weights_only=True; 0 panggilan training CNN.
- QA independen dari CSV: 88 baris metrik cocok; 3696 baris prediksi selaras dengan manifest/target. QA tidak memuat ulang array test.
- Sebanyak 483 berkas data/preprocessing/hasil historis aktif tidak berubah; tiga workbook termasuk pemeriksaan SHA-256 sebelum/sesudah. Verifier snapshot PASS: 571 berkas, 82 aset beku. Tidak ada inferensi dalam verifier.
- Tersimpan 75 grafik 300 DPI. Heatmap test dan Bland-Altman CNN utama diperiksa secara visual; Bland-Altman menyebut pengukuran berulang pasien dan tanpa koreksi/CI.
- Peringatan Windows/ZeroMQ memakai selector thread otomatis dan tidak menghambat notebook. Tidak ada perubahan konfigurasi sistem.

## Penyimpanan dan batasan

Hasil lokal: artifacts/tuning/grid_v1/evaluation_v1/. Protokol/pilihan dibekukan pada frozen_protocol.json dan selection_validation.json sebelum test_access_started.json dibuat. CSV metrik/prediksi, baseline, biaya, kontribusi transfer learning, pembahasan, dan grafik tersedia di direktori yang sama.

Notebook eksekusi aktual: artifacts/tuning/notebook_checks/05_Model_Comparison.tuning.executed.ipynb. Notebook laporan yang dijalankan kembali tanpa membaca array dataset: artifacts/tuning/notebook_checks/05_Model_Comparison.tuning.report.executed.ipynb. Pemeriksaan mesin dan script eksekusi berada di artifacts/tuning/evaluation_checks/. Seluruh artifacts dan informasi individual tetap lokal.

Target ekstrem, prediksi negatif, dan seluruh error dipertahankan. Ridge memiliki 12 prediksi validation dan sembilan prediksi test negatif. Tidak ada penghapusan kasus, perubahan seed/split, clipping, atau perluasan grid berdasarkan test. Kontrak metadata beku tetap mencatat status historisnya; keputusan penggunaan data terbaru dirujuk melalui [keputusan klien](konfirmasi_dataset_klien.md).

Kesimpulan dibatasi satu split/seed, ketergantungan pengukuran berulang/fitur identik, validitas label yang belum dibuktikan, serta test yang sudah terpapar. Tidak ada CI atau uji signifikansi. Tahap ini selesai; pembahasan/publikasi berikutnya memerlukan peninjauan tersendiri dan tidak boleh mengganti pilihan berdasarkan test.

Antislop R-17/R-36 PASS: seluruh angka berasal dari hasil aktual, pemilihan validation dipertahankan, dan tidak ada klaim validasi klinis. R-02 PASS: tanpa em/en dash; aturan UI tidak berlaku.
