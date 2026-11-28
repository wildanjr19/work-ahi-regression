# Tabel hasil tuning sembilan kondisi CNN

Hasil terbaru setelah full grid dan evaluasi test lanjutan pada 27 November 2026. Setiap kondisi memakai satu konfigurasi terbaik dari 12 kandidat berdasarkan RMSE validation, lalu MAE jika seri. Total pencarian: 108 kandidat. Semua kandidat dilatih secara mandiri; FT langsung memakai ImageNet dan head baru tanpa checkpoint FE.

Kode: A = scratch, B = feature extraction, C = FT langsung; nomor 1/2/3 = ResNet50/VGG16/VGG19. Tabel memakai hasil keseluruhan, dibulatkan empat desimal. MAE, MSE, dan RMSE lebih kecil lebih baik; R² lebih besar lebih baik.

Cetak tebal menandai skema dengan nilai metrik terbaik pada setiap baris arsitektur. Highlight test bersifat deskriptif; pemilihan model tetap berdasarkan RMSE validation sebelum test dibaca.

## Hasil validation

Seluruh kondisi memakai 157 sampel validation yang sama.

### RMSE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 18,1728** | B1: 21,7359 | C1: 18,6765 |
| VGG16 | A2: 20,1821 | B2: 22,4419 | **C2: 19,2202** |
| VGG19 | A3: 20,6907 | B3: 22,3300 | **C3: 19,3368** |

### MAE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 11,7013** | B1: 15,8232 | C1: 12,8360 |
| VGG16 | A2: 14,8720 | B2: 15,6031 | **C2: 13,0266** |
| VGG19 | A3: 15,6711 | B3: 15,2581 | **C3: 13,8110** |

### MSE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 330,2514** | B1: 472,4507 | C1: 348,8126 |
| VGG16 | A2: 407,3161 | B2: 503,6378 | **C2: 369,4148** |
| VGG19 | A3: 428,1046 | B3: 498,6296 | **C3: 373,9116** |

### R²

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 0,5009** | B1: 0,2859 | C1: 0,4728 |
| VGG16 | A2: 0,3844 | B2: 0,2388 | **C2: 0,4417** |
| VGG19 | A3: 0,3530 | B3: 0,2464 | **C3: 0,4349** |

## Hasil test

Seluruh kondisi memakai 179 sampel test yang sama. Test telah terpapar pada PHASE 5; hasil ini merupakan evaluasi lanjutan.

### RMSE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | A1: 29,4082 | B1: 30,4248 | **C1: 27,2430** |
| VGG16 | A2: 29,4735 | B2: 30,7553 | **C2: 28,8655** |
| VGG19 | A3: 30,2843 | B3: 29,9358 | **C3: 28,3112** |

### MAE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | A1: 17,8816 | B1: 21,2716 | **C1: 16,8037** |
| VGG16 | A2: 19,7819 | B2: 20,3990 | **C2: 18,4123** |
| VGG19 | A3: 20,7198 | B3: 20,6683 | **C3: 18,7888** |

### MSE

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | A1: 864,8428 | B1: 925,6658 | **C1: 742,1816** |
| VGG16 | A2: 868,6875 | B2: 945,8905 | **C2: 833,2189** |
| VGG19 | A3: 917,1390 | B3: 896,1513 | **C3: 801,5255** |

### R²

| Arsitektur | Scratch (A) | Feature extraction (B) | FT langsung (C) |
|---|---:|---:|---:|
| ResNet50 | A1: 0,0835 | B1: 0,0190 | **C1: 0,2135** |
| VGG16 | A2: 0,0794 | B2: -0,0024 | **C2: 0,1170** |
| VGG19 | A3: 0,0281 | B3: 0,0503 | **C3: 0,1506** |

## Konfigurasi terpilih dari validation

| Kode | Arsitektur | Skema | Learning rate awal | Dropout |
|---|---|---|---:|---:|
| A1 | ResNet50 | Scratch | `1e-04` | 0,1 |
| B1 | ResNet50 | Feature extraction | `1e-03` | 0,5 |
| C1 | ResNet50 | FT langsung | `1e-03` | 0,3 |
| A2 | VGG16 | Scratch | `1e-05` | 0,1 |
| B2 | VGG16 | Feature extraction | `1e-03` | 0,3 |
| C2 | VGG16 | FT langsung | `1e-03` | 0,5 |
| A3 | VGG19 | Scratch | `1e-05` | 0,1 |
| B3 | VGG19 | Feature extraction | `1e-03` | 0,3 |
| C3 | VGG19 | FT langsung | `1e-04` | 0,1 |

**A1 (ResNet50/Scratch) tetap menjadi CNN utama**, dengan RMSE validation 18,1728. Pilihan dan checkpoint dibekukan sebelum evaluasi test terbaru. ResNet50/FT langsung memperoleh RMSE test terendah (27,2430), tetapi hasil tersebut tidak mengubah pilihan utama.

## Baseline pembanding

| Subset | Model | n | MAE | MSE | RMSE | R² | Prediksi negatif |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | DummyMedian | 157 | 17,1430 | 735,8660 | 27,1269 | -0,1122 | 0 |
| Validation | Ridge | 157 | 54,4057 | 210845,6973 | 459,1794 | -317,6722 | 12 |
| Test | DummyMedian | 179 | 22,6362 | 1158,8185 | 34,0414 | -0,2280 | 0 |
| Test | Ridge | 179 | 21,2306 | 1190,3152 | 34,5009 | -0,2614 | 9 |

Prediksi negatif pada sembilan kondisi CNN terpilih: 0 validation dan 0 test. Jumlah ini berasal dari hasil penuh, bukan smoke. Prediksi negatif Ridge, target ekstrem, dan semua error besar tetap dipertahankan tanpa clipping. Baseline terpilih dari validation adalah DummyMedian; Ridge memakai alpha 1000 dan StandardScaler yang hanya fit pada train.

## Sumber dan riwayat

- Angka terbaru: [validation](../artifacts/tuning/grid_v1/evaluation_v1/comparison_validation.csv), [test](../artifacts/tuning/grid_v1/evaluation_v1/comparison_test.csv), dan [pilihan beku](../artifacts/tuning/grid_v1/evaluation_v1/selection_validation.json), tersedia lokal.
- Metrik per posisi seluruh kondisi: [validation](../artifacts/tuning/grid_v1/evaluation_v1/metrics_validation.csv) dan [test](../artifacts/tuning/grid_v1/evaluation_v1/metrics_test.csv), tersedia lokal.
- Penjelasan hasil terbaru: [ringkasan perbandingan](perbandingan_model.md) dan [laporan evaluasi tuning](tuning_evaluation_report.md).
- Hasil sebelum tuning tetap tersedia di artifacts/comparison/ serta [laporan PHASE 2](phase_2_report.md), [PHASE 3](phase_3_report.md), [PHASE 4](phase_4_report.md), dan [PHASE 5](phase_5_report.md). FT historis berasal dari checkpoint FE; tabel di halaman ini memakai FT langsung.

Seluruh 411 ID tetap tercakup pada split tersimpan. Kesimpulan terbatas pada kandidat yang diuji, satu split/seed, dan keterbatasan data; tidak ada CI atau uji signifikansi. Evaluasi ini belum membuktikan generalisasi eksternal atau validasi klinis.

Antislop R-17/R-36 PASS: angka dan highlight berasal dari CSV aktual; pemilihan validation dan hasil historis dipertahankan. R-02 PASS: tanpa em/en dash atau informasi pasien individual; aturan UI tidak berlaku.
