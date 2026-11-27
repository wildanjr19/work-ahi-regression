# Tabel hasil sembilan eksperimen CNN

Kode mengikuti rancangan penelitian: A = from scratch, B = feature extraction, C = fine-tuning; nomor 1/2/3 = ResNet50/VGG16/VGG19.

## Hasil pada data test

Setiap sel memuat kode eksperimen dan nilai metrik, dibulatkan menjadi empat angka desimal. Seluruh kondisi memakai 179 sampel test yang sama. MAE, MSE, dan RMSE lebih rendah menunjukkan error lebih kecil; R² lebih tinggi menunjukkan hasil lebih baik.

Cetak tebal menandai skema dengan nilai metrik terbaik pada setiap baris arsitektur dalam masing-masing tabel, termasuk tabel validation.

### RMSE

| Arsitektur | From scratch (A) | Feature extraction (B) | Fine-tuning (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 29,0473** | B1: 29,9371 | C1: 29,7947 |
| VGG16 | A2: 30,6702 | B2: 30,7553 | **C2: 29,3305** |
| VGG19 | **A3: 29,7234** | B3: 29,9358 | C3: 30,1204 |

### MAE

| Arsitektur | From scratch (A) | Feature extraction (B) | Fine-tuning (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 17,9986** | B1: 21,1154 | C1: 19,1027 |
| VGG16 | A2: 20,5500 | B2: 20,3990 | **C2: 19,9531** |
| VGG19 | A3: 20,9089 | B3: 20,6683 | **C3: 19,9233** |

### MSE

| Arsitektur | From scratch (A) | Feature extraction (B) | Fine-tuning (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 843,7468** | B1: 896,2322 | C1: 887,7238 |
| VGG16 | A2: 940,6605 | B2: 945,8905 | **C2: 860,2767** |
| VGG19 | **A3: 883,4782** | B3: 896,1513 | C3: 907,2405 |

### R²

| Arsitektur | From scratch (A) | Feature extraction (B) | Fine-tuning (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 0,1059** | B1: 0,0502 | C1: 0,0593 |
| VGG16 | A2: 0,0032 | B2: -0,0024 | **C2: 0,0883** |
| VGG19 | **A3: 0,0638** | B3: 0,0503 | C3: 0,0386 |

## Dasar pemilihan model

RMSE validation berikut dihitung pada 157 sampel yang sama untuk seluruh kondisi.

| Arsitektur | From scratch (A) | Feature extraction (B) | Fine-tuning (C) |
|---|---:|---:|---:|
| ResNet50 | **A1: 18,8352** | B1: 21,9343 | C1: 20,9724 |
| VGG16 | A2: 22,6913 | B2: 22,4419 | **C2: 21,7499** |
| VGG19 | A3: 21,6899 | B3: 22,3300 | **C3: 21,1850** |

A1 terpilih berdasarkan RMSE validation sebelum test dibaca. Pilihan tidak diubah berdasarkan test. Satu prediksi test negatif pada C3 tetap disertakan; tidak ada clipping atau penghapusan kasus buruk.

Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan. Hasil berasal dari satu split dan seed; belum ada CI atau uji signifikansi. Baseline median/Ridge tersedia pada [laporan PHASE 5](phase_5_report.md); pembahasan tiga arsitektur ada pada [ringkasan perbandingan](perbandingan_model.md).

Sumber angka: [hasil test](../artifacts/comparison/comparison_test.csv) dan [hasil validation](../artifacts/comparison/comparison_validation.csv) lokal.
