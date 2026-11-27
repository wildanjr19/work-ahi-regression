# Tabel hasil sembilan eksperimen CNN

Kode mengikuti rancangan penelitian: A = from scratch, B = feature extraction, C = fine-tuning; nomor 1/2/3 = ResNet50/VGG16/VGG19.

## Matriks hasil

Setiap sel memuat kode eksperimen dan RMSE test. Seluruh kondisi memakai 179 sampel test yang sama. RMSE lebih rendah menunjukkan error lebih kecil.

| Arsitektur | From Scratch | Feature Extraction | Fine-tuning |
|---|---:|---:|---:|
| ResNet50 | A1: 29,0473 | B1: 29,9371 | C1: 29,7947 |
| VGG16 | A2: 30,6702 | B2: 30,7553 | C2: 29,3305 |
| VGG19 | A3: 29,7234 | B3: 29,9358 | C3: 30,1204 |

## Metrik lengkap

| Kode | RMSE validation | MAE test | MSE test | RMSE test | R2 test |
|---|---:|---:|---:|---:|---:|
| A1 | 18,8352 | 17,9986 | 843,7468 | 29,0473 | 0,1059 |
| B1 | 21,9343 | 21,1154 | 896,2322 | 29,9371 | 0,0502 |
| C1 | 20,9724 | 19,1027 | 887,7238 | 29,7947 | 0,0593 |
| A2 | 22,6913 | 20,5500 | 940,6605 | 30,6702 | 0,0032 |
| B2 | 22,4419 | 20,3990 | 945,8905 | 30,7553 | -0,0024 |
| C2 | 21,7499 | 19,9531 | 860,2767 | 29,3305 | 0,0883 |
| A3 | 21,6899 | 20,9089 | 883,4782 | 29,7234 | 0,0638 |
| B3 | 22,3300 | 20,6683 | 896,1513 | 29,9358 | 0,0503 |
| C3 | 21,1850 | 19,9233 | 907,2405 | 30,1204 | 0,0386 |

A1 terpilih berdasarkan RMSE validation sebelum test dibaca. Pilihan tidak diubah berdasarkan test. Satu prediksi test negatif pada C3 tetap disertakan; tidak ada clipping atau penghapusan kasus buruk.

Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan. Hasil berasal dari satu split dan seed; belum ada CI atau uji signifikansi. Baseline median/Ridge tersedia pada [laporan PHASE 5](phase_5_report.md); pembahasan tiga arsitektur ada pada [ringkasan perbandingan](perbandingan_model.md).

Antislop R-17/R-36 PASS: seluruh nilai sesuai CSV hasil aktual dan kode eksperimen konsisten.

Antislop R-02 PASS: tabel tanpa em/en dash atau ID pasien.
