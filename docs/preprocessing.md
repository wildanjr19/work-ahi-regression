# Preprocessing data

Preprocessing menyiapkan fitur tiga sensor menjadi input CNN yang seragam serta memisahkan data untuk pelatihan dan evaluasi. Ringkasan sumber tersedia pada [keadaan data awal](data.md).

## 1. Pemeriksaan dan standardisasi

Tujuan: memastikan struktur data dan target konsisten sebelum pengolahan.

Pipeline merapikan spasi pada header dan metadata, mempertahankan `id` sebagai string, serta memeriksa kunci `(id, Posisi, IMF)`, kecocokan sheet-posisi, nilai numerik finite, dan konsistensi AHI dalam serta antar sensor. Tidak ditemukan masalah yang menghentikan pengolahan. Hash SHA-256 sebelum dan sesudah pembacaan memastikan workbook asli tidak berubah.

## 2. Penggabungan tiga sensor

Tujuan: memastikan setiap sampel memiliki Flow, PulseRate, dan Pleth.

Satu sampel adalah pasangan `(id, Posisi)`. Dari 1.107 pasangan, 1.091 memiliki ketiga sensor. Sebanyak 16 pasangan tanpa PulseRate (9 Left dan 7 Right) dikeluarkan dari eksperimen utama; seluruh 411 ID tetap tercakup. Pipeline tidak memakai SMOTE atau mengubah data sumber.

## 3. Pengelompokan pasien dan fitur identik

Tujuan: mencegah pasien atau salinan fitur tersebar ke subset yang berbeda.

ID yang memiliki fitur ketiga sensor identik pada posisi yang sama dihubungkan menjadi kelompok, termasuk hubungan transitif. Seluruh posisi milik satu ID selalu bersama. Pengelompokan mengabaikan AHI dan menghasilkan 380 komponen dari 411 ID. Identitas pasien tidak digabung dan target tidak dirata-ratakan; aturan ini tidak membuktikan kesamaan identitas biologis.

## 4. Pembagian training, validation, dan test

Tujuan: memisahkan data pembelajaran, pemilihan model, dan evaluasi akhir.

GroupShuffleSplit memakai komponen sebagai kelompok, seed 42, dan rasio awal 70/15/15. Persentase aktual berikut dihitung terhadap total masing-masing kolom.

| Subset | Sampel | ID | Komponen |
|---|---:|---:|---:|
| Training | 755 (69,20%) | 287 (69,83%) | 265 (69,74%) |
| Validation | 157 (14,39%) | 59 (14,36%) | 58 (15,26%) |
| Test | 179 (16,41%) | 65 (15,82%) | 57 (15,00%) |
| Total | 1.091 | 411 | 380 |

Ukuran kelompok dan pembulatan menyebabkan rasio aktual berbeda dari rasio awal. Satu manifest dipakai seluruh arsitektur dan baseline; notebook model tidak membagi ulang data. Validation digunakan untuk pemilihan model; test dievaluasi setelah protokol dibekukan.

## 5. Scaling fitur sensor

Tujuan: menyetarakan skala fitur dan membatasi pengaruh nilai fitur ekstrem.

Batas persentil 1 dan 99 dihitung per fitur, per sensor, hanya dari baris IMF nyata pada training. Nilai dipetakan ke rentang 0 sampai 1; nilai di luar batas di-clipping ke 0 atau 1. Validation dan test memakai parameter training yang sama; padding tidak ikut menentukan scaler.

Tidak ada fitur dengan rentang persentil nol. Rata-rata proporsi nilai fitur yang terkena clipping, dihitung atas 11 fitur pada tiap kombinasi sensor-subset, berkisar 1,65% sampai 2,53%. AHI tidak di-scaling atau di-clipping; target asli termasuk nilai 600 tetap dipertahankan.

## 6. Penyusunan matriks dan mask

Tujuan: membentuk ukuran input seragam sambil membedakan IMF tersedia dan padding.

Setiap sensor disusun menjadi matriks 50 × 11: baris mengikuti indeks IMF numerik, kolom mengikuti urutan 11 fitur yang sama. Slot IMF yang tidak tersedia diisi padding tanpa interpolasi. Mask mencatat keberadaan IMF. Gabungan channel Flow, PulseRate, lalu Pleth menghasilkan matriks 50 × 11 × 3.

## 7. Encoding dan input CNN

Tujuan: menyiapkan representasi sensor dan posisi tidur yang sesuai dengan model.

Fitur hasil scaling dipetakan menjadi piksel uint8 1 sampai 255, sedangkan padding memakai 0. Matriks diperbesar menjadi 224 × 224 × 3 dengan nearest-neighbor agar resize tidak mencampurkan nilai fitur. Representasi ini adalah pseudo-image, bukan citra fisiologis; indeks IMF yang sama antar sensor tidak membuktikan kesamaan frekuensi atau sinkronisasi.

Posisi diberikan melalui input one-hot terpisah dengan urutan Supine, Left, Right: `(1, 0, 0)`, `(0, 1, 0)`, dan `(0, 0, 1)`. ID, posisi, indeks IMF, dan AHI tidak masuk ke 11 fitur sensor.

Piksel disimpan sebelum normalisasi backbone. Di dalam model PyTorch, piksel dibagi 255 tepat sekali, lalu dinormalisasi memakai mean/std bobot TorchVision ImageNet. Tidak ada konversi BGR, normalisasi ganda, atau resize/crop tambahan.

## 8. Penyimpanan dan pemeriksaan akhir

Tujuan: menyediakan input yang dapat digunakan ulang dan diperiksa tanpa mengulang preprocessing.

Hasil tersimpan pada `artifacts/processed_data/`: `train.npz`, `validation.npz`, `test.npz`, beserta manifest, scaler, urutan fitur, metadata, dan audit. NPZ memuat citra, fitur scaled float32, mask, posisi one-hot, target asli float64, dan metadata sampel. Data serta artifacts tetap lokal.

Pemeriksaan tersimpan mencatat nol overlap ID, komponen, fingerprint fitur mentah, dan input hasil encoding antar subset. Seluruh array numerik finite, input-target selaras, serta hasil penyimpanan dan pemuatan ulang identik. Laporan ini merangkum hasil tersimpan tanpa menjalankan ulang preprocessing atau training.

Sumber: [laporan PHASE 1](phase_1_report.md), [implementasi preprocessing](../src/preprocessing.py), [normalisasi model](../src/model_utils.py), dan [konfirmasi dataset klien](konfirmasi_dataset_klien.md).
