# Desain eksperimen regresi AHI dengan hyperparameter tuning

Kesepakatan 27 November 2026. Dokumen ini menjadi acuan eksperimen tuning baru dan penjelasan kepada klien. Konfigurasi/model, runner, integrasi notebook, dan smoke aktual sudah diuji; full grid selesai dengan 108/108 kandidat berhasil. Evaluasi test tuning belum dijalankan. Lihat [laporan full grid](tuning_full_report.md). Lihat [laporan model](tuning_model_report.md), [runner](tuning_runner_report.md), dan [integrasi notebook](tuning_notebook_report.md). Untuk eksperimen baru, FT langsung menggantikan ketentuan FT dari checkpoint FE pada protokol lama; laporan PHASE 0 sampai PHASE 5 tetap menjadi catatan historis.

## 1. Tujuan dan data

Penelitian membandingkan ResNet50, VGG16, dan VGG19 untuk memprediksi AHI kontinu melalui tiga strategi pelatihan. Setiap model menggabungkan seluruh posisi tidur. Konfigurasi dipilih melalui grid search pada validation.

Gunakan hanya `Flow Patient.xlsx`, `PulseRate.xlsx`, dan `Pleth.xlsx`. Satu sampel adalah pasangan `(id, Posisi)` dengan ketiga sensor lengkap. Seluruh 411 ID tetap tercakup; 16 pasangan tanpa PulseRate dikeluarkan. Data dan target asli, termasuk AHI ekstrem, dipertahankan tanpa SMOTE atau perataan target. Rujukan: [keadaan data](data.md) dan [keputusan dataset klien](konfirmasi_dataset_klien.md).

Gunakan kembali preprocessing dan manifest yang terverifikasi: train 755, validation 157, test 179 sampel. Split seed 42 berbasis komponen pasien dan fitur identik menjaga seluruh posisi satu ID serta salinan fitur dalam subset yang sama. Scaler hanya fit pada train; notebook model tidak melakukan split ulang. Rincian tersedia pada [preprocessing](preprocessing.md) dan [laporan PHASE 1](phase_1_report.md).

## 2. Kondisi eksperimen

Ketiga strategi berdiri sendiri pada setiap arsitektur. Setiap kandidat mulai dari inisialisasi baru sesuai strateginya, tanpa melanjutkan bobot kandidat lain.

| Strategi | Inisialisasi backbone | Bobot yang dilatih |
|---|---|---|
| Scratch | Acak | Seluruh backbone dan head baru |
| Feature extraction (FE) | ImageNet `IMAGENET1K_V1` | Hanya head baru; backbone dibekukan |
| Fine-tuning langsung (FT) | ImageNet `IMAGENET1K_V1` | Head baru dan blok akhir backbone sejak awal; blok awal dibekukan |

FT membuka `layer4` ResNet50, block5 VGG16 mulai indeks `features[24]`, atau block5 VGG19 mulai `features[28]`. Batch Normalization ResNet50 tetap beku/eval pada FE dan FT. FT tidak memakai checkpoint FE dan tidak memiliki tahap pelatihan head pendahuluan.

Pola FT langsung menggunakan backbone pretrained dan head baru tersedia dalam [tutorial transfer learning PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Penelitian ini membatasi pembaruan backbone pada blok akhir.

## 3. Input, model, dan pelatihan

Implementasi memakai PyTorch/TorchVision dengan CUDA native Windows. Input terdiri atas pseudo-image tiga channel berurutan Flow, PulseRate, Pleth berukuran 224 x 224 dan one-hot posisi Supine, Left, Right. Di dalam model, piksel dibagi 255 tepat sekali, kemudian dinormalisasi sesuai bobot TorchVision. Tidak ada resize/crop tambahan.

Semua model menggunakan AdaptiveAvgPool2d(1), penggabungan fitur dengan posisi, head Linear 128/64 dengan ReLU dan dropout, serta satu output linear. ID dan AHI tidak menjadi fitur. Output dan loss tetap float32.

Gunakan Adam, MSE, batch 8, weight decay 0, maksimal 50 epoch, early stopping patience 10, dan pemulihan checkpoint terbaik berdasarkan MSE validation. ReduceLROnPlateau tetap sama untuk semua kandidat: faktor 0,5, patience 5, minimum learning rate `1e-7`. AMP/GradScaler aktif pada CUDA; langkah optimizer yang dilewati dicatat. Seed 42 diatur ulang setiap kandidat. Batch size dan batas epoch tetap dapat memengaruhi hasil meskipun tidak dicari.

## 4. Grid search dan pemilihan

| Hyperparameter yang dicari | Kandidat untuk scratch, FE, dan FT | Tujuan |
|---|---|---|
| Learning rate awal | `1e-6`, `1e-5`, `1e-4`, `1e-3` | Menguji skala pembaruan bobot |
| Dropout head | `0.1`, `0.3`, `0.5` | Menguji kekuatan regularisasi; kedua lapisan dropout memakai nilai yang sama |

Grid search menguji seluruh 12 kombinasi per kondisi. Tiga arsitektur dengan tiga strategi menghasilkan sembilan kondisi dan **108 pelatihan kandidat**, di luar smoke test. Parameter tetap beserta alasannya tersedia pada [search space](search_space.md).

Pilih satu konfigurasi per kondisi berdasarkan RMSE validation terkecil, kemudian MAE jika seri. Pilih CNN utama dari sembilan kondisi dengan aturan yang sama. Gunakan train untuk pembelajaran; validation untuk scheduler, early stopping, dan seluruh pemilihan. Bekukan konfigurasi serta checkpoint terpilih sebelum evaluasi test. Jangan melakukan refit train+validation, memperluas grid berdasarkan test, atau memilih model dari hasil test.

## 5. Evaluasi dan penyimpanan

Laporkan MAE, MSE, RMSE, dan R2 keseluruhan serta per posisi, beserta jumlah sampel. R2 pada subset yang tidak memadai dinyatakan tidak terdefinisi. Pertahankan prediksi negatif dan laporkan error besar tanpa clipping. Baseline median training dan Ridge tetap menjadi pembanding pada sampel serta split yang sama.

Setiap kandidat menyimpan konfigurasi, status termasuk kegagalan, seed, checkpoint, history, metrik/prediksi validation, waktu, jumlah parameter trainable, versi library, hardware, dan hash data/kode. Konfigurasi sama tidak menjamin waktu, epoch aktual, atau jumlah parameter sama; biaya aktual wajib dilaporkan.

Pelaksanaan tetap melalui `02_ResNet50.ipynb`, `03_VGG16.ipynb`, dan `04_VGG19.ipynb`, dengan logika tuning bersama di `src/`. Notebook `05_Model_Comparison.ipynb` membaca hasil terpilih tanpa training ulang CNN. Default API tuning nonaktif. Flag notebook saat persiapan push dipertahankan aktif/full sesuai arahan pengguna; pembacaan hasil saja memerlukan RUN_TUNING=False atau eksekusi sel hasil tersimpan. Flag tersebut tidak memberi persetujuan baru untuk pengulangan full training. Smoke test memvalidasi pipeline terlebih dahulu; full training memerlukan persetujuan setelah hasil smoke test dilaporkan.

Simpan hasil baru terpisah dari hasil historis, direncanakan di `artifacts/tuning/`. Arsipkan versi kode/protokol lama sebelum perubahan implementasi agar riwayat dapat ditelusuri. Notebook sumber tetap tanpa output; notebook beroutput dan data pasien tetap lokal.

## 6. Batasan interpretasi

Test telah digunakan pada PHASE 5. Pelatihan ulang tidak memulihkan independensinya; evaluasi pada test lama harus dinyatakan sebagai evaluasi lanjutan setelah test terpapar. Klaim evaluasi independen baru memerlukan data evaluasi yang benar-benar belum digunakan. Rujukan status historis: [laporan PHASE 5](phase_5_report.md).

Kesimpulan berlaku pada kandidat yang diuji, satu split, dan satu seed. Perbandingan menilai strategi pelatihan secara keseluruhan; FT langsung baru bukan kelanjutan hasil FE lama. Fitur identik antar-ID, perbedaan label pada sebagian kelompok, dan AHI ekstrem tetap menjadi batasan data. Eksperimen tidak membuktikan validitas klinis atau menjamin peningkatan performa.
