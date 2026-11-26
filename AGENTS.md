# Panduan proyek regresi AHI

## Status dan cakupan

Sumber kebutuhan: `BRIEF.md`. Bahasa dokumentasi dan notebook adalah Bahasa Indonesia formal. Nama variabel, fungsi, dan class menggunakan Bahasa Inggris.

Aturan tetap laporan Markdown (`.md`): tulis ringkas, padat, dan jelas. Cantumkan hanya informasi penting; hindari pengulangan, penjelasan bertele-tele, dan detail yang tidak diperlukan.

PHASE 0 sampai PHASE 5 selesai secara teknis. Sembilan CNN, baseline median/Ridge, perbandingan, pembekuan pilihan validation, dan evaluasi test telah dijalankan. Pengguna mengizinkan PHASE 5 melalui pesan "kerjakan next phase bro!". Laporan ada di `docs/phase_0_report.md` sampai `docs/phase_5_report.md`. Data dan kesimpulan penelitian tetap sementara sambil menunggu konfirmasi klien.

Perubahan arahan pengguna pada 26 November 2026: gunakan PyTorch/TorchVision dengan CUDA native Windows untuk seluruh model, menggantikan ketentuan TensorFlow/Keras dalam brief asli. Arsitektur dan protokol penelitian tetap. Implementasi ResNet50 dan smoke test tiga skenario sudah berjalan di GPU. Lihat `docs/phase_2_report.md`.

Setelah setiap phase, laporkan pekerjaan, perubahan file, pengujian aktual, masalah, keputusan metodologi, dan rekomendasi berikutnya. Tunggu persetujuan pengguna sebelum phase berikutnya. Full training memerlukan persetujuan setelah smoke test. Jangan memasang driver, WSL, atau mengubah konfigurasi sistem tanpa persetujuan.

## Data dan privasi

- Gunakan hanya `data/Flow Patient.xlsx`, `data/PulseRate.xlsx`, dan `data/Pleth.xlsx`. Jangan menggunakan file `_smote`.
- Jangan mengubah atau menghapus dataset asli. Audit mencatat SHA-256 sebelum dan sesudah pembacaan.
- Jangan mengunggah data pasien, ID, output notebook berisi data, atau artifacts ke layanan eksternal tanpa izin.
- Seluruh `data/` dan `artifacts/` diabaikan Git, kecuali `.gitkeep`. Dokumentasi yang boleh dibagikan hanya memuat statistik agregat.
- Notebook dapat memuat ID di output lokal. Bersihkan output yang mengandung informasi pasien sebelum commit atau berbagi.
- `id` tetap string, termasuk nol di awal. Standardisasi whitespace header dan metadata tanpa mengubah file sumber.

## Kontrak metodologi

- Tugas utama adalah regresi AHI kontinu. Jangan mengganti AHI dengan kelas tingkat keparahan.
- Satu sampel adalah satu pasangan `(id, Posisi)`, bukan satu baris IMF.
- Eksperimen utama menggunakan complete-case ketiga sensor. Laporkan semua pasangan yang dikeluarkan dan alasannya.
- Validasi kunci `(id, Posisi, IMF)`, sheet/posisi, nilai numerik finite, serta konsistensi AHI dalam dan antar sensor sebelum melanjutkan.
- Target yang berbeda dalam satu pasangan pasien-posisi atau antar sensor menghentikan tahap terkait. Jangan merata-ratakan target secara otomatis.
- AHI ekstrem dan prediksi negatif harus dilaporkan. Jangan clipping target, menghapus error buruk, atau mengubah prediksi diam-diam.
- AHI, ID, posisi, dan indeks IMF tidak masuk ke 11 kolom fitur sensor. Posisi masuk melalui input one-hot terpisah dengan urutan Supine, Left, Right.
- Tidak ada SMOTE. Status oversampling sumber belum dikonfirmasi.

## Temuan yang harus ditangani sebelum split

Audit PHASE 0 menemukan 160 sampel complete-case dalam 80 kelompok dengan fitur ketiga sensor identik pada posisi yang sama, melibatkan 62 ID. Lima kelompok memiliki AHI berbeda antar-ID. Kesamaan telah diverifikasi ulang dengan perbandingan elemen matriks dari Excel. Ini belum membuktikan oversampling atau identitas pasien yang sama.

Pengguna mengizinkan PHASE 1 sambil menunggu jawaban klien. Implementasi sementara mempertahankan semua baris dan mengelompokkan ID yang terhubung oleh fitur identik ketiga sensor pada posisi yang sama ke komponen terhubung. Aturan ini mengabaikan target dan menjaga seluruh posisi milik satu ID tetap bersama. Jangan menggabungkan identitas atau merata-ratakan AHI. GroupShuffleSplit hanya berbasis ID tidak cukup mencegah salinan fitur antar subset pada data ini. Provenance, lima kelompok dengan target berbeda, dan AHI ekstrem tetap perlu konfirmasi sebelum finalisasi penelitian.

Hasil sementara: 411 ID membentuk 380 komponen; train 755 sampel/287 ID, validation 157/59, test 179/65. Tidak ada overlap ID, komponen, fingerprint mentah, atau input hasil encoding antar subset. Jangan mengubah seed/split untuk memperbaiki distribusi target atau performa. Jika sumber diperbaiki klien, regenerasi seluruh preprocessing dan catat perubahan protokol sebelum memakai hasil model.

## Split, scaling, dan feature image

- Rasio awal 70/15/15 dengan seed 42, berbasis komponen pasien dan salinan. Laporkan rasio aktual pasien, kelompok, serta sampel; pembulatan jumlah kelompok dapat menggeser rasio.
- Simpan satu split manifest untuk seluruh arsitektur dan baseline. Assertion harus melarang overlap ID, dan jika diterapkan, overlap komponen salinan.
- Training saja menentukan parameter scaling, imputasi, dan transformasi yang dipelajari. Validation/test hanya ditransformasi. Statistik PHASE 0 adalah audit integritas, bukan parameter preprocessing.
- Pilihan awal adalah percentile scaling 1/99 per fitur per sensor, fit hanya pada baris IMF nyata di training. Laporkan fitur konstan dan clipping tiap subset.
- Susun setiap sensor menjadi matriks 50 x 11 berdasarkan indeks IMF numerik. Padding hanya untuk IMF yang tidak tersedia; jangan interpolasi IMF.
- Encoding terimplementasi: fitur nyata hasil scaling 0..1 dipetakan ke piksel uint8 1..255, padding 0; mask dan fitur scaled float32 disimpan terpisah. Target tetap float64 tanpa transformasi.
- Urutan channel: Flow, PulseRate, Pleth. Resize nearest-neighbor ke 224 x 224 x 3, ukuran configurable.
- Simpan piksel sebelum preprocessing backbone. Model PyTorch membagi piksel dengan 255 tepat sekali, lalu memakai mean/std dari bobot TorchVision. Jangan konversi BGR, melakukan normalisasi ganda, atau mengulang resize/crop.
- Feature image adalah pseudo-image. IMF bernomor sama antar sensor tidak membuktikan kesamaan frekuensi atau sinkronisasi fisiologis.
- Simpan data, mask, posisi one-hot, target, metadata sampel, feature order, scaler, manifest, konfigurasi, dan audit. Notebook model tidak boleh split ulang.
- Gunakan `artifacts/processed_data/{train,validation,test}.npz` dan `metadata.json` bersama manifest. Citra belum melewati preprocessing backbone. Baseline memakai flattened scaled_features, mask, dan posisi; ID/target bukan fitur.
- Hasil eksekusi notebook disimpan di `artifacts/notebooks/`; notebook sumber di `notebooks/` tetap tanpa output untuk mencegah data penelitian ikut ter-commit.

## Model dan evaluasi

- Tiga arsitektur: ResNet50, VGG16, VGG19; masing-masing satu model gabungan semua posisi.
- Masing-masing memiliki scratch, feature extraction, dan fine-tuning: sembilan kondisi eksperimen.
- PyTorch nn.Module dua input; buang classifier pretrained, gunakan AdaptiveAvgPool2d(1), concatenate posisi, Linear 128/64 dengan dropout, output Linear 1 float32.
- Head, split, metrik, dan protokol pemilihan harus sama. Konfigurasi berada di `src/config.py`.
- Transfer learning memakai ImageNet. Feature extraction membekukan backbone. Fine-tuning mulai dari checkpoint terbaik feature extraction dan membuka bagian akhir saja, lalu compile ulang dengan learning rate rendah.
- Untuk fine-tuning ResNet50, buka layer4 dan pertahankan Batch Normalization beku/eval. Untuk VGG, buka block5: indeks features mulai 24 pada VGG16 dan 28 pada VGG19.
- Adam, MSE, batch 8, learning rate awal 1e-3 dan fine-tuning 1e-5, maksimal 50 epoch per skenario, early stopping patience 10, restore best weights, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN.
- AMP/GradScaler aktif pada CUDA; output dan loss tetap float32. Catat skipped optimizer steps. Gunakan DataLoader num_workers=0 pada Windows, prefetch terbatas, del/garbage collection dan torch.cuda.empty_cache antar eksperimen.
- Bobot awal dipatok ke IMAGENET1K_V1, tidak memakai DEFAULT yang dapat berubah. Normalisasi RGB mengikuti mean/std bobot; spatial transform tetap nearest-neighbor dari PHASE 1 untuk mempertahankan grid fitur. TorchVision ResNet50 menggunakan varian V1.5.
- Checkpoint `.pt` memuat state_dict dan konfigurasi; muat dengan weights_only=True. Fine-tuning harus memvalidasi hash checkpoint feature extraction, mode smoke/full, arsitektur, dan fingerprint preprocessing.
- Smoke test tersimpan terpisah per arsitektur di `artifacts/smoke_tests/{ResNet50,VGG16,VGG19}/`; hasil ini tidak masuk perbandingan performa penelitian. Runner development hanya menerima train/validation dan menolak test set.
- Notebook `03_VGG16.ipynb` telah dieksekusi dari kernel baru dengan full training nonaktif. Hasil ada di `artifacts/notebooks/03_VGG16.executed.ipynb`; QA memverifikasi bobot ImageNet feature extraction tetap, fine-tuning hanya mengubah block5, checkpoint/reload konsisten, dan test NPZ tidak dibaca. Sebanyak 25 tes lulus. Full VGG16 harus memakai parent feature extraction full, bukan smoke.
- Hasil penuh ResNet50 tersimpan di `artifacts/experiments/ResNet50/`. Scratch terpilih dalam arsitektur ini berdasarkan RMSE validation; lihat `selection_validation.json`. Ini bukan pemilihan berdasarkan test atau kesimpulan antararsitektur.
- Hasil penuh VGG16 tersimpan di `artifacts/experiments/VGG16/`. Fine-tuning terpilih dalam arsitektur ini berdasarkan RMSE validation. Checkpoint terbaik berasal dari epoch 8; tahap dilanjutkan sampai early stopping pada epoch 18. Model reload, metrik, pembekuan blok awal, parent feature extraction full, hash data/kode, dan pemilihan telah diverifikasi. Test NPZ tidak dimuat.
- Hasil penuh VGG19 tersimpan di `artifacts/experiments/VGG19/`. Fine-tuning terpilih dalam arsitektur ini berdasarkan RMSE validation; checkpoint terbaik epoch 2, early stopping epoch 12. Feature extraction tetap memakai bobot ImageNet; fine-tuning hanya mengubah convolution block5 pada indeks 28/30/32/34 dari parent feature extraction full. Reload, metrik, pemilihan, hash data/kode, dan pembekuan blok awal diverifikasi. Sebanyak 26 tes lulus. Test NPZ tidak dimuat.
- Gunakan `src/reporting.py` untuk merangkum tiga kondisi penuh dengan fingerprint yang sama. Kriteria tetap: RMSE validation terkecil, MAE jika seri. Waktu kumulatif fine-tuning memasukkan tahap feature extraction.
- Notebook sumber default menonaktifkan full training agar tidak terulang. Notebook laporan dalam `artifacts/notebooks/`: `02_ResNet50.full_results.executed.ipynb`, `03_VGG16.full_results.executed.ipynb`, dan `04_VGG19.full_results.executed.ipynb` membaca hasil tersimpan dan tidak mengulang training. VGG16/VGG19 juga menonaktifkan penulisan ulang runtime/ringkasan smoke saat `RUN_SMOKE_TESTS=False`.
- Gunakan validation untuk pemilihan, tuning, early stopping, dan pembekuan konfigurasi. Test hanya setelah protokol dibekukan. Jangan memilih model berdasarkan test.
- Laporkan MAE, MSE, RMSE, dan R2 keseluruhan serta per posisi beserta jumlah sampelnya. R2 pada subset terlalu kecil harus dinyatakan tidak terdefinisi.
- Simpan checkpoint, history, waktu, parameter trainable, versi library, hardware, konfigurasi, metrik, dan prediksi yang memungkinkan penghitungan ulang metrik.
- Jika memakai confidence interval, bootstrap kelompok pasien; pertimbangkan komponen salinan sesuai keputusan split.
- Baseline tambahan: DummyRegressor median training dan Ridge atau Random Forest dengan split dan sampel yang sama.
- PHASE 5 memakai DummyMedian dan Ridge dengan 1.803 fitur: flattened scaled_features, mask IMF, one-hot posisi. StandardScaler hanya fit pada train. Alpha Ridge 0.1/1/10/100/1000 dipilih dari validation; alpha 1000 terpilih. Tidak ada refit train+validation atau training ulang CNN.
- Hasil akhir ada di `artifacts/comparison/`. `frozen_protocol.json` dan `selection_validation.json` dibuat sebelum test NPZ dimuat sekali. CNN utama dan pemenang keseluruhan yang dipilih dari validation adalah ResNet50/scratch; baseline terpilih DummyMedian. Test 179 sampel/65 ID/57 komponen, dengan Supine 65, Left 58, Right 56. Hasil test tidak mengubah pemilihan.
- Source `05_Model_Comparison.ipynb` default `RUN_FINAL_EVALUATION=False`; membaca artifacts. Runner menolak direktori hasil yang sudah ada untuk mencegah pengulangan fitting setelah test terpapar. Laporan dieksekusi dari kernel baru dalam `artifacts/notebooks/05_Model_Comparison.executed.ipynb`.
- Test sekarang sudah terpapar. Jangan tuning memakai test sambil menyebut hasil baru sebagai evaluasi independen. Verifikasi aset beku dengan `verify_frozen_protocol`; jika memperbaiki sumber atau metode, catat protokol baru dan konsekuensi terhadap independensi evaluasi.
- Ridge validation RMSE 459,18 didominasi satu sampel (99,60% kuadrat error), dengan ekstrapolasi fitur setelah StandardScaler. Diagnosis baca-saja tersimpan; model dan alpha tidak diubah setelah test. Prediksi negatif dipertahankan: Ridge 12 validation/9 test, VGG19 fine-tuning 1 test. Panel CNN pendamping hanya memperjelas skala grafik; data lengkap tetap tersedia.
- Plot akademik minimal 300 DPI. Bland-Altman harus menyebut pengukuran berulang pasien.
- Interpretasi memakai output aktual. Jangan membuat angka performa, menganggap smoke test sebagai hasil penelitian, atau mengklaim validasi klinis.

## Rencana pelaksanaan

| Phase | Implementasi dan validasi | Syarat selesai |
|---|---|---|
| 0 | Struktur direktori, audit aktual, config, requirements, panduan, laporan | Dataset tidak berubah; audit dan pengujian tercatat |
| 1 | `src/preprocessing.py`, `notebooks/01_Preprocessing.ipynb`; putuskan grouping salinan; bangun manifest, scaler, matriks, mask, one-hot dan dataset tersimpan | Notebook dijalankan; tidak ada overlap pasien/salinan sesuai protokol; target dan input selaras; shape dan finite valid |
| 2 | `src/model_utils.py`, `src/evaluation.py`, `02_ResNet50.ipynb`; scratch, feature extraction, fine-tuning | GPU dan VRAM diuji lewat smoke test; full eksperimen disetujui; reload model dan metrik diverifikasi |
| 3 | `03_VGG16.ipynb` memakai pipeline tervalidasi | Smoke test lalu eksperimen; checkpoint, prediksi, history dan metrik tersimpan |
| 4 | `04_VGG19.ipynb` memakai pipeline yang sama | Smoke test lalu eksperimen; artifacts tersimpan |
| 5 | `05_Model_Comparison.ipynb`, baseline tabular, plot dan pembahasan aktual; README final | Sembilan kondisi tersedia, tidak training ulang CNN pada notebook perbandingan, QA reproduksibilitas selesai |

Jangan membuat notebook atau modul fase lanjutan sebagai placeholder. Gunakan akar workspace yang sudah memiliki `data/`, tanpa menyalin ulang data ke folder proyek bersarang.

## Pemeriksaan dan gaya implementasi

Jalankan dari akar proyek:

```powershell
python -m src.audit
python -m unittest discover -s tests -v
python -m pip check
```

Gunakan pathlib. Jangan hardcode path komputer, jumlah dataset sebagai hasil, atau nilai metrik. Hindari helper berulang antar notebook. Komentar cukup menjelaskan alasan atau batasan yang tidak terlihat dari kode. Pada sesi ini antislop dipilih dalam mode DURING; jangan menyimpan preferensi global tanpa permintaan pengguna.

Environment aktif: `.venv` Python 3.11.0 dengan system-site-packages; nbformat 5.11.1 dan nbclient 0.11.0 dipasang lokal. Kernel: `osa-ahi-preprocessing`. Torch 2.9.0+cu130 dan TorchVision 0.24.0+cu130 memakai RTX 3060 Ti native Windows, CUDA 13.0. WSL tidak diperlukan. TensorFlow sempat dipasang lokal saat persiapan awal, tetapi tidak digunakan oleh pipeline model setelah perubahan arahan.
