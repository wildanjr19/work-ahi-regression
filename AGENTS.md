# Panduan proyek regresi AHI

## Acuan dan status

- Bahasa dokumentasi/notebook: Bahasa Indonesia formal; nama variabel, fungsi, dan class: Bahasa Inggris. Markdown harus ringkas, jelas, dan tanpa pengulangan.
- Baca `docs/desain_eksperimen.md` dan `docs/search_space.md` sebelum mengubah model/tuning. Kesepakatan 27 November 2026 menggantikan ketentuan model lama yang berbeda dalam `BRIEF.md` dan laporan fase, termasuk FT dari checkpoint FE.
- PHASE 0 sampai PHASE 5 selesai. Konfigurasi/model, runner, integrasi notebook, smoke aktual, dan full grid tuning selesai: 108 kandidat berhasil, tanpa kegagalan. Evaluasi test tuning lanjutan selesai: pilihan CNN tetap ResNet50/scratch dari validation; sembilan kondisi CNN dan dua baseline dievaluasi tanpa training ulang CNN. Lihat `docs/tuning_full_report.md` dan `docs/tuning_evaluation_report.md`. Rujukan: `docs/tuning_model_report.md`, `docs/tuning_runner_report.md`, dan `docs/tuning_notebook_report.md`. Arsip prasyarat: 571 berkas, 82 aset beku, 88 baris metrik cocok, 32 tes lulus; angka ini historis.
- Rujukan wajib sesuai pekerjaan: `docs/konfirmasi_dataset_klien.md` untuk keputusan data; `docs/preprocessing.md` dan `docs/phase_1_report.md` untuk preprocessing; `docs/arsip_verifikasi.md` untuk integritas; `docs/phase_0_report.md` sampai `docs/phase_5_report.md` untuk riwayat implementasi/hasil.
- Setelah setiap tahap, laporkan pekerjaan, perubahan file, pengujian aktual, masalah, keputusan metodologi, dan rekomendasi; tunggu persetujuan sebelum tahap berikutnya. Full training baru memerlukan persetujuan setelah laporan smoke test; persetujuan training historis tidak berlaku.
- Jangan memasang driver/WSL atau mengubah konfigurasi sistem tanpa persetujuan.

## Data, privasi, dan integritas

- Gunakan hanya `data/Flow Patient.xlsx`, `data/PulseRate.xlsx`, dan `data/Pleth.xlsx`; tanpa file `_smote` atau SMOTE. Jangan mengubah/menghapus workbook asli; audit mencatat SHA-256 sebelum/sesudah pembacaan.
- Jangan mengunggah data pasien, ID, notebook berisi data, atau artifacts tanpa izin. `data/` dan `artifacts/` diabaikan Git kecuali `.gitkeep`; dokumentasi yang dibagikan hanya memuat agregat.
- Notebook sumber `notebooks/` harus tanpa output/execution count. Notebook beroutput tetap lokal; bersihkan informasi pasien sebelum commit/berbagi.
- Pertahankan dataset, target, preprocessing, manifest, metadata, hasil historis, dan protokol beku. Jangan memperbarui hash lama agar cocok dengan kode baru. Hasil baru di `artifacts/tuning/`; jangan menimpa `artifacts/experiments/`, `artifacts/smoke_tests/`, `artifacts/comparison/`, atau notebook historis di `artifacts/notebooks/`.
- Snapshot lama: `artifacts/archives/pre_tuning_2026-11-27/`. Sebelum kode berubah, `verify_frozen_protocol` memverifikasi aset aktif; setelah perubahan kode, periksa snapshot dengan verifier arsip. Snapshot berada pada drive yang sama dan memerlukan workbook asli, bukan cadangan data lengkap pada media terpisah.

## Kontrak data dan split

- Tugas: regresi AHI kontinu; satu sampel = `(id, Posisi)`, bukan baris IMF. Gunakan complete-case ketiga sensor; laporkan pasangan yang dikeluarkan beserta alasan.
- Pertahankan `id` sebagai string termasuk nol awal. Rapikan whitespace header/metadata tanpa mengubah sumber. Validasi kunci `(id, Posisi, IMF)`, sheet/posisi, numerik finite, dan konsistensi AHI dalam/antar sensor. Konflik target menghentikan tahap terkait; jangan merata-ratakan target.
- Arahan klien: seluruh 411 ID tetap tercakup; hanya 16 pasangan tanpa PulseRate dikeluarkan. Pertahankan AHI ekstrem, prediksi negatif, dan error besar; jangan clipping target/prediksi atau menghapus hasil buruk.
- Fitur identik ditemukan pada 160 sampel/80 kelompok/62 ID; lima kelompok memiliki AHI berbeda antar-ID. Temuan ini tidak membuktikan oversampling atau kesamaan identitas; persetujuan penggunaan data tidak memverifikasi asal/validitas klinis label.
- Split memakai komponen terhubung ID dengan fitur ketiga sensor identik pada posisi sama, mengabaikan target. Seluruh posisi satu ID dan salinan fitur harus bersama. Jangan menggabungkan identitas/target; GroupShuffleSplit berbasis ID saja tidak cukup.
- Gunakan manifest yang sama untuk seluruh CNN/baseline: seed 42, rasio awal 70/15/15, 380 komponen; train 755 sampel/287 ID, validation 157/59, test 179/65. Laporkan rasio aktual sampel, ID, dan komponen, termasuk dampak pembulatan.
- Assert nol overlap ID, komponen, fingerprint mentah, dan input encoding. Jangan split ulang atau mengubah seed untuk distribusi/performa. Jika klien memperbaiki sumber, catat protokol baru dan regenerasi seluruh preprocessing sebelum memakai hasil model.

## Input dan preprocessing

- Gunakan `artifacts/processed_data/{train,validation,test}.npz`, `metadata.json`, dan manifest tersimpan. Tuning hanya memuat train/validation.
- Seluruh transformasi yang dipelajari, termasuk scaling/imputasi, hanya fit pada train. Percentile scaling 1/99 per fitur/sensor memakai baris IMF nyata training; laporkan fitur konstan dan clipping tiap subset. Statistik audit bukan parameter preprocessing.
- Setiap sensor membentuk matriks 50 x 11 menurut indeks IMF numerik. Padding hanya untuk IMF hilang, tanpa interpolasi. AHI, ID, posisi, dan indeks IMF bukan 11 fitur sensor.
- Fitur scaled float32 0..1 dipetakan ke piksel uint8 1..255, padding 0; mask disimpan terpisah, target asli float64. Channel: Flow, PulseRate, Pleth. Resize nearest-neighbor ke 224 x 224 x 3 melalui konfigurasi.
- Posisi masuk lewat one-hot terpisah: Supine, Left, Right. Simpan input, mask, posisi, target, metadata sampel, feature order, scaler, manifest, konfigurasi, dan audit; periksa shape, finite, serta keselarasan input-target.
- Piksel tersimpan belum dinormalisasi backbone. Model membagi 255 tepat sekali lalu memakai mean/std bobot TorchVision; tanpa BGR, normalisasi ganda, atau resize/crop tambahan.
- Feature image adalah pseudo-image; indeks IMF sama antar sensor tidak membuktikan frekuensi sama atau sinkronisasi fisiologis.
- Baseline median training dan Ridge memakai split/sampel sama; input Ridge = flattened scaled_features, mask IMF, dan posisi, tanpa ID/target. StandardScaler hanya fit pada train.

## Model dan grid search baru

- PyTorch/TorchVision dengan CUDA native Windows menggantikan TensorFlow/Keras. ResNet50 (varian TorchVision V1.5), VGG16, dan VGG19 masing-masing memakai satu model semua posisi.
- Sembilan kondisi = tiga arsitektur x scratch/FE/FT. Setiap kandidat diinisialisasi baru, tanpa meneruskan bobot kandidat lain. Backbone pretrained dipatok ke `IMAGENET1K_V1`, bukan `DEFAULT`.

| Strategi | Inisialisasi dan bobot yang dilatih |
|---|---|
| Scratch | Backbone acak dan head baru; seluruh bobot dilatih |
| FE | Backbone ImageNet beku; hanya head baru dilatih |
| FT langsung | Backbone ImageNet dan head baru; head serta blok akhir dilatih sejak awal, tanpa checkpoint FE atau pelatihan head pendahuluan |

- FT membuka `layer4` ResNet50, block5 VGG16 mulai `features[24]`, atau block5 VGG19 mulai `features[28]`; blok awal beku. Batch Normalization ResNet50 beku/eval pada FE dan FT.
- Semua model: `nn.Module` dua input, classifier pretrained dibuang, AdaptiveAvgPool2d(1), concatenate posisi, Linear 128/64 dengan ReLU dan dua dropout bernilai sama, output Linear 1. Output/loss float32.
- Grid sama untuk semua kondisi: learning rate awal `1e-6, 1e-5, 1e-4, 1e-3` x dropout `0.1, 0.3, 0.5`. Uji 12 kombinasi/kondisi = 108 pelatihan kandidat di luar smoke test.
- Tetap: Adam, MSE, batch 8, weight decay 0, maksimal 50 epoch, early stopping patience 10, restore checkpoint terbaik berdasarkan MSE validation. ReduceLROnPlateau: faktor 0,5, patience 5, minimum LR `1e-7`. Reset seed 42 tiap kandidat; hentikan/catat kandidat nonfinite.
- AMP/GradScaler aktif pada CUDA; catat skipped optimizer steps. Windows memakai DataLoader `num_workers=0`, prefetch terbatas; lakukan del/garbage collection dan `torch.cuda.empty_cache` antar eksperimen.
- Checkpoint `.pt` memuat state_dict/konfigurasi; load dengan `weights_only=True`. Validasi inisialisasi, arsitektur, mode smoke/full, konfigurasi kandidat, dan fingerprint preprocessing; FT baru tidak memuat parent FE.
- Konfigurasi bersama di `src/config.py`, grid search bersama di `src/`; perbarui `02_ResNet50.ipynb`, `03_VGG16.ipynb`, dan `04_VGG19.ipynb` tanpa duplikasi helper. Tuning/full training default nonaktif.
- Runner baru: `src.tuning.run_grid_search()`, default `enabled=False`, mode `smoke`. Smoke memakai LR `1e-3`/dropout `0.3` pada tiga strategi; full memakai seluruh grid dan memerlukan `full_training_approved=True`. Ringkasan smoke tidak boleh dipakai memilih model.
- Tiga notebook memanggil runner tanpa loop training tersalin. Sesuai arahan pengguna saat persiapan push, flag notebook dipertahankan pada `RUN_TUNING=True`, `RUN_MODE='full'`, dan `FULL_TRAINING_APPROVED=True`; default API runner tetap nonaktif/smoke. Namespace `artifacts/tuning/grid_v1/` sudah berisi hasil dan menolak penimpaan. Untuk membaca hasil saja, nonaktifkan `RUN_TUNING` atau jalankan sel hasil tersimpan. Pengulangan full training tetap memerlukan persetujuan baru, terlepas dari flag tersimpan. Salinan notebook smoke beroutput berada di `artifacts/tuning/notebook_checks/`, bukan direktori notebook historis.
- Pisahkan smoke dan hasil penuh per arsitektur di `artifacts/tuning/`; runner tuning menolak test. Smoke memeriksa inisialisasi mandiri, bobot/Batch Normalization beku, pembaruan blok FT, checkpoint/reload, dan larangan akses test.
- Simpan tiap kandidat termasuk gagal: status, seed, konfigurasi, checkpoint, history, metrik/prediksi validation, waktu, parameter trainable, versi library, hardware, dan hash data/kode. Laporkan biaya/epoch aktual; waktu FT langsung tidak memasukkan FE terpisah.

## Pemilihan, evaluasi, dan riwayat

- Validation dipakai untuk scheduler, early stopping, tuning, dan seluruh pemilihan. Pilih konfigurasi tiap kondisi lalu CNN utama dari sembilan kondisi dengan RMSE validation terkecil, MAE jika seri.
- Bekukan konfigurasi/checkpoint sebelum evaluasi test. Jangan refit train+validation, memilih model dari test, atau memperluas grid berdasarkan test.
- Test sudah terpapar pada PHASE 5. Evaluasi ulang adalah evaluasi lanjutan setelah paparan; evaluasi independen baru memerlukan data yang benar-benar belum digunakan.
- Ringkasan memakai `src/reporting.py` dengan fingerprint sama. `05_Model_Comparison.ipynb` membaca hasil tuning terpilih melalui `src.tuning_evaluation.run_tuning_evaluation()` tanpa training ulang CNN; `RUN_FINAL_EVALUATION=False` secara default. Namespace evaluasi: `artifacts/tuning/grid_v1/evaluation_v1/`; hasil historis di `artifacts/comparison/` tetap dipertahankan. Runner evaluasi menolak direktori hasil yang sudah ada.
- Laporkan MAE, MSE, RMSE, R2 keseluruhan/per posisi beserta jumlah sampel; R2 subset tidak memadai dinyatakan tidak terdefinisi. Jika memakai CI, bootstrap kelompok pasien dengan mempertimbangkan komponen salinan.
- Plot akademik minimal 300 DPI; Bland-Altman menyebut pengukuran berulang pasien. Kesimpulan dibatasi kandidat, satu split/seed, dan keterbatasan data. Gunakan hasil aktual; smoke bukan hasil penelitian; jangan mengklaim validasi klinis atau menjamin peningkatan performa.
- Hasil historis dan diagnostik lengkap tetap dirujuk melalui laporan fase. FT lama berasal dari checkpoint FE; validasi parent FE dan biaya kumulatif FE+FT hanya berlaku pada hasil lama. PHASE 5 memilih ResNet50/scratch dan baseline DummyMedian dari validation; test tidak mengubah pilihan. Rincian Ridge, prediksi negatif, epoch, serta metrik historis tidak menjadi aturan tuning baru.

## Pemeriksaan dan pelaksanaan

Smoke aktual sembilan kondisi selesai dari tiga kernel baru: input 224 x 224, batch 8, sembilan optimizer update, nol skipped AMP steps; test NPZ tidak dimuat. Sebanyak 49 tes lulus. VGG19 FE menghasilkan delapan prediksi validation negatif pada smoke dan tetap dipertahankan. Hasil ini hanya diagnostik; lihat `docs/tuning_notebook_report.md`.

Full grid dan evaluasi test lanjutan selesai. Pilihan validation serta checkpoint dibekukan sebelum test dimuat sekali; 53 tes lulus, 88 baris metrik dihitung ulang, dan 483 berkas aktif historis/data tidak berubah. Hasil evaluasi: `artifacts/tuning/grid_v1/evaluation_v1/`; laporan: `docs/tuning_evaluation_report.md`. Tunggu persetujuan sebelum tahap berikutnya. `train_experiment` menyediakan loop bersama; jalur default `historical` tetap dari parent FE, sedangkan runner tuning memakai `initialization="independent"`. Jangan membuat placeholder fase lanjutan atau menyalin data ke proyek bersarang; gunakan akar workspace.

Jalankan pemeriksaan implementasi dari akar proyek dengan environment `.venv`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Perubahan dokumentasi saja cukup diperiksa lewat diff dan keselarasan acuan. Audit ulang memakai `src.audit.run_audit()` dengan direktori keluaran baru sesuai `docs/arsip_verifikasi.md`, agar audit lama tidak tertimpa. Verifikasi snapshot tanpa inferensi atau memuat array test:

```powershell
.\.venv\Scripts\python.exe artifacts/archives/pre_tuning_2026-11-27/verify_archive.py
```

Gunakan pathlib; jangan hardcode path komputer, jumlah data sebagai hasil, atau metrik. Komentar menjelaskan alasan/batasan yang tidak terlihat dari kode. Antislop DURING berlaku pada sesi ini; jangan menyimpan preferensi global tanpa permintaan.

Environment: `.venv` Python 3.11.0 dengan system-site-packages; kernel `osa-ahi-preprocessing`; Torch 2.9.0+cu130/TorchVision 0.24.0+cu130, RTX 3060 Ti, CUDA 13.0 native Windows. Detail persiapan/library notebook tersedia pada laporan fase; WSL tidak diperlukan.
