# Prediksi AHI menggunakan tiga sensor dan transfer learning

Proyek penelitian ini memprediksi AHI kontinu per pasien dan posisi tidur menggunakan fitur IMF dari Flow, Pulse Rate, dan Pleth. ResNet50, VGG16, dan VGG19 dibandingkan melalui scratch, feature extraction, dan fine-tuning, dengan baseline median dan Ridge. Data sumber berupa fitur tabular, bukan citra fisiologis atau sinyal mentah.

Hasil historis: PHASE 0 sampai PHASE 5 selesai secara teknis. ResNet50 scratch terpilih dari validation sebelum test; pada 179 sampel test, MAE 17,9986, RMSE 29,0473, R2 0,1059. Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan. Lihat [laporan PHASE 5](docs/phase_5_report.md), [PHASE 4](docs/phase_4_report.md), [PHASE 3](docs/phase_3_report.md), [PHASE 2](docs/phase_2_report.md), dan [aturan proyek](AGENTS.md).

Tuning baru: tiga notebook memakai runner bersama dengan grid sama dan FT langsung dari ImageNet. Integrasi dan smoke aktual sembilan kondisi lulus; full grid selesai dengan 108/108 kandidat berhasil. Evaluasi test tuning lanjutan selesai; ResNet50/scratch tetap terpilih dari validation, dengan test MAE 17.8816, RMSE 29.4082, dan R2 0.0835. Lihat [laporan evaluasi tuning](docs/tuning_evaluation_report.md). Test telah terpapar pada PHASE 5. Lihat [laporan full grid](docs/tuning_full_report.md). Acuan: [desain eksperimen](docs/desain_eksperimen.md), [search space](docs/search_space.md), dan [laporan smoke](docs/tuning_notebook_report.md). Hasil baru terpisah dari seluruh hasil historis.

## Struktur proyek

Workspace saat ini menjadi akar proyek, tanpa folder `OSA_Transfer_Learning/` tambahan.

```text
data/                     Tiga workbook asli, tidak diubah
src/config.py             Path, kontrak fitur, konfigurasi preprocessing/training
src/audit.py              Audit dataset dan environment yang dapat diulang
src/preprocessing.py      Grouping, split, scaling, encoding, penyimpanan/validasi
src/model_utils.py        Model PyTorch, DataLoader, checkpoint, runner smoke/full
src/tuning.py             Grid search bersama, default nonaktif, tanpa akses test
src/evaluation.py         Metrik, prediksi, dan grafik regresi
src/reporting.py          Ringkasan tiga eksperimen penuh dan pemilihan validation
src/comparison.py         Baseline, protokol beku, pemilihan validation, inferensi test
src/comparison_plots.py   Grafik perbandingan dan pembahasan dari hasil aktual
notebooks/01_Preprocessing.ipynb  Notebook preprocessing tanpa output pasien
notebooks/02_ResNet50.ipynb       Tuning ResNet50 melalui runner bersama
notebooks/03_VGG16.ipynb         Tuning VGG16 melalui runner bersama
notebooks/04_VGG19.ipynb         Tuning VGG19 melalui runner bersama
notebooks/05_Model_Comparison.ipynb  Evaluasi tuning sembilan CNN dan dua baseline
tests/                    Pengujian audit, preprocessing, model dan pelaporan
docs/                     Laporan per phase dan keputusan penggunaan data
artifacts/audit/           Bukti audit lokal, termasuk metadata pasien
artifacts/processed_data/ Dataset hasil PHASE 1
artifacts/notebooks/      Notebook yang telah dieksekusi, tetap lokal
artifacts/smoke_tests/    Checkpoint/history/prediksi smoke, bukan hasil penelitian
artifacts/experiments/   Hasil penuh tiap arsitektur, termasuk metrik validation
artifacts/comparison/    Protokol, baseline, metrik validation/test, grafik, dan QA
artifacts/tuning/        Hasil tuning baru, smoke, serta notebook verifikasi lokal
artifacts/models/         Checkpoint model
artifacts/metrics/        Metrik dan log eksperimen
artifacts/predictions/    Hasil prediksi
artifacts/figures/        Visualisasi penelitian
requirements.txt          Versi paket yang benar-benar tersedia saat PHASE 0
requirements-notebooks.txt Kebutuhan notebook yang diuji pada PHASE 1
requirements-training.txt PyTorch/TorchVision CUDA sesuai environment aktual
```

Data dan artifacts diabaikan Git. Jangan mengunggah dataset, ID pasien, atau output notebook yang mengandung informasi pasien tanpa izin.

## Melanjutkan di laptop

Clone kode dari repository:

```powershell
git clone https://github.com/wildanjr19/work-ahi-regression.git
cd work-ahi-regression
```

Pindahkan `data/` serta `artifacts/processed_data/`, `artifacts/experiments/`, `artifacts/comparison/`, dan `artifacts/notebooks/` melalui USB atau jaringan lokal. Pertahankan struktur folder; hasil inti historis sekitar 1,4 GiB. Sertakan `artifacts/tuning/` untuk hasil baru. `smoke_tests/` menyimpan diagnostik historis; `torch_cache/` tidak diperlukan untuk membaca hasil atau memuat checkpoint terlatih.

Buat `.venv` baru dan pasang dependensi sesuai perangkat laptop. Buka notebook 05 dengan `RUN_FINAL_EVALUATION=False` untuk membaca hasil tuning di `artifacts/tuning/grid_v1/evaluation_v1/` tanpa fitting atau inferensi ulang. Notebook beroutput terbaru tersedia di `artifacts/tuning/notebook_checks/05_Model_Comparison.tuning.report.executed.ipynb`. Protokol beku menyimpan path absolut PC; verifikasi aset di lokasi baru memerlukan pemetaan path tanpa menimpa protokol asli. Clone saja tidak menyertakan data atau hasil lokal.

## Dataset

Sediakan `Flow Patient.xlsx`, `PulseRate.xlsx`, dan `Pleth.xlsx` dalam `data/`. Masing-masing harus memiliki sheet Supine, Left, dan Right serta kolom `id`, `Posisi`, `IMF`, 11 fitur numerik, dan `AHI`. Daftar fitur dan urutannya ada di `src/config.py`. File `_smote` tidak dipakai.

Satu sampel adalah pasangan pasien-posisi. Baris IMF merupakan komponen sampel tersebut. Audit memeriksa whitespace header, ID string, posisi, tipe numerik, nilai hilang/infinite, duplikasi kunci, urutan dan kelengkapan IMF, konsistensi target, serta kelengkapan sensor.

## Environment audit dan cara menjalankan

Python 3.11.0 berjalan pada Windows 11. `.venv` dengan system-site-packages memakai kembali paket ilmiah dan PyTorch CUDA yang sudah tersedia; nbformat/nbclient dipasang lokal. Versi paket ada di requirements dan log runtime. TensorFlow sempat dipasang saat persiapan, tetapi pipeline model sekarang memakai PyTorch. WSL tidak diperlukan.

Untuk mereproduksi audit dalam environment terpisah:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.audit
```

Untuk notebook, pasang tooling dan daftarkan kernel lokal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name osa-ahi-preprocessing --display-name "OSA AHI Preprocessing"
```

Buka `notebooks/01_Preprocessing.ipynb` di VS Code, pilih interpreter/kernel `.venv`, lalu jalankan seluruh sel. Notebook tidak membutuhkan TensorFlow atau GPU dan tidak bergantung pada state notebook sebelumnya. Hasil eksekusi yang sudah diverifikasi ada di `artifacts/notebooks/01_Preprocessing.executed.ipynb`; jangan commit output tersebut.

Untuk audit saja, jalankan `python -m src.audit`. Hasil masuk ke `artifacts/audit/`. Jika ditemukan masalah integritas, command gagal setelah menyimpan rincian; schema yang tidak sesuai menghentikan pembacaan lebih awal.

Requirements memuat dependensi langsung, bukan lockfile transitif. Instalasi bersih pada komputer lain belum diuji. Versi preprocessing dicatat di `artifacts/processed_data/environment_versions.json`; versi model di `artifacts/smoke_tests/ResNet50/runtime.json` dan `run.json` tiap skenario.

## GPU PyTorch pada Windows

Environment aktual memakai torch 2.9.0+cu130 dan torchvision 0.24.0+cu130. Operasi CUDA dan smoke test tiga skenario berhasil pada RTX 3060 Ti 8 GB, driver 610.88. Pilihan paket CUDA Windows tersedia pada [panduan resmi PyTorch](https://pytorch.org/get-started/previous-versions/).

Untuk environment baru, pasang paket proyek dan verifikasi CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Buka notebook 02, 03, atau 04 dengan kernel `.venv` yang sama. Sesuai arahan pengguna, flag tersimpan dipertahankan pada `RUN_TUNING=True`, `RUN_MODE='full'`, dan `FULL_TRAINING_APPROVED=True`. Run All memanggil training; runner menolak namespace hasil yang sudah ada. Untuk hanya membaca hasil, ubah `RUN_TUNING=False` atau jalankan sel hasil tersimpan. Default API runner tetap nonaktif/smoke. Flag tersimpan tidak menggantikan persetujuan baru untuk mengulang full training.

Smoke aktual sudah dijalankan dari tiga kernel baru: batch 8, satu batch train/validation, satu epoch, AMP. Hasil berada di `artifacts/tuning/grid_v1/{ResNet50,VGG16,VGG19}/smoke/`; salinan notebook beroutput di `artifacts/tuning/notebook_checks/`. Smoke tidak memilih model. Untuk mengulang, gunakan namespace baru karena runner menolak direktori hasil yang sudah ada.

Full grid memakai 12 kombinasi per strategi dan memerlukan persetujuan setelah laporan smoke. Jangan mengaktifkan flag full sebelum persetujuan. Runner hanya membaca train/validation. Hasil historis tetap di `artifacts/experiments/` dan `artifacts/smoke_tests/`; laporan ketiga arsitektur tersedia melalui `artifacts/notebooks/*.full_results.executed.ipynb`.

## Perbandingan historis dan reproduksibilitas

Ringkasan khusus tiga arsitektur tersedia pada [perbandingan ResNet50, VGG16, dan VGG19](docs/perbandingan_model.md).

Laporan beroutput historis tetap tersedia di `artifacts/notebooks/05_Model_Comparison.executed.ipynb`. Notebook sumber `05_Model_Comparison.ipynb` kini membaca hasil tuning di `artifacts/tuning/grid_v1/evaluation_v1/`, dengan `RUN_FINAL_EVALUATION=False`; laporan tuning beroutput tersedia di `artifacts/tuning/notebook_checks/05_Model_Comparison.tuning.report.executed.ipynb`. Membaca laporan tidak melatih ulang model atau mengulang inferensi test.

Hasil ada di `artifacts/comparison/`: `comparison_validation.csv`, `comparison_test.csv`, metrik dan prediksi per posisi, `training_costs.csv`, `transfer_contributions.csv`, checkpoint baseline, `research_discussion.md`, dan `quality_checks.json`. Sebanyak 32 tes lulus dan 95 grafik disimpan pada 300 DPI. Pengujian setelah paket training terpasang:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Baseline memakai 1.803 fitur dari scaled_features, mask IMF, dan one-hot posisi. Median serta StandardScaler hanya fit pada train. Ridge memilih alpha 1000 dari grid validation 0.1/1/10/100/1000; tidak ada refit memakai validation. Baseline ini menguji representasi sebelum kuantisasi citra, sehingga bukan perbandingan arsitektur murni.

`frozen_protocol.json` mencatat pilihan validation dan hash checkpoint, data, serta kode sebelum test dimuat. Test mencakup 179 sampel dari 65 ID dan 57 komponen: Supine 65, Left 58, Right 56. Test sekarang sudah terpapar; jangan memakainya untuk tuning sambil mengklaim evaluasi independen. Runner menolak direktori hasil yang sudah ada. Jangan menghapus artifacts untuk mengulang pencarian memakai test yang sama.

Ridge memiliki RMSE validation 459,18; satu kasus menyumbang 99,60% kuadrat error. Diagnosis baca-saja menunjukkan ekstrapolasi setelah StandardScaler, bukan perubahan model setelah test. Semua hasil buruk dan prediksi negatif dipertahankan. Grafik khusus CNN melengkapi grafik seluruh kondisi agar skala Ridge tidak menutupi perbedaan CNN.

## Urutan notebook

Notebook dibuat dan dijalankan satu fase per persetujuan, bukan sekaligus.

1. `01_Preprocessing.ipynb`: audit/EDA, complete-case sensor, grouping, split manifest, fit scaler training, matriks dan mask, penyimpanan dataset.
2. `02_ResNet50.ipynb`: tiga skenario ResNet50, dimulai smoke test.
3. `03_VGG16.ipynb`: tiga skenario VGG16 dengan pipeline dan split yang sama.
4. `04_VGG19.ipynb`: tiga skenario VGG19 dengan pipeline dan split yang sama.
5. `05_Model_Comparison.ipynb`: membaca hasil tersimpan, menjalankan baseline tabular, membandingkan dan membahas hasil aktual.

## Preprocessing dan model

Preprocessing memakai complete-case tiga sensor. Seluruh 411 ID tetap terwakili; 16 pasangan ID-posisi tanpa PulseRate dikeluarkan. ID yang terhubung oleh fitur mentah identik ketiga sensor pada posisi sama mengikuti satu komponen split; seluruh posisi pasien tetap bersama. Identitas dan target dipertahankan. Keputusan penggunaan data tercatat dalam [keputusan klien](docs/konfirmasi_dataset_klien.md).

Rasio awal training/validation/test adalah 70/15/15, seed 42. Fit percentile scaling 1/99 per fitur dan sensor hanya dari training. Target AHI dipertahankan dalam skala aslinya. Posisi dipakai sebagai one-hot terpisah; ID dan AHI tidak menjadi fitur.

Setiap sensor menjadi matriks 50 x 11 dengan IMF diurutkan numerik. Fitur yang tersedia dipetakan ke piksel uint8 1..255, padding ke 0, dan mask disimpan. Gabungan Flow/PulseRate/Pleth menjadi 50 x 11 x 3 lalu di-resize nearest-neighbor ke 224 x 224 x 3. Ini pseudo-image; nomor IMF sama tidak membuktikan sinkronisasi fisiologis.

Dataset tersimpan sebagai `train.npz`, `validation.npz`, dan `test.npz`. Setiap file memuat `images`, `feature_images`, `scaled_features`, `imf_mask`, `position_one_hot`, `targets`, `patient_ids`, `positions`, `sample_ids`, dan `patient_components`. Muat dengan `np.load(path, allow_pickle=False)`.

Citra belum mendapat preprocessing backbone. Dataset PyTorch mengubah NHWC menjadi NCHW. Model membagi piksel dengan 255 tepat sekali, lalu memakai mean/std RGB dari bobot TorchVision IMAGENET1K_V1. Resize bilinear dan center crop bawaan recipe fotografi tidak diterapkan agar grid fitur tetap utuh. Target float64 menyimpan AHI asli; loss memakai float32. Baseline memakai flattened scaled_features, mask, dan posisi dengan manifest yang sama.

`metadata.json` menyimpan catatan saat preprocessing, konfigurasi, versi, hash sumber/output, dan kontrak array. Metadata serta protokol beku dipertahankan sebagai rekam eksperimen; keputusan penggunaan data terbaru tercatat pada [keputusan klien](docs/konfirmasi_dataset_klien.md). Manifest, scaler, feature order, statistik split, laporan clipping, audit, dan hasil QA juga tersedia. Notebook model wajib memakai hasil ini tanpa split atau fit scaler ulang. Jika sumber dikoreksi klien, regenerasi seluruh preprocessing sebelum eksperimen.

Backbone memakai Global Average Pooling dan head Linear 128/64 yang sama. Feature extraction membekukan backbone ImageNet. FT baru memakai ImageNet dan head baru tanpa checkpoint FE, membuka layer4 ResNet50 atau block5 VGG16/VGG19; Batch Normalization ResNet50 tetap eval/beku pada FE dan FT. FT dari checkpoint FE hanya berlaku pada hasil historis. [TorchVision ResNet50](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet50.html) memakai varian V1.5. Checkpoint `.pt` disimpan sebagai state_dict dan dimuat dengan weights_only=True. Konfigurasi ada di `src/config.py`.

## Evaluasi dan keterbatasan

Pemilihan model memakai validation. Test dipakai setelah konfigurasi dibekukan. Metrik utama adalah MAE, MSE, RMSE, dan R2 keseluruhan serta per posisi, disertai jumlah sampel. Prediksi negatif dilaporkan; tidak dilakukan clipping diam-diam. DummyRegressor median dan Ridge menjadi baseline tambahan. Plot hasil eksperimen disimpan minimal 300 DPI. Hasil satu split dan seed ini tidak mengukur variasi antar-run; CI dan uji signifikansi tidak dihitung.

Dataset berasal dari 411 ID, tetapi ID berbeda belum menjamin observasi independen. Audit menemukan 160 sampel dengan fitur tiga sensor identik; lima kelompok salinan memiliki target berbeda. Fitur identik, target berbeda antar-ID, dan AHI ekstrem tetap dipertahankan sesuai arahan klien. Temuan ini tidak membuktikan oversampling atau validitas klinis label. Generalisasi di luar dataset, manfaat ImageNet pada pseudo-image, serta manfaat fine-tuning harus dibuktikan melalui eksperimen. Evaluasi internal tidak membuktikan kesiapan klinis.

## Troubleshooting

- Workbook tidak ditemukan: jalankan dari akar proyek dan periksa tiga nama file dalam `data/`.
- Schema, AHI, atau kunci tidak konsisten: baca audit, periksa sumber, dan hentikan tahap terkait; jangan mengoreksi target secara otomatis.
- Import paket gagal: pastikan interpreter/kernel sesuai environment dan jalankan `python -m pip check`.
- CUDA tidak terdeteksi: periksa interpreter `.venv`, build torch yang memuat CUDA, dan driver melalui `nvidia-smi`; jangan mengubah driver secara otomatis.
- VRAM habis: periksa penggunaan GPU dan cleanup antar kandidat. Batch 8 serta input 224 x 224 tetap dalam protokol tuning; perubahan memerlukan pencatatan protokol dan smoke ulang sebelum full training.
