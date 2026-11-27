# PROJECT: PREDIKSI AHI OBSTRUCTIVE SLEEP APNEA MENGGUNAKAN TRANSFER LEARNING RESNET50, VGG16, DAN VGG19

## 1. Role

Bertindaklah sebagai Senior Machine Learning Engineer, Deep Learning Engineer, Data Scientist, dan Research Engineer yang berpengalaman dalam:

- Python dan Jupyter Notebook.
- TensorFlow dan Keras.
- Deep Learning Regression.
- Transfer Learning dan Fine-tuning.
- Tabular-to-Image Transformation.
- Multi-Sensor Data Fusion.
- Biomedical Signal Feature Processing.
- Experimental Design.
- Evaluasi model Machine Learning untuk penelitian akademik.

Bantu saya membangun project penelitian yang lengkap, modular, reproducible, mudah dipahami, dan dapat dipertanggungjawabkan secara ilmiah.

Project ini dikerjakan untuk klien penelitian akademik. Oleh karena itu, jangan hanya berfokus pada performa model, tetapi juga pada validitas metodologi, kualitas implementasi, dan penjelasan setiap tahapan.

## 2. Latar Belakang Project

Penelitian bertujuan melakukan penilaian atau prediksi nilai Apnea-Hypopnea Index (AHI) pada kondisi Obstructive Sleep Apnea (OSA) berdasarkan posisi tidur pasien dengan menggunakan tiga jenis sensor:

1. Flow.
2. Pulse Rate.
3. Plethysmography (Pleth).

Data yang tersedia berupa data tabular numerik hasil ekstraksi fitur Intrinsic Mode Function (IMF), bukan citra atau sinyal mentah.

Klien secara khusus meminta penggunaan tiga arsitektur:

- ResNet50.
- VGG16.
- VGG19.

**Transfer learning harus menjadi fokus dan kekuatan utama penelitian ini.**

Data tabular perlu ditransformasikan menjadi representasi citra yang kompatibel dengan pretrained Convolutional Neural Network (CNN).

Permasalahan penelitian adalah REGRESI, bukan klasifikasi.

Target prediksi adalah nilai AHI numerik kontinu berdasarkan pasien dan posisi tidur.

Jangan mengubah target AHI menjadi label Normal, Mild, Moderate, atau Severe untuk training model utama.

## 3. Dataset

Gunakan hanya tiga dataset terbaru berikut:

- Flow Patient.xlsx
- PulseRate.xlsx
- Pleth.xlsx

Jangan menggunakan dataset lama dengan nama yang mengandung "_smote".

Masing-masing file memiliki tiga sheet:

- Supine: posisi tidur telentang.
- Left: posisi tidur miring kiri.
- Right: posisi tidur miring kanan.

Struktur kolom dataset:

1. id
2. Posisi
3. IMF
4. Energi IMF
5. Energi Relatif (%)
6. Frekuensi Dominan (Hz)
7. Rentang Frek (Hz)
8. Mean Period (s)
9. Mean Amplitudo
10. STD Amplitudo
11. Instant Freq (Hz)
12. STD Instant Freq (Hz)
13. Entropi
14. Korelasi dengan sinyal asli
15. AHI

Terdapat 11 fitur numerik hasil ekstraksi IMF.

Perhatikan kemungkinan whitespace pada nama kolom, seperti "AHI " pada salah satu sheet. Lakukan standardisasi nama kolom sebelum pengolahan.

Berdasarkan pemeriksaan awal:

| Dataset | Jumlah baris | ID unik |
|---|---:|---:|
| Flow | 19.632 | 411 |
| Pulse Rate | 14.700 | 411 |
| Pleth | 16.417 | 411 |

Jumlah pasangan pasien dan posisi tidur yang memiliki ketiga sensor:

- Supine: 411.
- Left: 330.
- Right: 350.
- Total: 1.091 sampel.

Jumlah pasangan pasien–posisi pada Flow dan Pleth masing-masing adalah 1.107. Sebanyak 16 pasangan tidak memiliki data Pulse Rate.

Indeks IMF tersedia dari IMF1 hingga IMF50, tetapi jumlah IMF untuk setiap pasien, posisi, dan sensor tidak selalu sama.

Nilai AHI untuk pasien dan posisi tidur yang sama konsisten pada ketiga sensor. Namun, nilai AHI pasien dapat berbeda pada posisi tidur yang berbeda.

Terdapat nilai AHI ekstrem hingga 600.

Seluruh informasi tersebut merupakan hasil pemeriksaan awal. Lakukan validasi ulang menggunakan kode, jangan melakukan hardcoding jumlah data sebagai hasil analisis.

Pembaruan arahan klien, 27 November 2026: sertakan seluruh 411 ID dengan data dan target asli. Catat temuan fitur identik dan jangan mengasumsikan data bebas dari duplikasi sintetis.

## 4. Tujuan Penelitian

Bangun tiga arsitektur deep learning regression berbasis transfer learning untuk memprediksi nilai AHI menggunakan gabungan tiga sensor.

Tujuan eksperimen:

1. Mengolah dan menggabungkan fitur dari tiga sensor.
2. Mengubah data tabular menjadi representasi citra.
3. Mengimplementasikan transfer learning menggunakan pretrained ImageNet.
4. Melakukan feature extraction.
5. Melakukan fine-tuning.
6. Mengukur kontribusi transfer learning terhadap performa regresi.
7. Membandingkan performa ResNet50, VGG16, dan VGG19.
8. Mengevaluasi performa prediksi AHI secara keseluruhan.
9. Mengevaluasi performa prediksi AHI berdasarkan posisi tidur Supine, Left, dan Right.

Gunakan satu model gabungan untuk ketiga posisi tidur pada setiap arsitektur dan tampilkan evaluasi tambahan berdasarkan posisi tidur.

Tidak perlu membuat model terpisah untuk setiap posisi tidur.

## 5. Environment dan Hardware

Project dijalankan secara lokal menggunakan:

- Visual Studio Code.
- Jupyter Notebook (.ipynb).
- Python.
- GPU dengan VRAM 8 GB.

Framework dan library utama:

- TensorFlow/Keras.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib.
- Seaborn.
- SciPy.
- OpenPyXL.
- Joblib.
- Jupyter.
- Nbformat.

Sebelum membuat environment:

1. Periksa operating system.
2. Periksa versi Python.
3. Periksa GPU menggunakan nvidia-smi apabila tersedia.
4. Periksa driver GPU dan kompatibilitas TensorFlow.
5. Periksa apakah TensorFlow dapat mendeteksi GPU.
6. Pilih versi package yang kompatibel berdasarkan environment aktual.
7. Jika menggunakan Windows dan TensorFlow modern, pertimbangkan WSL2 apabila GPU tidak didukung secara native.
8. Jangan menginstal ulang driver atau mengubah konfigurasi sistem tanpa persetujuan.
9. Jangan mengasumsikan GPU tersedia sebelum melakukan pemeriksaan.

Optimalkan penggunaan VRAM dengan:

- GPU memory growth jika didukung.
- Mixed precision jika hardware mendukung.
- Batch size awal 8.
- Batch size yang configurable.
- TensorFlow data pipeline.
- Prefetch yang sesuai kapasitas memori.
- Early stopping.
- Model checkpoint.
- Clear session dan garbage collection setelah eksperimen.

Apabila terjadi Out of Memory, turunkan batch size secara bertahap.

Untuk mixed precision, pastikan output regresi dan perhitungan yang membutuhkan stabilitas numerik menggunakan float32.

## 6. Struktur Project

Buat struktur project berikut dengan tetap menyesuaikan kondisi workspace yang sudah tersedia.

```text
OSA_Transfer_Learning/
│
├── data/
│   ├── Flow Patient.xlsx
│   ├── PulseRate.xlsx
│   └── Pleth.xlsx
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_ResNet50.ipynb
│   ├── 03_VGG16.ipynb
│   ├── 04_VGG19.ipynb
│   └── 05_Model_Comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model_utils.py
│   └── evaluation.py
│
├── artifacts/
│   ├── processed_data/
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
├── README.md
├── AGENTS.md
├── requirements.txt
└── .gitignore
```

Gunakan pathlib untuk pengelolaan path.

Hindari absolute path yang bergantung pada komputer tertentu.

Jangan menghapus atau mengubah file dataset asli.

Notebook harus dapat dijalankan berdasarkan urutan yang ditentukan.

Simpan keputusan metodologi dan aturan project dalam AGENTS.md agar tetap konsisten pada setiap sesi pengembangan.

Jangan mengunggah data pasien, ID, atau file penelitian ke layanan eksternal tanpa izin. Masukkan data mentah dan artifacts yang mengandung informasi pasien ke dalam .gitignore.

## 7. Notebook 01 — Preprocessing

Buat notebook preprocessing yang lengkap dan memiliki penjelasan Markdown pada setiap tahapan.

### 7.1 Data Loading

- Membaca ketiga file Excel.
- Membaca seluruh sheet Supine, Left, dan Right.
- Menggabungkan sheet pada setiap sensor.
- Menambahkan identitas sensor.
- Menyeragamkan nama kolom.
- Mempertahankan ID sebagai string.
- Memvalidasi tipe data.
- Memvalidasi kesesuaian nama sheet dengan kolom Posisi.

### 7.2 Data Understanding dan Exploratory Data Analysis

Lakukan analisis:

- Jumlah baris dan kolom.
- Jumlah pasien unik.
- Jumlah pasangan pasien dan posisi tidur.
- Missing values.
- Duplicate values.
- Nilai infinite dan invalid.
- Statistik deskriptif.
- Distribusi AHI.
- Distribusi AHI berdasarkan posisi tidur.
- Distribusi fitur setiap sensor.
- Korelasi antarfitur.
- Distribusi jumlah IMF.
- Identifikasi nilai ekstrem.
- Konsistensi AHI dalam kelompok pasien–posisi.
- Konsistensi AHI antar sensor.
- Jumlah sampel yang tersedia dan tidak tersedia pada setiap sensor.

Gunakan visualisasi yang informatif dan profesional.

AHI ekstrem, termasuk nilai 600, harus dilaporkan dan ditandai. Jangan menghapus atau melakukan clipping terhadap target tanpa dasar yang jelas.

### 7.3 Multi-Sensor Fusion

Gunakan pasangan kolom berikut sebagai identitas sampel:

- id
- Posisi

Satu pasangan id dan Posisi merepresentasikan satu sampel dengan satu nilai target AHI.

Jangan menganggap setiap baris IMF sebagai sampel independen.

Gunakan pendekatan complete-case pada eksperimen utama, yaitu hanya menggunakan pasangan pasien–posisi yang memiliki ketiga sensor.

Catat dan laporkan jumlah sampel yang dikeluarkan karena ketidaklengkapan sensor.

Jangan langsung melakukan merge ketiga tabel pada id dan Posisi tanpa pengelompokan IMF karena dapat menyebabkan Cartesian product.

Bangun representasi masing-masing sensor terlebih dahulu sebelum menggabungkannya.

Validasi bahwa nilai AHI konsisten dalam setiap kelompok dan antar sensor. Apabila ditemukan inkonsistensi, laporkan dan hentikan tahapan terkait sampai masalah ditangani. Jangan merata-ratakan target yang berbeda secara otomatis.

### 7.4 Data Splitting

Data splitting harus dilakukan berdasarkan ID pasien, bukan berdasarkan baris IMF atau pasangan pasien–posisi secara acak.

Gunakan rasio awal:

- Training: 70%.
- Validation: 15%.
- Testing: 15%.

Gunakan random seed 42.

Gunakan group-based splitting seperti GroupShuffleSplit atau pendekatan lain yang menjamin pemisahan ID pasien.

Pastikan satu pasien tidak muncul pada lebih dari satu subset, termasuk ketika pasien tersebut memiliki beberapa posisi tidur.

Buat assertion untuk memvalidasi:

- Tidak ada overlap ID antara training dan validation.
- Tidak ada overlap ID antara training dan testing.
- Tidak ada overlap ID antara validation dan testing.

Laporkan distribusi jumlah pasien, sampel, posisi tidur, dan AHI pada setiap subset.

Simpan split manifest agar seluruh model menggunakan pembagian data yang sama.

Jangan menggunakan test set untuk pemilihan model, tuning hyperparameter, penentuan preprocessing, atau early stopping.

### 7.5 Feature Engineering

Gunakan 11 fitur numerik IMF sebagai fitur utama.

Kolom berikut tidak boleh menjadi fitur numerik yang dimasukkan secara langsung ke dalam matriks sensor:

- id.
- AHI.
- Posisi.
- IMF.

Gunakan id hanya sebagai identitas dan dasar grouping.

Gunakan IMF sebagai indeks baris matriks.

Gunakan Posisi sebagai input kategorikal terpisah melalui one-hot encoding:

- Supine.
- Left.
- Right.

Jangan melakukan SMOTE karena target merupakan nilai regresi kontinu.

### 7.6 Feature Scaling

Lakukan scaling secara terpisah untuk setiap fitur pada setiap sensor.

Gunakan metode yang sesuai dengan karakteristik distribusi data, seperti robust scaling atau percentile-based scaling.

Jika menggunakan percentile-based scaling, tentukan batas persentil dari training set, misalnya persentil 1 dan 99, kemudian petakan fitur ke rentang 0–1.

Seluruh scaler, parameter imputasi, dan transformasi yang dipelajari dari data harus di-fit hanya pada training set.

Validation dan testing hanya menggunakan transformasi yang telah dipelajari dari training set.

Laporkan dampak clipping atau transformasi terhadap fitur.

Simpan parameter preprocessing agar dapat digunakan kembali.

Jangan melakukan normalisasi target AHI jika tidak diperlukan. Jika target scaling digunakan, lakukan fitting hanya pada training set dan inverse transform sebelum evaluasi.

### 7.7 Tabular-to-Image Transformation

Implementasikan pendekatan structured feature matrix sebagai representasi citra.

Untuk setiap pasangan pasien–posisi dan setiap sensor:

- Urutkan IMF secara numerik, bukan alfabetis.
- IMF1, IMF2, IMF3, dan seterusnya hingga IMF50.
- Gunakan IMF sebagai sumbu baris.
- Gunakan 11 fitur numerik sebagai sumbu kolom.
- Gunakan urutan fitur yang sama pada seluruh sensor.
- Buat matriks berukuran 50 × 11.
- Tangani IMF yang tidak tersedia menggunakan padding yang konsisten.
- Jangan melakukan interpolasi untuk mengarang nilai IMF yang tidak tersedia.
- Bedakan padding dari nilai fitur asli, misalnya dengan menggunakan nilai padding khusus pada encoding.
- Simpan mask atau informasi keberadaan IMF untuk keperluan audit.

Gabungkan ketiga matriks sebagai representasi tiga channel:

- Channel 1: Flow.
- Channel 2: Pulse Rate.
- Channel 3: Pleth.

Hasil awal memiliki dimensi:

50 × 11 × 3

Kemudian sesuaikan ukuran input menjadi:

224 × 224 × 3

Gunakan metode resizing yang tidak mencampurkan fitur secara berlebihan, misalnya nearest-neighbor sebagai konfigurasi awal.

Jadikan ukuran citra configurable untuk menyesuaikan kebutuhan GPU.

Gunakan skala piksel yang sesuai dengan preprocessing pretrained model Keras.

Jangan menerapkan normalisasi dua kali secara tidak sengaja.

Tampilkan visualisasi representasi citra dari setiap sensor dan hasil penggabungannya.

Jelaskan bahwa representasi tersebut merupakan feature image atau pseudo-image, bukan citra fisiologis asli.

Jelaskan juga bahwa kesamaan indeks IMF antar sensor tidak otomatis membuktikan kesamaan komponen frekuensi atau sinkronisasi fisiologis. Penggabungan ini merupakan pendekatan representasi yang perlu divalidasi melalui eksperimen.

Jangan menyebut metode tersebut sebagai DeepInsight, GAF, MTF, atau spectrogram apabila tidak benar-benar mengimplementasikan metode terkait.

### 7.8 Save Processed Dataset

Simpan hasil preprocessing:

- Input citra untuk training, validation, dan testing.
- Input one-hot posisi tidur.
- Target AHI.
- ID pasien.
- Posisi tidur.
- Split manifest.
- Parameter scaling.
- Feature order.
- Dataset metadata.
- Data audit report.

Gunakan format yang efisien seperti NPZ, NPY, CSV, JSON, atau Joblib sesuai kebutuhan.

Pertahankan keterkaitan setiap sampel dengan metadata, target, dan posisi tidurnya.

Notebook model harus menggunakan hasil preprocessing yang sama tanpa melakukan splitting ulang.

## 8. Notebook 02, 03, dan 04 — Deep Learning Regression

Buat tiga notebook model:

- 02_ResNet50.ipynb.
- 03_VGG16.ipynb.
- 04_VGG19.ipynb.

Gunakan API resmi TensorFlow/Keras untuk masing-masing arsitektur.

Gunakan:

- include_top=False.
- weights="imagenet" untuk eksperimen transfer learning.
- Input shape sesuai hasil preprocessing.

Gunakan preprocessing input resmi yang sesuai untuk setiap pretrained backbone.

### 8.1 Model Architecture

Gunakan pendekatan functional API dengan dua input:

Input 1:

- Representasi citra tiga sensor.

Input 2:

- One-hot encoding posisi tidur.

Rancangan arsitektur:

```text
Sensor Feature Image
        |
Pretrained CNN Backbone
        |
Global Average Pooling
        |
        +-----------------+
                          |
Sleep Position Input -----+
                          |
                     Concatenate
                          |
                     Dense (128)
                          |
                     Dropout
                          |
                     Dense (64)
                          |
                     Dropout
                          |
                     Dense (1)
                          |
                     Predicted AHI
```

Gunakan activation linear dan dtype float32 pada output regresi.

Jelaskan bahwa prediksi negatif secara fisiologis tidak masuk akal. Laporkan jika terdapat prediksi negatif dan jangan melakukan clipping diam-diam.

Gunakan Global Average Pooling, bukan Flatten yang menghasilkan terlalu banyak parameter.

Gunakan regression head yang sama pada seluruh arsitektur agar perbandingan lebih konsisten.

### 8.2 Training Configuration

Konfigurasi awal:

- Optimizer: Adam.
- Loss: Mean Squared Error sebagai baseline utama.
- Batch size: 8.
- Random seed: 42.
- Initial learning rate: 1e-3.
- Fine-tuning learning rate: 1e-5.
- Feature extraction epochs: maksimal 50.
- Fine-tuning epochs: maksimal 50.
- Early stopping patience: 10.
- Restore best weights: True.
- ReduceLROnPlateau.
- ModelCheckpoint.
- TerminateOnNaN.

Seluruh konfigurasi harus dapat diubah melalui config.py.

Jika ingin menggunakan Huber Loss sebagai alternatif karena terdapat target ekstrem, lakukan pemilihan hanya berdasarkan validation set dan terapkan protokol yang konsisten pada seluruh arsitektur.

### 8.3 Experiment A — Training From Scratch

Implementasikan model menggunakan:

- weights=None.
- Random weight initialization.
- Seluruh backbone dapat dilatih.
- Regression head yang sama.
- Dataset split yang sama.

Gunakan skenario ini sebagai pembanding untuk mengetahui apakah pretrained weights memberikan manfaat.

Terapkan anggaran training dan tuning yang wajar serta dokumentasikan jumlah parameter trainable, epoch, dan waktu training.

### 8.4 Experiment B — Transfer Learning: Feature Extraction

Implementasikan:

- Pretrained ImageNet weights.
- Freeze seluruh backbone.
- Training hanya pada regression head.
- Validation monitoring.
- Penyimpanan best checkpoint.

Tampilkan:

- Total parameters.
- Trainable parameters.
- Non-trainable parameters.
- Training loss.
- Validation loss.
- MAE dan RMSE.
- Waktu training.

Jelaskan bagaimana pretrained backbone dimanfaatkan sebagai feature extractor tanpa melatih ulang seluruh jaringan.

### 8.5 Experiment C — Transfer Learning: Fine-tuning

Mulai dari checkpoint terbaik Experiment B, bukan dari random initialization.

Implementasikan:

- Unfreeze sebagian layer akhir pretrained backbone.
- Gunakan learning rate rendah.
- Compile ulang model setelah mengubah trainable layers.
- Pertahankan Batch Normalization pada inference mode ketika diperlukan, khususnya ResNet50.
- Hindari membuka seluruh layer secara langsung.
- Gunakan regularization dan early stopping.
- Simpan checkpoint terbaik fine-tuning.

Untuk VGG16 dan VGG19, pertimbangkan membuka convolutional block terakhir sebagai konfigurasi awal.

Untuk ResNet50, pertimbangkan membuka residual stage terakhir dengan perlakuan Batch Normalization yang tepat.

Gunakan validation set untuk menentukan apakah hasil fine-tuning memberikan peningkatan dibandingkan feature extraction.

Jangan mengklaim fine-tuning pasti meningkatkan performa.

### 8.6 Fair Comparison

Setiap arsitektur memiliki tiga skenario:

1. Training from scratch.
2. Transfer learning dengan feature extraction.
3. Transfer learning dengan fine-tuning.

Total terdapat sembilan kondisi model yang dievaluasi.

Ini bukan sembilan model berdasarkan posisi tidur, melainkan sembilan kondisi eksperimen untuk membuktikan kontribusi transfer learning.

Gunakan pembagian dataset, fitur, regression head, metrik, dan protokol evaluasi yang konsisten.

Simpan checkpoint dan training history untuk setiap skenario.

Catat keterbatasan perbandingan apabila terdapat perbedaan jumlah parameter trainable atau anggaran training.

## 9. Evaluasi Model

Gunakan metrik regresi utama:

- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- Coefficient of Determination (R²).

Opsional:

- Median Absolute Error.
- Pearson correlation.
- Spearman correlation.

Jangan menggunakan accuracy, precision, recall, F1-score, atau confusion matrix sebagai evaluasi utama regresi.

Jangan menggunakan MAPE tanpa penanganan yang benar karena dataset memiliki nilai AHI nol.

Evaluasi menggunakan validation set selama pengembangan.

Gunakan test set setelah konfigurasi model dibekukan.

Laporkan performa:

- Keseluruhan test set.
- Posisi Supine.
- Posisi Left.
- Posisi Right.

Laporkan jumlah sampel pada setiap posisi.

Apabila menghitung confidence interval, gunakan bootstrap berbasis kelompok pasien, bukan bootstrap baris secara independen.

Visualisasi:

1. Training dan validation loss curves.
2. Training dan validation MAE curves.
3. Actual vs predicted scatter plot.
4. Residual plot.
5. Error distribution histogram.
6. Prediksi AHI berdasarkan posisi tidur.
7. Bland–Altman plot sebagai analisis kesepakatan tambahan, dengan catatan terdapat pengukuran berulang per pasien.

Simpan seluruh metrik dalam format CSV atau JSON.

Simpan hasil prediksi dengan kolom:

- Patient ID.
- Sleep Position.
- Actual AHI.
- Predicted AHI.
- Residual.
- Absolute Error.
- Model.
- Experiment.

Jangan merekayasa angka performa. Jangan menghapus prediksi yang buruk untuk meningkatkan metrik.

## 10. Notebook 05 — Model Comparison

Buat notebook khusus untuk membandingkan seluruh hasil eksperimen.

Muat metrik, training history, dan hasil prediksi dari ketiga notebook model.

Tidak perlu melakukan training ulang seluruh CNN di notebook ini.

### 10.1 Baseline Tambahan

Untuk memberikan konteks ilmiah, tambahkan baseline sederhana:

- DummyRegressor menggunakan median training.
- Ridge Regression atau Random Forest Regressor.

Gunakan representasi fitur tabular dari sampel pasien–posisi yang sama, termasuk informasi posisi tidur dan split pasien yang sama.

Tempatkan baseline tabular sebagai pembanding tambahan, bukan fokus utama penelitian.

### 10.2 Model Performance Comparison

Tampilkan tabel:

| Model | Experiment | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| ResNet50 | Scratch | | | | |
| ResNet50 | Feature Extraction | | | | |
| ResNet50 | Fine-tuning | | | | |
| VGG16 | Scratch | | | | |
| VGG16 | Feature Extraction | | | | |
| VGG16 | Fine-tuning | | | | |
| VGG19 | Scratch | | | | |
| VGG19 | Feature Extraction | | | | |
| VGG19 | Fine-tuning | | | | |

Tambahkan tabel pembanding baseline tabular.

### 10.3 Transfer Learning Contribution

Bandingkan:

- Scratch vs Feature Extraction.
- Scratch vs Fine-tuning.
- Feature Extraction vs Fine-tuning.

Hitung perubahan absolut dan persentase penurunan MAE atau RMSE jika denominator valid.

Analisis apakah pretrained weights:

- Mengurangi prediction error.
- Mempercepat convergence.
- Mengurangi overfitting.
- Mengurangi jumlah parameter yang dilatih.
- Memberikan generalisasi lebih baik.

Jangan mengklaim keberhasilan transfer learning apabila hasil eksperimen tidak mendukung klaim tersebut.

### 10.4 Position-Based Evaluation

Bandingkan performa masing-masing arsitektur pada:

- Supine.
- Left.
- Right.

Buat tabel dan visualisasi perbandingan MAE, RMSE, dan R² berdasarkan posisi tidur.

Analisis posisi tidur dengan prediction error terbesar dan terkecil.

Tidak perlu membuat sembilan model terpisah berdasarkan posisi tidur.

### 10.5 Visualizations

Buat:

- Bar chart perbandingan MAE.
- Bar chart perbandingan RMSE.
- Bar chart perbandingan R².
- Perbandingan training time.
- Perbandingan trainable parameters.
- Perbandingan training dan validation curves.
- Heatmap ringkasan metrik.
- Actual vs predicted comparison.
- Position-based performance comparison.

Gunakan desain visual yang konsisten dan cocok untuk penelitian akademik.

Simpan visualisasi dengan resolusi minimal 300 DPI.

### 10.6 Research Discussion

Berikan pembahasan berdasarkan hasil eksperimen aktual mengenai:

- Model dengan performa terbaik berdasarkan protokol pemilihan validation set.
- Kontribusi transfer learning.
- Dampak fine-tuning.
- Perbedaan antararsitektur.
- Dampak posisi tidur.
- Overfitting dan generalization.
- Keterbatasan jumlah pasien.
- Domain gap antara ImageNet dan feature image fisiologis.
- Keterbatasan transformasi tabular-to-image.
- Pengaruh data ekstrem.
- Temuan fitur identik antar-ID dan keputusan klien untuk mempertahankan seluruh 411 ID.
- Keterbatasan generalisasi terhadap pasien di luar dataset.

Jangan mengklaim penelitian ini merupakan penelitian pertama tanpa kajian literatur yang memadai.

Jangan menyimpulkan bahwa model telah tervalidasi untuk penggunaan klinis hanya berdasarkan evaluasi internal.

## 11. Notebook Writing Standards

Seluruh notebook harus memiliki:

- Judul yang jelas.
- Tujuan notebook.
- Daftar tahapan.
- Penjelasan Markdown sebelum tahapan penting.
- Penjelasan konsep dan alasan pemilihan metode.
- Kode yang terstruktur.
- Komentar yang relevan.
- Visualisasi.
- Interpretasi hasil berdasarkan output aktual.
- Kesimpulan notebook.

Gunakan Bahasa Indonesia yang formal, mudah dipahami, dan tetap mempertahankan terminologi teknis Machine Learning yang tepat.

Gunakan Bahasa Inggris untuk nama variabel, fungsi, dan class.

Hindari:

- Kode yang tidak diperlukan.
- Pengulangan fungsi antar notebook.
- Hardcoded metrics.
- Hardcoded absolute paths.
- Dummy results.
- Placeholder yang dianggap sebagai hasil eksperimen.
- Data leakage.
- Manipulasi target.
- Klaim performa yang tidak didukung eksperimen.

Notebook harus cukup jelas untuk dipahami oleh pengguna yang belum ahli dalam Machine Learning.

## 12. Reproducibility dan Quality Assurance

Pastikan:

- Random seed dikendalikan.
- Dataset split disimpan.
- Preprocessing hanya di-fit pada training set.
- Tidak ada overlap ID pasien.
- AHI tidak menjadi input model.
- Input dan target tetap selaras.
- Input sensor dan posisi tidur memiliki jumlah sampel yang sama.
- Seluruh input memiliki shape yang benar.
- Tidak terdapat NaN atau infinite values.
- Model dapat disimpan dan dimuat kembali.
- Prediksi dari model yang dimuat konsisten.
- Metrik dapat dihitung ulang dari file prediksi.
- Notebook tidak bergantung pada state tersembunyi dari notebook sebelumnya.
- Seluruh eksperimen memiliki log konfigurasi dan training history.

Lakukan smoke test sebelum full training.

Smoke test cukup menggunakan sejumlah kecil batch dan epoch untuk memastikan pipeline berjalan.

Jangan menganggap hasil smoke test sebagai hasil penelitian.

Dokumentasikan versi library, GPU, konfigurasi training, serta keputusan metodologi.

## 13. README dan Dokumentasi

Buat README.md yang menjelaskan:

- Latar belakang penelitian.
- Tujuan project.
- Struktur dataset.
- Struktur folder.
- Environment setup.
- Cara mengaktifkan GPU.
- Cara menjalankan notebook.
- Urutan notebook.
- Metodologi preprocessing.
- Multi-sensor fusion.
- Tabular-to-image transformation.
- Transfer learning.
- Fine-tuning.
- Evaluasi regresi.
- Lokasi output.
- Troubleshooting.
- Keterbatasan penelitian.

README harus memungkinkan pengguna menjalankan project pada komputer lain setelah menyediakan dataset dan environment yang sesuai.

## 14. Aturan Pelaksanaan

Kerjakan project secara bertahap.

Jangan membuat dan menjalankan seluruh notebook sekaligus tanpa memvalidasi tahap sebelumnya.

Gunakan urutan berikut:

**PHASE 0 — Project Initialization**

- Inspeksi workspace.
- Inspeksi environment.
- Inspeksi GPU.
- Audit ketiga dataset.
- Buat project structure.
- Buat AGENTS.md.
- Buat config.py.
- Buat requirements.txt berdasarkan environment aktual.
- Laporkan hasil pemeriksaan.

**PHASE 1 — Preprocessing**

- Implementasikan preprocessing.py.
- Buat 01_Preprocessing.ipynb.
- Jalankan notebook.
- Validasi hasil sensor fusion.
- Validasi patient-based splitting.
- Validasi transformasi tabular-to-image.
- Simpan processed dataset.
- Laporkan hasil dan keterbatasan.

**PHASE 2 — ResNet50**

- Buat model_utils.py dan evaluation.py.
- Implementasikan ResNet50.
- Lakukan smoke test.
- Periksa penggunaan GPU dan VRAM.
- Jalankan eksperimen setelah disetujui.
- Simpan model dan hasil evaluasi.

**PHASE 3 — VGG16**

- Gunakan pipeline yang sudah tervalidasi.
- Implementasikan VGG16.
- Jalankan smoke test dan eksperimen.
- Simpan hasil.

**PHASE 4 — VGG19**

- Gunakan pipeline yang sama.
- Implementasikan VGG19.
- Jalankan smoke test dan eksperimen.
- Simpan hasil.

**PHASE 5 — Model Comparison**

- Gabungkan seluruh hasil.
- Implementasikan baseline tambahan.
- Buat visualisasi dan pembahasan.
- Finalisasi README.
- Lakukan quality assurance terhadap seluruh project.

Setelah menyelesaikan setiap phase:

1. Laporkan pekerjaan yang telah selesai.
2. Laporkan file yang dibuat atau diubah.
3. Laporkan hasil pengujian aktual.
4. Laporkan masalah yang ditemukan.
5. Laporkan keputusan metodologi.
6. Berikan rekomendasi tahapan selanjutnya.

Tunggu persetujuan saya sebelum melanjutkan ke phase berikutnya.

Jangan melakukan full training seluruh model secara otomatis.

## 15. Instruksi Pengerjaan Sekarang

Mulai dari PHASE 0 saja.

Periksa workspace, ketiga dataset terbaru, environment Python, dan ketersediaan GPU.

Buat struktur dasar project dan dokumentasikan keseluruhan rencana pada AGENTS.md.

Lakukan audit dataset secara aktual.

Jika terdapat perbedaan antara kondisi aktual dan informasi dalam prompt ini, prioritaskan hasil pemeriksaan aktual dan laporkan perbedaannya.

Jangan mulai training model.

Setelah PHASE 0 selesai, tampilkan ringkasan hasil pemeriksaan dan rencana implementasi PHASE 1.