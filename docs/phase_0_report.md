# Laporan PHASE 0

Tanggal pemeriksaan: 26 November 2026, zona waktu Asia/Jakarta.

## Pekerjaan yang selesai

Workspace, ketiga workbook terbaru, Python, dependensi, GPU, CUDA, dan status WSL telah diperiksa. Struktur dasar proyek, konfigurasi bersama, skrip audit yang dapat diulang, requirements, panduan agen, README, serta pengujian audit telah dibuat.

Akar workspace yang sudah memiliki `data/` dipertahankan. Tidak dibuat salinan dataset atau folder proyek bersarang. Belum dibuat notebook, split manifest training, scaler terlatih, processed dataset, atau model. Tidak ada training dan tidak ada perubahan driver maupun konfigurasi sistem.

## Hasil audit dataset aktual

| Sensor | Baris IMF | Kolom | ID unik | Pasangan pasien-posisi | Supine | Left | Right |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flow | 19.632 | 15 | 411 | 1.107 | 411 | 339 | 357 |
| Pulse Rate | 14.700 | 15 | 411 | 1.091 | 411 | 330 | 350 |
| Pleth | 16.417 | 15 | 411 | 1.107 | 411 | 339 | 357 |

Hasil ini dihitung dari file aktual dan sesuai jumlah awal dalam brief. Setiap workbook memiliki sheet Supine, Left, dan Right. Terdapat 11 fitur numerik dan empat kolom metadata/target.

Gabungan complete-case mencakup 1.091 sampel dari 411 ID: Supine 411, Left 330, dan Right 350. Dari gabungan 1.107 pasangan yang tersedia, 16 pasangan tidak memiliki Pulse Rate: sembilan Left dan tujuh Right. Flow dan Pleth memiliki semua pasangan dalam gabungan tersebut. Jumlah ini belum merupakan hasil split.

### Integritas dan indeks IMF

- Missing value, numerik invalid, dan nilai infinite: nol pada ketiga sensor.
- Duplikasi seluruh baris dan duplikasi kunci `(id, Posisi, IMF)`: nol.
- Ketidaksesuaian sheet dan kolom Posisi: nol.
- Konflik AHI dalam satu pasien-posisi dan konflik antar sensor: nol.
- Header `AHI ` pada Flow/Right memiliki trailing whitespace. Audit menormalkannya di memori; file sumber tetap sama.
- ID dibaca sebagai string. Semua indeks IMF valid dalam rentang 1 sampai 50, tanpa lubang indeks internal.
- SHA-256 setiap workbook tetap sama sebelum dan sesudah audit.

| Sensor | IMF minimum per sampel | Median | Maksimum |
|---|---:|---:|---:|
| Flow | 3 | 18 | 50 |
| Pulse Rate | 1 | 13 | 50 |
| Pleth | 1 | 16 | 50 |

Jumlah IMF bervariasi antar sampel dan sensor. PHASE 1 memerlukan padding dan mask; indeks IMF yang tidak tersedia tidak boleh diinterpolasi.

### AHI dan fitur ekstrem

Statistik berikut menghitung satu AHI per sampel complete-case, bukan mengulangnya untuk setiap baris IMF.

| Posisi | Sampel | Minimum | Median | Maksimum | AHI nol | AHI > 100 |
|---|---:|---:|---:|---:|---:|---:|
| Keseluruhan | 1.091 | 0 | 13,3416 | 600 | 180 | 11 |
| Supine | 411 | 0 | 21,0663 | 120,6239 | 1 | 3 |
| Left | 330 | 0 | 4,4944 | 200 | 115 | 5 |
| Right | 350 | 0 | 9,8961 | 600 | 64 | 3 |

Nilai 600 muncul pada satu sampel Right. Batas >100 hanya penanda audit yang configurable, bukan ambang diagnosis atau alasan penghapusan. Target tidak diubah. Validitas dan satuan target ekstrem perlu dikonfirmasi melalui sumber data klien.

Rentang fitur lebar: energi IMF maksimum sekitar 7,748e8 pada Flow, 2,512e9 pada Pulse Rate, dan 3,961e8 pada Pleth. Mean Period mencapai sekitar 26.374 detik pada Flow, 23.289 detik pada Pulse Rate, dan 25.748 detik pada Pleth. Semua finite; angka besar tidak otomatis dianggap salah. Pemeriksaan karakteristik fitur dan dampak scaling dilanjutkan pada PHASE 1. Statistik audit seluruh data tidak digunakan sebagai batas scaler.

### Kesamaan fitur antar-ID

Audit tambahan membandingkan urutan IMF dan 11 fitur per sensor, dengan mengabaikan ID dan target saat membuat fingerprint. Perbandingan dibatasi pada posisi yang sama. Hasil berikut menghitung seluruh anggota kelompok salinan, bukan hanya salinan tambahan setelah anggota pertama.

| Representasi | Sampel yang memiliki pasangan fitur identik |
|---|---:|
| Flow | 162 |
| Pulse Rate | 160 |
| Pleth | 164 |
| Ketiga sensor sekaligus | 160 |

Pada gabungan ketiga sensor, 160 sampel membentuk 80 kelompok, melibatkan 62 ID. Rinciannya: 62 sampel Supine, 48 Left, dan 50 Right. Lima dari 80 kelompok memiliki AHI berbeda antar-ID, walaupun fitur input ketiga sensornya identik. Target masing-masing ID tetap konsisten dalam dan antar sensor.

Fingerprint hanya dipakai mencari kandidat. Pembacaan ulang workbook dan perbandingan elemen matriks dengan `numpy.array_equal` mengonfirmasi kesamaan pada 240 perbandingan sensor-kelompok, yaitu 80 kelompok x tiga sensor.

Temuan ini memperluas brief: tidak ada duplikasi baris biasa, tetapi terdapat salinan representasi sampel antar-ID. Ini belum membuktikan oversampling, identitas pasien yang sama, atau kesalahan target. Jangan menghapus baris, menyatukan ID, atau merata-ratakan target secara otomatis.

Patient-based split tetap wajib, tetapi belum cukup untuk mencegah salinan ini tersebar antar subset. Sebelum split, minta provenance bila tersedia dan tetapkan kebijakan. Rekomendasi: pertahankan seluruh observasi, bentuk komponen terhubung antar-ID berdasarkan fitur identik tiga sensor pada posisi yang sama, dan pisahkan berdasarkan komponen tersebut. Semua posisi satu ID selalu mengikuti kelompok yang sama. Laporkan rasio aktual dan jumlah kelompok efektif. Keputusan ini belum diterapkan pada PHASE 0.

## Environment dan GPU

| Komponen | Hasil aktual |
|---|---|
| OS | Windows 11 Pro for Workstations, 64-bit, build 26200 |
| RAM yang terlihat OS | Sekitar 63,8 GiB |
| Python | 3.11.0, 64-bit |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| VRAM total | 8192 MiB |
| VRAM terpakai pada snapshot audit | Sekitar 1,4 GiB; berubah sesuai aplikasi aktif |
| Compute capability | 8.6 |
| Driver | 610.88 |
| CUDA menurut nvcc | Toolkit 13.0, V13.0.88 |
| CUDA menurut nvidia-smi | UMD 13.3 |
| TensorFlow/Keras | Belum terpasang |
| GPU melalui TensorFlow | Belum dapat diuji karena TensorFlow tidak tersedia |
| WSL | Belum terpasang |
| nbformat/nbclient/JupyterLab/Notebook | Belum terpasang |
| IPykernel | 7.1.0, import berhasil |

NumPy 2.3.4, Pandas 2.3.3, Scikit-learn 1.9.1, Matplotlib 3.10.8, Seaborn 0.13.2, SciPy 1.17.0, OpenPyXL 3.1.5, dan Joblib 1.6.0 berhasil diimpor. `pip check` tidak melaporkan dependensi rusak. Versi tersebut dicatat sebagai keadaan aktual, tanpa mengganti paket global.

[Panduan resmi TensorFlow](https://www.tensorflow.org/install/pip) menjelaskan bahwa GPU native Windows berhenti pada TF 2.10, sementara TensorFlow modern memakai WSL2 untuk jalur GPU Windows. Rekomendasi proyek adalah WSL2 dengan Python 3.11 dan kandidat TensorFlow 2.20.0. [Metadata resmi TensorFlow 2.20.0](https://pypi.org/project/tensorflow/2.20.0/) menyediakan wheel Python 3.11 pada Windows dan Linux x86-64 serta extra NVIDIA `and-cuda` untuk Linux. Pemasangan paket dan pengujian runtime belum dilakukan; kompatibilitas akhir belum dinyatakan lulus.

`requirements.txt` mencatat dependensi audit yang terpasang. `requirements-training.txt` memisahkan kandidat kebutuhan notebook/model: nbformat 5.11.1, nbclient 0.11.0, JupyterLab 4.6.6, dan TensorFlow 2.20.0, dengan marker Windows CPU/Linux and-cuda. Metadata paket telah diperiksa. Ini bukan lockfile lengkap atau bukti environment training telah siap.

## File yang dibuat

- `AGENTS.md`: kontrak penelitian, privasi, keputusan metodologi, temuan salinan, dan rencana semua phase.
- `README.md`: struktur, setup, cara audit, rencana GPU/notebook, metodologi, troubleshooting, dan keterbatasan.
- `src/__init__.py`, `src/config.py`, `src/audit.py`: package dasar, konfigurasi bersama, audit aktual.
- `requirements.txt`, `requirements-training.txt`: versi audit dan kandidat setup selanjutnya.
- `tests/test_phase0.py`: pengujian audit dan penjagaan konfigurasi.
- `.gitignore` dan `.gitkeep`: perlindungan data/artifacts dan direktori dasar.
- `docs/phase_0_report.md`: laporan ini.

Direktori `notebooks/`, `artifacts/processed_data/`, `models/`, `metrics/`, `predictions/`, dan `figures/` telah disiapkan. Modul preprocessing dan model akan dibuat pada phase masing-masing.

Bukti lokal di `artifacts/audit/`:

| File | Isi |
|---|---|
| `dataset_audit.json` | Ringkasan dataset, hash sumber, temuan integritas, dan peringatan |
| `environment.json` | Versi/import paket, Python, OS, GPU, CUDA, WSL, pip check |
| `sheet_summary.csv` | Baris, kolom, pasien, dan kesesuaian posisi per sheet |
| `numeric_statistics.csv` | Statistik fitur/target per sensor, termasuk persentil |
| `imf_counts.csv` | Jumlah IMF tiap sampel dan sensor |
| `sample_availability.csv` | Ketersediaan sensor dan target setiap pasien-posisi |
| `within_sensor_target_conflicts.csv` | Konflik target dalam sensor, kosong pada audit aktual |
| `cross_sensor_target_conflicts.csv` | Konflik target antar sensor, kosong pada audit aktual |
| `ahi_summary.csv` | Statistik AHI pada sampel complete-case |
| `high_ahi_samples.csv` | Sampel di atas penanda audit 100, tanpa mengubah target |
| `identical_feature_samples.csv` | Kandidat salinan matriks per sensor |
| `identical_fused_samples.csv` | Kelompok salinan gabungan tiga sensor, dengan target asli |

CSV yang memuat ID dan seluruh artifacts tetap lokal dan diabaikan Git.

## Pengujian aktual

| Pemeriksaan | Hasil |
|---|---|
| `python -m src.audit` pada tiga workbook asli | Berhasil; nol masalah integritas yang menghentikan audit; peringatan salinan disimpan |
| `python -m unittest discover -s tests -v` | 7 tes lulus |
| Header whitespace, ID string, dan hash sumber | Lulus pada fixture dan data aktual |
| Nilai invalid/infinite/missing, IMF di luar batas, duplikasi kunci, posisi salah | Terdeteksi pada fixture |
| Konflik target dalam sensor | Dilaporkan tanpa averaging pada fixture |
| Konflik target antar sensor | Menghasilkan laporan dan menghentikan tahap lanjutan pada fixture |
| Salinan input dengan ID/target berbeda | Terdeteksi tanpa mengganti ID/target pada fixture |
| Schema yang bertabrakan setelah trim | Ditolak dan file handle ditutup; pengujian berjalan pada Windows |
| Validasi split fraction, batas scaling, ukuran citra, padding terpisah | Lulus |
| Perbandingan matriks asli kandidat salinan | 240 pemeriksaan sensor-kelompok lulus |
| `python -m pip check` | Tidak ada dependensi rusak pada environment audit |
| Konsistensi manifest audit dan hash sumber akhir | Lulus; setiap pasangan unik dan jumlah complete-case sesuai ringkasan |
| `.gitignore` pada repository sementara | Lulus untuk workbook, metadata pasien, model, processed data, dan environment; `.gitkeep` tetap dapat dilacak |
| Dokumentasi yang dapat dibagikan | Tidak mengandung ID pasien aktual; teks buatan diperiksa tanpa em/en dash |
| Batas cakupan phase | Notebook dan direktori hasil fase berikutnya masih kosong |

Tes menggunakan workbook sintetis di direktori sementara, bukan mengubah workbook pasien. Saat pengujian ditemukan file handle yang masih terbuka pada jalur penolakan schema dan statistik NaN yang tidak dapat diserialisasi ke JSON. Keduanya sudah diperbaiki: workbook memakai context manager, dan statistik yang tidak terdefinisi disimpan sebagai null.

Split leakage, citra, runtime GPU TensorFlow, penyimpanan/pemuatan model, dan metrik model belum diuji karena belum masuk fase implementasinya. Tidak ada angka performa penelitian pada PHASE 0.

## Keputusan metodologi

1. Pertahankan sumber dan ID; audit ulang data aktual menjadi dasar keputusan.
2. Gunakan complete-case 1.091 sampel untuk rencana eksperimen utama. Enam belas pasangan tanpa Pulse Rate hanya akan dikeluarkan dari eksperimen utama, bukan dihapus dari sumber.
3. Pertahankan AHI asli termasuk nol dan 600. Penanda ekstrem tidak digunakan sebagai filter.
4. Tangani risiko salinan sebelum split. Pengelompokan komponen salinan merupakan rekomendasi untuk persetujuan PHASE 1, belum hasil preprocessing.
5. Fit scaler hanya pada training. Statistik seluruh data pada laporan ini tidak menjadi parameter scaler.
6. Rencana fitur nyata memakai piksel 1..255 dan padding 0, disertai mask. Preprocessing resmi backbone diterapkan satu kali di model.
7. Gunakan WSL2 sebagai rekomendasi jalur GPU. Setup sistem, instalasi environment training, dan smoke test dilakukan pada tahap berikutnya sesuai persetujuan.

## Rencana implementasi PHASE 1

1. Tetapkan kebijakan salinan dan provenance berdasarkan temuan di atas. Jika provenance belum tersedia, rekomendasi konservatif adalah grouping komponen salinan tanpa menghapus observasi.
2. Siapkan environment notebook terpisah. Preprocessing dapat berjalan di CPU Windows; WSL2 perlu diselesaikan sebelum smoke test model GPU pada PHASE 2.
3. Implementasikan `src/preprocessing.py` memakai kontrak konfigurasi yang sama dan pemeriksaan yang menghentikan proses bila target/kunci tidak konsisten.
4. Buat dan jalankan `01_Preprocessing.ipynb`: penjelasan formal, EDA aktual, kelengkapan sensor, AHI ekstrem, jumlah IMF, serta risiko salinan.
5. Buat split kelompok 70/15/15 dengan seed 42 dan manifest bersama. Validasi tidak ada overlap ID maupun komponen salinan sesuai protokol yang dipilih. Laporkan jumlah pasien, kelompok, posisi, sampel, dan AHI tiap subset.
6. Fit percentile scaler training per sensor/fitur. Tangani fitur konstan secara eksplisit; laporkan clipping pada semua subset. Jangan memakai target untuk scaling fitur.
7. Susun 50 x 11 x 3, simpan mask, encode padding secara terpisah, resize nearest-neighbor ke ukuran configurable. Visualisasikan tiap sensor dan gabungannya sebagai pseudo-image.
8. Simpan input citra, one-hot posisi, target, ID, metadata, manifest, feature order, scaler, konfigurasi, dan laporan audit. Periksa shape, finite values, dan keselarasan setelah reload.
9. Laporkan hasil aktual dan keterbatasan PHASE 1, lalu tunggu persetujuan PHASE 2.

## Pemeriksaan antislop

Pengguna memilih mode DURING. Cakupan hasil saat ini adalah kode audit dan dokumentasi penelitian. Tidak ada UI, aset ilustrasi, atau interaksi yang dibuat; pemeriksaan visual, dials, breakpoint, tema, kontras, navigasi, dan click-through tidak berlaku pada hasil ini.

| Gate yang berlaku | Status dan bukti |
|---|---|
| Hard gate R-02 | PASS: teks buatan proyek diperiksa tanpa em/en dash; brief asli tidak diubah |
| Hard gate R-17, R-18, R-36, R-38 | PASS: angka dataset berasal dari audit; rekomendasi dipisahkan dari hasil; tidak ada angka performa model rekaan |
| Hard gate R-23, R-33 | PASS: tidak dibuat aset visual atau script yang menambal UI/CSS |
| Hard gate R-35 | PASS untuk kode: audit dijalankan dan tujuh tes lulus; runtime TensorFlow dinyatakan belum diuji |
| Purpose gate | PASS untuk dokumentasi: tabel membandingkan sensor, posisi, environment, dan bukti pengujian; tidak ada teknik dekorasi UI |
| Liveliness | Tidak berlaku: tidak ada desain antarmuka atau visualisasi pada PHASE 0 |
| Craftsmanship C-1, C-3, C-5 dan R-16, R-31 | PASS: konfigurasi/keputusan dijelaskan oleh kebutuhan brief dan hasil audit; klaim dibatasi bukti |
| Antislop-code | PASS: komentar menjelaskan urutan channel, padding, fingerprint, dan statistik yang tidak terdefinisi; tidak ada komentar dekoratif |
| Antislop-copywriting | PASS: dokumentasi memakai istilah penelitian yang konkret dan mencatat keterbatasan environment serta provenance |

Status ini tidak menyatakan pipeline preprocessing atau training siap. Persetujuan phase berikutnya tetap mengikuti bagian 14 `BRIEF.md`.
