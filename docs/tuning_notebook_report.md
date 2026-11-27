# Integrasi notebook dan smoke aktual

Status terbaru: full grid telah selesai; lihat [laporan full grid](tuning_full_report.md). Isi berikut merekam tahap implementasi dan smoke sebelum full training.

Selesai pada 27 November 2026. Tiga notebook terhubung ke runner bersama; sembilan kondisi lulus smoke train/validation aktual. Full grid belum dijalankan.

## Perubahan

- `02_ResNet50.ipynb`, `03_VGG16.ipynb`, dan `04_VGG19.ipynb`: masing-masing 44 baris kode, tanpa loop training tersalin. Notebook memuat konfigurasi, memanggil `run_grid_search()`, lalu membaca hasil tanpa menulis ulang ringkasan historis.
- Default `RUN_TUNING=False`, `RUN_MODE='smoke'`, dan `FULL_TRAINING_APPROVED=False`. Aktivasi tuning tidak otomatis memberi izin full training. Hasil baru memakai `artifacts/tuning/grid_v1/`; sumber notebook tetap tanpa output/execution count.
- `tests/test_model_notebooks.py`: dua tes baru memeriksa default tanpa akses dataset/penulisan dan penolakan full training tanpa persetujuan. `AGENTS.md`, README, dan desain eksperimen diperbarui; modul training tidak diubah pada tahap ini.

## Pengujian aktual

Ketiga salinan notebook dijalankan melalui nbclient dari kernel `osa-ahi-preprocessing` yang baru untuk setiap arsitektur. Loader memuat train 755 dan validation 157 sampel, berukuran NHWC 224 x 224 x 3. Setiap strategi hanya melatih delapan sampel dan memvalidasi delapan sampel selama satu epoch, dengan LR 1e-3, dropout 0.3, dan AMP aktif.

| Arsitektur | Kondisi lulus | Optimizer update / skipped AMP | Peak alokasi training MiB |
|---|---:|---:|---:|
| ResNet50 | 3/3 | 3/0 | 569,35 |
| VGG16 | 3/3 | 3/0 | 509,27 |
| VGG19 | 3/3 | 3/0 | 581,93 |

Peak tersebut adalah alokasi tensor PyTorch per run, bukan seluruh penggunaan VRAM. Seluruh checkpoint/reload konsisten; output dan loss finite. Hash parameter membuktikan backbone FE tetap, FT hanya mengubah parameter blok akhir yang dibuka, dan head diperbarui. Inisialisasi backbone serta head FE/FT sama sebelum training, membuktikan FT tidak melanjutkan head FE terlatih. State BN transfer tetap.

Pembacaan NPZ dicatat dan dibatasi ke `train.npz` serta `validation.npz`; akses `test.npz`, full training, dan unduhan bobot diblokir pada verifikasi. Hasil smoke tidak masuk pemilihan model. VGG19 FE menghasilkan delapan dari delapan prediksi validation negatif; nilainya disimpan tanpa clipping. Temuan ini tidak mengubah grid atau protokol.

Unit test lengkap: 49 lulus; `pip check` bersih. Eksekusi default notebook diuji tanpa akses dataset atau pembuatan direktori. Peringatan event loop Windows/ZeroMQ ditangani otomatis melalui selector thread dan tidak menghambat eksekusi; tidak ada perubahan konfigurasi sistem.

## Penyimpanan dan integritas

Hasil smoke: `artifacts/tuning/grid_v1/{ResNet50,VGG16,VGG19}/smoke/`. Salinan notebook beroutput, script eksekusi, pemeriksaan bobot/akses NPZ, dan log pengujian tersedia di `artifacts/tuning/notebook_checks/`. Tidak ada notebook sumber yang menyimpan output atau identitas pasien.

Verifier arsip tetap lulus untuk 571 berkas dan 82 aset beku. Tiga workbook serta 480 berkas artifacts historis aktif cocok dengan snapshot, termasuk preprocessing, split, hasil eksperimen/perbandingan, dan notebook beroutput. Sebanyak 80 aset beku aktif selain dua sumber kode yang sebelumnya diubah tetap cocok. Notebook preprocessing dan perbandingan tidak berubah. Hash aset beku lama tidak diperbarui.

## Berikutnya

Full grid menunggu persetujuan setelah laporan ini: 36 kandidat per arsitektur, 108 keseluruhan. Satu batch/satu epoch hanya memvalidasi pipeline, bukan performa atau konvergensi penuh. Gunakan validation untuk seluruh pemilihan; test lama telah terpapar dan tidak menjadi evaluasi independen baru.

Antislop R-31 PASS: notebook memakai runner bersama tanpa helper atau loop training tambahan.

Antislop R-17/R-36 PASS: angka berasal dari eksekusi aktual; prediksi negatif dilaporkan dan smoke dibedakan dari hasil penelitian.

Antislop R-02 PASS: laporan tanpa em/en dash atau identitas pasien; aturan UI tidak berlaku.
