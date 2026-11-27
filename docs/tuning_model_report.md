# Fondasi konfigurasi dan model tuning

Status terbaru: full grid telah selesai; lihat [laporan full grid](tuning_full_report.md). Isi berikut merekam tahap implementasi dan smoke sebelum full training.

Selesai pada 27 November 2026. Tahap ini menyiapkan konfigurasi dan pembentukan kandidat; belum mengimplementasikan runner grid atau memperbarui notebook.

## Perubahan

- `src/config.py`: satu search space dan `tuning_candidates()` menghasilkan 12 konfigurasi yang sama untuk seluruh kondisi. Learning rate scratch/FE/FT disamakan pada setiap kandidat; kedua dropout memakai nilai sama. Weight decay 0 dicatat eksplisit; default learning rate historis tetap dipertahankan.
- `src/model_utils.py`: `build_candidate_model()` mereset seed dan membentuk model baru. Scratch memakai bobot acak; FE/FT memakai ImageNet `IMAGENET1K_V1` dan head baru. FT langsung membuka blok akhir tanpa parent FE. Konfigurasi di luar grid/protokol ditolak. Runner lama dipertahankan sebagai jalur historis.
- `tests/test_tuning_models.py`: pengujian grid, penolakan konfigurasi tidak sah, inisialisasi ulang, cakupan gradien, pembekuan BN, dan pemuatan checkpoint lama tanpa field weight decay.
- `AGENTS.md` dan `docs/desain_eksperimen.md`: status dan langkah berikutnya diperbarui.

## Pengujian aktual

| Pemeriksaan | Hasil |
|---|---|
| `python -m unittest discover -s tests -v` | 36 tes lulus, termasuk empat tes baru; unit test tidak mengunduh bobot |
| `python -m pip check` | Tidak ada dependensi rusak |
| Probe GPU dengan bobot ImageNet lokal | Sembilan kondisi lulus; satu optimizer update per kondisi |
| Inisialisasi FE/FT | Backbone cocok dengan cache yang hash-nya sesuai arsip; head FT baru, tanpa checkpoint FE |
| Pembaruan bobot | Head berubah; backbone FE tetap; FT hanya memperbarui blok akhir; parameter beku dan state BN transfer tetap |
| Output, gradien, dan checkpoint | Finite; output float32; prediksi sesudah reload konsisten |
| Integritas aktif | 80 aset beku selain dua sumber kode yang sengaja diubah tetap cocok; kelima notebook sumber tidak berubah |
| Arsip sebelum perubahan | 571 berkas, 82 aset beku, dan tiga workbook asli lolos verifier |

Probe memakai input sintetis NCHW `2 x 3 x 32 x 32`, float32 tanpa autocast, pada RTX 3060 Ti; peak alokasi GPU 538,75 MiB. Tidak membaca data pasien atau array test, mengunduh bobot, maupun menyimpan hasil ke direktori eksperimen lama. Bukti lokal: `artifacts/tuning/model_initialization/verification.json` dan `verify_models.py`.

## Masalah dan batasan

Verifier awal perlu menyesuaikan cache ResNet50 lama yang tidak menyimpan `num_batches_tracked`: counter tersebut diperiksa bernilai nol sesuai inisialisasi PyTorch. Bobot model dan protokol tidak diubah untuk menangani hal ini.

Probe sintetis ini bukan smoke pipeline penelitian atau hasil performa. AMP/GradScaler, input 224 x 224 dengan batch 8, scheduler, early stopping, metrik validation, dan pencatatan kegagalan kandidat melalui runner baru belum diuji. Tuning belum terhubung ke notebook; `train_experiment()` masih menjalankan protokol historis FT dari FE. Tidak ada full training atau evaluasi test baru.

## Berikutnya

Implementasikan runner grid bersama yang memakai `tuning_candidates()` dan `build_candidate_model()`, menolak test, serta menyimpan hasil terpisah di `artifacts/tuning/`. Integrasikan notebook dengan tuning default nonaktif, kemudian jalankan smoke pipeline train/validation. Full training tetap menunggu persetujuan setelah laporan smoke tersebut.

Antislop R-31 PASS: memakai konfigurasi dan model bersama tanpa menyalin implementasi backbone.

Antislop R-17/R-36 PASS: angka berasal dari pengujian aktual; probe dibedakan dari hasil penelitian.

Antislop R-02 PASS: laporan tanpa em/en dash atau identitas pasien; aturan UI tidak berlaku.
