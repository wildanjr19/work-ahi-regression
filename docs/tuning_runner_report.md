# Runner grid search bersama

Status terbaru: full grid telah selesai; lihat [laporan full grid](tuning_full_report.md). Isi berikut merekam tahap implementasi dan smoke sebelum full training.

Selesai pada 27 November 2026. Runner tersedia dan diuji dengan data sintetis. Notebook belum diintegrasikan; tidak ada full training atau evaluasi test baru.

## Perubahan

- `src/tuning.py`: `run_grid_search()` memakai konfigurasi bersama dan loop training yang memanggil `build_candidate_model()`. Default `enabled=False`; input hanya train/validation. Direktori di luar `artifacts/tuning/` dan hasil yang sudah ada ditolak sebelum data dimuat.
- `src/model_utils.py`: loop training dipakai bersama melalui `initialization="independent"`, tanpa parent FE. Jalur historis tetap tersedia. Reload kandidat memvalidasi SHA-256, arsitektur, strategi, konfigurasi, mode, fingerprint data, protokol, dan hash kode. Konfigurasi runtime dapat dipanggil ulang dengan jumlah thread yang sama.
- `src/reporting.py`: ringkasan grid penuh memilih kandidat tiap strategi berdasarkan RMSE validation lalu MAE. FT langsung hanya menghitung biaya tahapnya sendiri. Seluruh konfigurasi harus sudah dicoba; kegagalan tetap dilaporkan. Jika satu strategi tidak memiliki kandidat berhasil, pemilihan tidak diterbitkan.
- `tests/test_tuning_runner.py`: sebelas tes baru untuk pengamanan, alur aktual, checkpoint, pemilihan, kegagalan, scheduler, early stopping, bobot terbaik, dan interupsi. `AGENTS.md` serta desain eksperimen diperbarui.

## Aturan runner

Smoke menjalankan satu konfigurasi tetap, LR `1e-3` dan dropout `0.3`, pada scratch/FE/FT; satu epoch dan satu batch per subset. Hasilnya tidak dipilih atau dibandingkan sebagai performa penelitian. Full search memerlukan persetujuan eksplisit dan CUDA, kemudian mencoba seluruh 12 kombinasi per strategi: 36 per arsitektur, 108 untuk tiga arsitektur.

Struktur hasil: `artifacts/tuning/<namespace>/<arsitektur>/{smoke,full}/<strategi>/<kandidat>/`. Namespace opsional; setiap direktori arsitektur/mode hanya boleh dibuat sekali. Protokol menyimpan konfigurasi, hash data/kode, runtime, dan status paparan test. Tiap kandidat menyimpan status, checkpoint terbaik, history, prediksi/metrik validation per posisi, grafik, waktu, parameter, optimizer update, dan skipped AMP steps. Kegagalan menyimpan error serta kemajuan yang tersedia; kandidat berikutnya tetap dijalankan. Interupsi dicatat lalu diteruskan.

## Pengujian aktual

| Pemeriksaan | Hasil |
|---|---|
| Unit test lengkap | 47 tes lulus; termasuk sebelas tes runner baru |
| `pip check` | Tidak ada dependensi rusak |
| Training/reload sintetis melalui runner di GPU | Sembilan kondisi CNN asli lulus, AMP aktif |
| Optimizer dan numerik | Sembilan update, nol skipped steps; loss/prediksi finite dan reload konsisten |
| Scheduler/early stopping/best weights | Skenario terkontrol lulus; checkpoint terbaik dipulihkan |
| Full-grid orchestration | 36 kandidat per arsitektur disimulasikan; tie-break, kegagalan parsial/total per strategi, serta larangan pemilihan smoke lulus |
| Pengamanan | Default nonaktif, izin full, test, overlap ID/sampel, konfigurasi checkpoint berbeda, penimpaan, dan output di luar tuning ditolak |
| Aset historis aktif | 80 aset beku selain dua sumber kode yang berubah tetap cocok; lima notebook sumber tidak berubah |
| Verifier arsip | 571 berkas, 82 aset beku, dan tiga workbook asli lulus |

Probe GPU memakai data sintetis NHWC `8 x 32 x 32 x 3` pada RTX 3060 Ti; peak alokasi training 531,33 MiB. Bobot ImageNet memakai cache lokal, tanpa unduhan. Semua pembacaan NPZ diblokir pada probe; tidak ada data pasien atau array test yang dimuat. Sebanyak 63 grafik tersimpan pada 300 DPI.

Bukti lokal: `artifacts/tuning/runner_checks/`, termasuk `unit_tests.log`, `unit_tests.json`, `synthetic_gpu_verification.json`, `preservation.json`, dan script probe. Checkpoint serta hasil sintetis ada di subdirektori `synthetic_gpu/` dan tidak masuk hasil penelitian.

## Batasan dan berikutnya

Tidak ada penghambat tersisa. Cabang full search diuji melalui simulasi, bukan full training CNN. Waktu dan VRAM probe 32 x 32 tidak mewakili biaya input penelitian 224 x 224. Smoke pipeline train/validation dengan preprocessing aktual, integrasi notebook, dan pemilihan antararsitektur belum dijalankan.

Berikutnya integrasikan tiga notebook dengan tuning default nonaktif, kemudian jalankan smoke pipeline aktual dan laporkan hasilnya. Full training tetap menunggu persetujuan setelah laporan smoke tersebut. Test lama sudah terpapar dan tidak boleh dipakai untuk tuning atau diklaim sebagai evaluasi independen baru.

Antislop R-31 PASS: runner memakai loop training, konfigurasi, pembentukan model, dan evaluasi bersama.

Antislop R-17/R-36 PASS: angka berasal dari eksekusi; simulasi dan probe sintetis dibedakan dari pelatihan penelitian.

Antislop R-02 PASS: laporan tanpa em/en dash atau identitas pasien; aturan UI tidak berlaku.
