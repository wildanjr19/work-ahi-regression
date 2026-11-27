# Penyelesaian full grid tuning

Diperiksa pada 27 November 2026 sebelum push. Seluruh 108 kandidat selesai dengan status succeeded: 36 per arsitektur, tanpa kandidat gagal. Penanda completion.json ketiga arsitektur menyatakan completed dan test_accessed=False. Evaluasi test tuning belum dijalankan.

## Hasil pelaksanaan

Angka berikut dihitung dari candidates.json di artifacts/tuning/grid_v1/{ResNet50,VGG16,VGG19}/full/.

| Arsitektur | Kandidat berhasil | Total epoch | Waktu training (detik) | Rata-rata detik/epoch | Skipped AMP steps |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 36 | 1253 | 2660.73 | 2.123 | 197 |
| VGG16 | 36 | 1410 | 4011.42 | 2.845 | 184 |
| VGG19 | 36 | 1374 | 4504.35 | 3.278 | 194 |

Waktu training adalah jumlah training_seconds kandidat yang selesai, termasuk loop train/validation dan penyimpanan checkpoint selama loop, tanpa konstruksi model serta evaluasi akhir/reload. Rata-rata menggabungkan scratch, FE, dan FT; bukan estimasi waktu tiap strategi. FT langsung tidak memasukkan biaya FE terpisah. Skipped AMP steps pada full grid dicatat apa adanya dan berbeda dari smoke yang tidak melewati optimizer step. Tabel tidak mencakup percobaan yang diarsipkan atau terhenti di luar inventaris full grid ini.

Ringkasan comparison_validation.csv dan selection_validation.json tersedia secara lokal untuk setiap arsitektur. Pemeriksaan membaca status 108 kandidat, kesesuaian fingerprint/protokol, keunikan kombinasi, serta kecocokan pilihan tiap strategi dengan RMSE validation terkecil dan MAE jika seri. Pemeriksaan ini tidak menghitung ulang prediksi atau memuat checkpoint/test. Pemilihan lintas arsitektur dan pembekuan pilihan untuk evaluasi lanjutan masih menunggu tahap berikutnya.

## Persiapan repositori dan pengujian

- Kode grid bersama, tiga notebook, pengujian, dokumentasi, dan penataan aset presentasi disiapkan untuk commit.
- Notebook beroutput disalin beserta SHA-256 ke artifacts/tuning/notebook_checks/pre_push_20261127_171751/. Notebook sumber dibersihkan dari output dan execution count.
- Sesuai arahan pengguna, RUN_TUNING=True, RUN_MODE='full', dan FULL_TRAINING_APPROVED=True dipertahankan. Run All memanggil training dan menolak namespace grid_v1 yang sudah ada. Untuk membaca hasil saja, nonaktifkan RUN_TUNING atau jalankan sel hasil tersimpan. Flag tersimpan tidak memberi persetujuan baru untuk mengulang full training.
- Tes notebook menguji mode nonaktif secara eksplisit dan penolakan full training tanpa persetujuan, tanpa bergantung pada flag tersimpan.
- Unit test aktual: 49 lulus. pip check tidak menemukan dependensi rusak.
- Verifier arsip: PASS untuk 571 berkas snapshot, 82 aset beku, dan tiga workbook eksternal; tanpa memuat array test atau menjalankan inferensi. Pemeriksaan ini memverifikasi snapshot, bukan seluruh aset aktif.

Data, checkpoint, prediksi individual, notebook beroutput, dan hasil eksperimen tetap lokal serta diabaikan Git. Dokumen ini hanya memuat agregat. Hash pada protokol dan hasil historis tidak diperbarui; perubahan dokumentasi setelah training tidak mengganti rekam hash saat eksperimen.

## Keputusan dan rekomendasi

Tidak ada perubahan model, split, preprocessing, atau grid pada persiapan push ini. Smoke tetap diagnostik. Tinjau hasil validation sembilan kondisi dan bekukan pilihan sebelum evaluasi test lanjutan setelah persetujuan tahap berikutnya. Test lama telah terpapar pada PHASE 5; hasil berikutnya tidak menjadi evaluasi independen baru dan tidak membuktikan validitas klinis.

Antislop R-17/R-36 PASS: jumlah dan biaya dihitung dari rekaman kandidat aktual; tidak ada klaim performa atau validasi klinis. R-02 PASS: tanpa em/en dash; ketentuan UI tidak berlaku.
