# Arsip dan verifikasi sebelum tuning

Selesai pada 27 November 2026. Kode model, notebook sumber, preprocessing, protokol, dan hasil PHASE 0 sampai PHASE 5 dipertahankan. Tidak ada training atau inferensi ulang.

## Arsip lokal

Lokasi: `artifacts/archives/pre_tuning_2026-11-27/`.

Snapshot memuat **571 berkas, 3,862 GiB**: kode/config, pengujian, dokumentasi, notebook sumber/beroutput, preprocessing, hasil penuh/smoke, perbandingan, dan cache bobot ImageNet. Setiap salinan dicocokkan dengan SHA-256 sumber. Commit Git, status perubahan, dan diff turut dicatat; dokumen yang belum dilacak Git tetap masuk snapshot.

Workbook asli tetap di `data/`, tanpa salinan tambahan. Manifest mencatat hash ketiganya. Arsip diabaikan Git dan tetap lokal pada drive yang sama.

## Hasil pemeriksaan

| Pemeriksaan aktual | Hasil |
|---|---|
| `verify_frozen_protocol` | Lulus; seluruh 82 aset cocok. Hash protokol sama dengan selection, completion, dan penanda akses test. |
| Snapshot dan sumber | Seluruh 571 salinan cocok dengan sumber; berkas sumber yang diarsipkan tidak berubah selama pemeriksaan. |
| Audit ulang tiga workbook | Nol masalah penghambat; hash sumber tetap; temuan agregat sama dengan audit historis. |
| Manifest split | 1.091 sampel, 411 ID, 380 komponen; train/validation/test 755/157/179. Nol overlap ID, komponen, dan fingerprint mentah. |
| Metrik dari prediksi tersimpan | Seluruh 88 baris metrik cocok; 3.696 baris prediksi diperiksa terhadap manifest dan target. |
| Notebook sumber | Kelima notebook tanpa output dan execution count. |
| Unit test | 32 tes lulus. |
| `pip check` | Tidak ada dependensi rusak. |
| Verifier arsip | Berkas valid diterima; perubahan isi dan berkas hilang ditolak pada pemeriksaan sintetis. |

Audit ulang menjalankan `src.audit.run_audit()` dengan direktori keluaran diarahkan ke `verification/audit/` dalam arsip agar audit historis tidak tertimpa. NPZ penelitian hanya disalin dan diperiksa hash; array test tidak dimuat ulang. Pemeriksaan metrik menggunakan CSV prediksi yang sudah tersimpan.

Manifest, log, dan ringkasan mesin tersedia dalam arsip. Untuk memeriksa snapshot setelah kode workspace berubah:

```powershell
.\.venv\Scripts\python.exe artifacts/archives/pre_tuning_2026-11-27/verify_archive.py
```

Verifier memetakan path aset ke snapshot tanpa mengubah protokol beku. Pemeriksaan tetap memerlukan ketiga workbook asli; arsip ini bukan salinan lengkap data pada media terpisah.

## Keputusan dan langkah berikutnya

Tidak ada kegagalan integritas. Temuan fitur identik, perbedaan AHI antar-ID tertentu, dan target ekstrem tetap berlaku. Test sudah terpapar; hasil historis dipertahankan dan evaluasi lanjutan tidak menjadi evaluasi independen baru.