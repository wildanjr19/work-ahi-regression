# PHASE 5: perbandingan dan evaluasi akhir

Sembilan CNN dibandingkan dengan baseline median dan Ridge tanpa training ulang CNN. ResNet50 scratch terpilih berdasarkan RMSE validation sebelum test dibaca. Baseline terpilih: median training. Pilihan, checkpoint, kode, dan data dibekukan dalam `frozen_protocol.json`.

## Hasil utama

Test: 179 sampel, 65 ID, 57 komponen; Supine 65, Left 58, Right 56.

| Model | RMSE validation | MAE test | RMSE test | R2 test |
|---|---:|---:|---:|---:|
| ResNet50 scratch | 18,8352 | 17,9986 | 29,0473 | 0,1059 |
| Median training | 27,1269 | 22,6362 | 34,0414 | -0,2280 |
| Ridge | 459,1794 | 21,2306 | 34,5009 | -0,2614 |

Pada CNN terpilih, RMSE test terbesar adalah Left 34,8519; terkecil Supine 23,8864. Transfer learning tidak konsisten: fine-tuning VGG16 menurunkan RMSE test terhadap scratch, sedangkan ResNet50 dan VGG19 meningkat. Hasil test tidak mengubah pemilihan.

## Temuan dan QA

- Ridge memilih alpha 1000 dari lima kandidat validation. Satu sampel mendominasi 99,60% kuadrat error validation, konsisten dengan ekstrapolasi fitur setelah StandardScaler. Scaler tetap hanya fit pada 755 sampel train; model tidak diubah setelah test.
- Prediksi negatif dipertahankan: Ridge 12 validation/9 test, VGG19 fine-tuning 1 test. Tidak ada clipping atau penghapusan error buruk.
- Sebanyak 32 tes dan pip check lulus. Seluruh 88 baris metrik dihitung ulang secara independen; hash aset beku tetap. Test NPZ dimuat sekali setelah pembekuan.
- Notebook laporan dijalankan dari kernel baru tanpa fitting ulang. Sebanyak 95 grafik 300 DPI tersedia, termasuk panel CNN pendamping agar error Ridge tidak mendominasi skala.

## File dan batasan

Baru: `src/comparison.py`, `src/comparison_plots.py`, `tests/test_comparison.py`, dan `notebooks/05_Model_Comparison.ipynb`. `src/evaluation.py` diperluas untuk nama file validation/test; rumus metrik tetap. README dan AGENTS diperbarui.

Hasil lokal: `artifacts/comparison/`; notebook beroutput: `artifacts/notebooks/05_Model_Comparison.executed.ipynb`. Pembahasan lengkap dan semua kondisi tersimpan di sana.

PHASE 0 sampai PHASE 5 selesai secara teknis. Data masih sementara menunggu konfirmasi klien. Satu seed/split, tanpa CI, dan test telah terpapar; hasil ini belum membuktikan generalisasi eksternal atau validasi klinis.

Antislop R-31 PASS: memakai checkpoint/runner dan fungsi evaluasi bersama.

Antislop R-17/R-36 PASS: angka berasal dari artifacts dan QA; temuan negatif dilaporkan.

Antislop R-02 PASS: laporan tanpa em/en dash atau ID pasien; aturan UI tidak berlaku.
