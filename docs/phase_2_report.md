# PHASE 2: hasil ResNet50

Tiga eksperimen penuh selesai setelah persetujuan pengguna. PyTorch CUDA native Windows memakai RTX 3060 Ti, batch 8, AMP, Adam/MSE, seed 42, maksimal 50 epoch, dan early stopping patience 10. Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan.

## Hasil validation

Training: 755 sampel. Validation: 157 sampel dari 59 ID, terdiri atas Supine 59, Left 48, Right 50. Test belum digunakan.

| Skenario | MAE | MSE | RMSE | R2 | Epoch selesai/terbaik |
|---|---:|---:|---:|---:|---:|
| Scratch | 13,3048 | 354,7664 | 18,8352 | 0,4638 | 35/25 |
| Feature extraction | 16,4157 | 481,1118 | 21,9343 | 0,2728 | 34/24 |
| Fine-tuning | 14,0015 | 439,8411 | 20,9724 | 0,3352 | 13/3 |

Scratch terpilih berdasarkan RMSE validation, sesuai kriteria sebelum training. Transfer learning belum mengungguli scratch secara keseluruhan pada konfigurasi ini. Fine-tuning menunjukkan pola overfitting setelah epoch terbaiknya. Ini belum merupakan hasil test atau kesimpulan antararsitektur.

## Validasi dan biaya

- Checkpoint, reload CUDA, metrik CSV, epoch terbaik, hash kode/data, dan parent fine-tuning diverifikasi. Test NPZ tidak dibuka. Target asli, termasuk 600, dipertahankan.
- Batch Normalization tetap beku pada transfer learning. Tidak ada prediksi validation negatif; tidak diterapkan clipping prediksi.
- Durasi training stage: scratch 126,44 detik; feature extraction 44,92; fine-tuning 24,54. Biaya kumulatif fine-tuning 69,47 detik termasuk feature extraction.
- Peak alokasi VRAM: 759,96/209,27/397,01 MiB. AMP melewati 6/5/5 langkah overflow; seluruh skenario tetap menyelesaikan optimizer update. Loss dan prediksi finite.
- Sebanyak 24 tes dan pip check lulus. Dua puluh satu grafik full disimpan pada 300 DPI. Notebook laporan dijalankan dari kernel baru tanpa mengulang training.

## File dan langkah berikutnya

Hasil penuh: `artifacts/experiments/ResNet50/`, termasuk checkpoint, history, prediksi, metrik per posisi, `comparison_validation.csv`, `selection_validation.json`, dan QA. Smoke tetap terpisah.

Perubahan: `src/reporting.py`, pengujiannya, `02_ResNet50.ipynb`, README, dan AGENTS. Notebook laporan: `artifacts/notebooks/02_ResNet50.full_results.executed.ipynb`.

Berikutnya PHASE 3: VGG16 dengan split, head, dan protokol yang sama, setelah persetujuan pengguna. Fitur identik dan AHI ekstrem dipertahankan sesuai arahan klien.

Antislop: PASS; angka berasal dari eksekusi dan laporan tidak memuat ID pasien. Keterbatasan validation dan status data dinyatakan eksplisit.
