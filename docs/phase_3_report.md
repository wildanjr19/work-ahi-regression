# PHASE 3: hasil VGG16

Tiga eksperimen penuh selesai setelah persetujuan pengguna. Protokol tetap: PyTorch CUDA native Windows, RTX 3060 Ti, batch 8, AMP, Adam/MSE, seed 42, maksimal 50 epoch, early stopping patience 10. Dataset masih sementara menunggu konfirmasi klien.

## Hasil validation

Training 755 sampel; validation 157 sampel dari 59 ID, terdiri atas Supine 59, Left 48, Right 50. Test belum dimuat untuk pengembangan atau evaluasi.

| Skenario | MAE | MSE | RMSE | R2 | Epoch selesai/terbaik |
|---|---:|---:|---:|---:|---:|
| Scratch | 15,6927 | 514,8964 | 22,6913 | 0,2218 | 26/16 |
| Feature extraction | 15,6031 | 503,6378 | 22,4419 | 0,2388 | 35/25 |
| Fine-tuning | 15,3328 | 473,0566 | 21,7499 | 0,2850 | 18/8 |

Fine-tuning terpilih berdasarkan RMSE validation sesuai kriteria sebelum training. Transfer learning mengungguli scratch pada konfigurasi VGG16 ini. Kurva fine-tuning menunjukkan overfitting setelah epoch terbaik; prediksi memakai checkpoint epoch 8. Ini belum merupakan hasil test atau kesimpulan antararsitektur.

## Validasi dan biaya

- Sebanyak 25 tes dan pip check lulus. Reload CUDA, epoch terbaik, penghitungan ulang metrik keseluruhan/per posisi, serta pemilihan model telah diverifikasi.
- Backbone feature extraction tetap identik dengan bobot ImageNet. Fine-tuning mulai dari checkpoint feature extraction full dan hanya mengubah block5; blok sebelumnya tetap. VGG16 ini tidak memakai Batch Normalization.
- Hash dataset mentah, preprocessing, dan kode bersama tidak berubah. Target tetap; tidak ada prediksi validation negatif atau clipping prediksi.
- Waktu training tiap tahap: 154,15/120,23/62,27 detik; biaya kumulatif fine-tuning 182,50 detik. Peak alokasi VRAM: 624,84/327,89/406,90 MiB. AMP melewati 6/4/4 langkah overflow; seluruh tahap menyelesaikan optimizer update dan menghasilkan loss finite.
- Dua puluh satu grafik full tersimpan pada 300 DPI. Notebook laporan dijalankan dari kernel baru tanpa training ulang.

## File dan langkah berikutnya

Hasil: `artifacts/experiments/VGG16/`, termasuk checkpoint, history, prediksi, metrik, pemilihan, dan QA. Notebook laporan: `artifacts/notebooks/03_VGG16.full_results.executed.ipynb`. Smoke tetap terpisah.

Notebook sumber, README, AGENTS, dan laporan diperbarui. Runner/config bersama tidak berubah. Berikutnya PHASE 4 VGG19 setelah persetujuan pengguna. Provenance dan AHI ekstrem tetap perlu konfirmasi klien.

Antislop R-31 PASS: runner dipakai ulang; notebook default menonaktifkan full training.

Antislop R-17/R-36 PASS: angka berasal dari hasil eksekusi yang diverifikasi; keterbatasan validation dinyatakan.

Antislop R-02 PASS: laporan tanpa em/en dash atau ID pasien; aturan UI tidak berlaku.
