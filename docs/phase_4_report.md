# PHASE 4: hasil VGG19

Smoke test, QA, dan tiga eksperimen penuh selesai sesuai izin pengguna. Protokol tetap: PyTorch CUDA, RTX 3060 Ti, batch 8, AMP, Adam/MSE, seed 42, maksimal 50 epoch, early stopping patience 10. Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan.

## Hasil validation

Training 755 sampel; validation 157 sampel dari 59 ID, terdiri atas Supine 59, Left 48, Right 50. Test tidak dimuat untuk pengembangan atau evaluasi.

| Skenario | MAE | MSE | RMSE | R2 | Epoch selesai/terbaik |
|---|---:|---:|---:|---:|---:|
| Scratch | 16,5906 | 470,4501 | 21,6899 | 0,2890 | 28/18 |
| Feature extraction | 15,2581 | 498,6296 | 22,3300 | 0,2464 | 48/38 |
| Fine-tuning | 14,0135 | 448,8038 | 21,1850 | 0,3217 | 12/2 |

Fine-tuning terpilih berdasarkan RMSE validation sesuai kriteria awal. Feature extraction menurunkan MAE tetapi meningkatkan RMSE dibanding scratch. Fine-tuning menunjukkan pola overfitting setelah epoch terbaik; prediksi memakai checkpoint epoch 2. R2 posisi Right negatif pada ketiga skenario. Ini belum merupakan hasil test atau kesimpulan antararsitektur.

## Validasi dan biaya

- Sebanyak 26 tes dan pip check lulus. Reload CUDA, epoch terbaik, metrik keseluruhan/per posisi, serta pemilihan model diverifikasi.
- Backbone feature extraction identik dengan ImageNet. Fine-tuning memakai parent feature extraction full dan hanya mengubah block5; blok sebelumnya tetap. VGG19 ini tanpa Batch Normalization.
- Hash data mentah, preprocessing, dan kode bersama tidak berubah. Target tetap; tidak ada prediksi validation negatif atau clipping prediksi.
- Waktu training tiap tahap: 154,54/102,23/30,91 detik; kumulatif fine-tuning 133,14 detik. Peak alokasi VRAM: 721,23/347,27/454,79 MiB. AMP melewati 8/4/4 langkah overflow; optimizer update tetap berjalan dan loss finite.
- Dua puluh satu grafik full tersimpan pada 300 DPI. Notebook laporan dieksekusi dari kernel baru tanpa training ulang.

## File dan berikutnya

Notebook baru: `notebooks/04_VGG19.ipynb`. Pengujian, README, dan AGENTS diperbarui. Runner/config bersama tetap.

Hasil lokal: `artifacts/experiments/VGG19/`, mencakup checkpoint, history, prediksi, metrik, pemilihan, dan QA. Laporan beroutput: `artifacts/notebooks/04_VGG19.full_results.executed.ipynb`. Smoke disimpan terpisah.

Sembilan eksperimen CNN tersedia. Berikutnya PHASE 5: perbandingan model dan baseline setelah persetujuan. Fitur identik dan AHI ekstrem dipertahankan sesuai arahan klien.

Antislop R-31 PASS: memakai runner bersama dan komentar seperlunya.

Antislop R-17/R-36 PASS: angka berasal dari eksekusi terverifikasi; batasan validation dinyatakan.

Antislop R-02 PASS: tanpa em/en dash atau ID pasien; aturan UI tidak berlaku.
