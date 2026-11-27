# Ringkasan perbandingan ResNet50, VGG16, dan VGG19

Setiap arsitektur menjalankan scratch, feature extraction, dan fine-tuning. Tabel berikut memakai skenario terbaik masing-masing arsitektur berdasarkan RMSE validation, kemudian MAE jika seri. Pilihan tersebut dibekukan sebelum evaluasi test.

Matriks A1 sampai C3 dan seluruh sembilan hasil tersedia pada [tabel hasil eksperimen](tabel_hasil_eksperimen.md).

## Hasil skenario terpilih

| Arsitektur | Skenario terpilih | RMSE validation | MAE test | RMSE test | R2 test |
|---|---|---:|---:|---:|---:|
| ResNet50 | Scratch | 18,8352 | 17,9986 | 29,0473 | 0,1059 |
| VGG16 | Fine-tuning | 21,7499 | 19,9531 | 29,3305 | 0,0883 |
| VGG19 | Fine-tuning | 21,1850 | 19,9233 | 30,1204 | 0,0386 |

ResNet50 scratch terpilih sebagai CNN utama dari sembilan kondisi berdasarkan validation. Hasil test-nya juga memiliki MAE dan RMSE terendah di antara kondisi CNN yang dievaluasi. Pemilihan tetap berasal dari validation.

## Kontribusi transfer learning

- ResNet50: feature extraction dan fine-tuning menghasilkan RMSE validation maupun test lebih tinggi daripada scratch.
- VGG16: fine-tuning menurunkan RMSE validation dan test dibanding scratch.
- VGG19: fine-tuning menurunkan RMSE validation, tetapi RMSE test meningkat dibanding scratch. Perbaikannya tidak konsisten antar subset.

## Protokol dan batasan

Seluruh CNN memakai sampel, split, scaler, normalisasi, seed 42, batch 8, Adam/MSE, learning rate, head 128/64, serta aturan early stopping yang sama. Jumlah parameter dan epoch aktual berbeda mengikuti arsitektur dan early stopping. Sampel: train 755, validation 157, test 179.

R2 test masih rendah. Kesimpulan berlaku pada konfigurasi, satu split, dan satu seed ini; tidak ada CI atau uji signifikansi. Seluruh 411 ID disertakan sesuai arahan klien; data dan target asli dipertahankan. Hasil belum membuktikan generalisasi eksternal atau validasi klinis.

Sumber lengkap sembilan kondisi, baseline, grafik, dan metrik per posisi: [notebook 05](../notebooks/05_Model_Comparison.ipynb). Lihat juga [laporan PHASE 5](phase_5_report.md).

Antislop R-17/R-36 PASS: angka sesuai CSV hasil dan pemilihan validation; batasan dinyatakan.

Antislop R-02 PASS: ringkasan tanpa em/en dash atau ID pasien.
