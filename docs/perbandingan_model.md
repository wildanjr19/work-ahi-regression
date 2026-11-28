# Ringkasan perbandingan ResNet50, VGG16, dan VGG19

Ringkasan terbaru setelah full grid 108 kandidat dan evaluasi test lanjutan pada 27 November 2026. Setiap arsitektur memakai skema terbaik berdasarkan RMSE validation, kemudian MAE jika seri. Pemilihan dan checkpoint dibekukan sebelum test terbaru dibaca.

Matriks A1 sampai C3, seluruh metrik validation/test sembilan kondisi, konfigurasi terpilih, dan dua baseline tersedia pada [tabel hasil eksperimen](tabel_hasil_eksperimen.md). FT pada ringkasan ini adalah FT langsung dari ImageNet dan head baru.

## Skema terpilih per arsitektur

Validation: 157 sampel. Cetak tebal menandai CNN utama berdasarkan validation.

Setiap kandidat memakai checkpoint dari epoch dengan MSE validation terendah. Dari 12 kombinasi per kondisi, konfigurasi dipilih berdasarkan RMSE validation terkecil, lalu MAE jika seri; ketiga skema kemudian dibandingkan untuk memilih skema tiap arsitektur.

| Arsitektur | Skema terpilih | Learning rate awal | Dropout | Checkpoint epoch | RMSE validation |
|---|---|---:|---:|---:|---:|
| **ResNet50** | **Scratch** | `1e-04` | 0,1 | 11 | **18,1728** |
| VGG16 | FT langsung | `1e-03` | 0,5 | 5 | 19,2202 |
| VGG19 | FT langsung | `1e-04` | 0,1 | 9 | 19,3368 |

Learning rate pada tabel merupakan nilai awal; scheduler dapat menurunkannya selama pelatihan. Checkpoint epoch menunjukkan bobot terbaik yang dipulihkan, bukan jumlah epoch yang dijalankan. Konfigurasi merujuk pada [tabel hasil eksperimen](tabel_hasil_eksperimen.md#konfigurasi-terpilih-dari-validation), sedangkan epoch merujuk pada [laporan evaluasi tuning](tuning_evaluation_report.md#biaya-kandidat-terpilih).

**ResNet50/Scratch menjadi CNN utama** karena memiliki RMSE validation terendah di antara sembilan kondisi. Pilihan konfigurasi dan checkpoint dibekukan sebelum evaluasi test.

## Hasil test skema terpilih

Test: 179 sampel. Tabel berikut melaporkan performa skema yang telah dipilih dari validation.

| Arsitektur | Skema terpilih | RMSE validation | MAE test | MSE test | RMSE test | R² test |
|---|---|---:|---:|---:|---:|---:|
| **ResNet50** | **Scratch** | **18,1728** | **17,8816** | **864,8428** | **29,4082** | **0,0835** |
| VGG16 | FT langsung | 19,2202 | 18,4123 | 833,2189 | 28,8655 | 0,1170 |
| VGG19 | FT langsung | 19,3368 | 18,7888 | 801,5255 | 28,3112 | 0,1506 |

ResNet50/FT langsung memperoleh RMSE test lebih rendah (27,2430), tetapi hasil test tidak mengubah pilihan model. Baseline terpilih dari validation tetap DummyMedian.

## Kontribusi transfer learning

- ResNet50: RMSE validation FT langsung 18,6765 lebih tinggi daripada scratch 18,1728; RMSE test turun dari 29,4082 menjadi 27,2430.
- VGG16: RMSE validation FT langsung 19,2202 lebih rendah daripada scratch 20,1821; RMSE test turun dari 29,4735 menjadi 28,8655.
- VGG19: RMSE validation FT langsung 19,3368 lebih rendah daripada scratch 20,6907; RMSE test turun dari 30,2843 menjadi 28,3112.

Pada ketiga arsitektur, FE memiliki RMSE validation lebih tinggi daripada scratch. Pada test, FE memperbesar RMSE ResNet50 dan VGG16, tetapi menurunkan RMSE VGG19. Hasil ini berlaku untuk kandidat terpilih pada split ini; tidak membuktikan bahwa transfer learning selalu lebih baik atau lebih buruk.

## Protokol dan batasan

Seluruh kondisi memakai sampel, split, scaler, normalisasi, seed 42, batch 8, Adam/MSE, head 128/64, batas 50 epoch, dan aturan early stopping yang sama. Grid learning rate/dropout sama untuk semua kondisi; konfigurasi terpilih, parameter trainable, serta epoch aktual dapat berbeda. Setiap kandidat diinisialisasi mandiri. Biaya FT langsung tidak memasukkan FE terpisah.

Training memakai 755 sampel; validation 157; test 179. Data dan target asli tetap dipertahankan sesuai keputusan penggunaan seluruh 411 ID. Tidak ada clipping atau penghapusan kasus buruk. Angka baseline, prediksi negatif, biaya training, serta metrik per posisi tersedia pada tabel hasil dan laporan evaluasi tuning.

Test telah terpapar pada PHASE 5. Hasil ini merupakan evaluasi lanjutan pada test yang sama; kesimpulan dibatasi satu split/seed dan keterbatasan data. Tidak ada CI atau uji signifikansi. R² test masih rendah; evaluasi belum membuktikan generalisasi eksternal atau validasi klinis.

## Rujukan

Hasil terbaru: [laporan evaluasi tuning](tuning_evaluation_report.md), [laporan full grid](tuning_full_report.md), dan [notebook 05](../notebooks/05_Model_Comparison.ipynb). CSV lengkap berada di artifacts/tuning/grid_v1/evaluation_v1/ secara lokal.

Hasil historis sebelum tuning tetap tersedia di artifacts/comparison/ serta [laporan PHASE 2](phase_2_report.md), [PHASE 3](phase_3_report.md), [PHASE 4](phase_4_report.md), dan [PHASE 5](phase_5_report.md). FT historis berasal dari checkpoint FE dan tidak disamakan dengan FT langsung pada hasil terbaru.

Antislop R-17/R-36 PASS: angka dan kesimpulan sesuai CSV terbaru; highlight tidak mengubah pilihan validation. R-02 PASS: tanpa em/en dash atau informasi pasien individual; aturan UI tidak berlaku.
