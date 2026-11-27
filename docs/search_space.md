# Kesepakatan search space

Disepakati pada 27 November 2026: grid search menggunakan kandidat yang sama untuk scratch, feature extraction (FE), dan fine-tuning langsung (FT) pada ResNet50, VGG16, serta VGG19. Dokumen ini menetapkan rencana pencarian, bukan hasil eksperimen. Acuan pelaksanaan dan evaluasi: [desain eksperimen](desain_eksperimen.md).

| Hyperparameter | Keputusan | Alasan |
|---|---|---|
| Learning rate awal | Cari: `1e-6`, `1e-5`, `1e-4`, `1e-3` | Menguji skala pembaruan bobot dengan kandidat yang sama pada ketiga skenario. |
| Dropout head | Cari: `0.1`, `0.3`, `0.5` | Menguji kekuatan regularisasi head. Nilai yang sama diterapkan pada kedua lapisan dropout. |
| Batch size | Tetap `8` | Menyesuaikan memori GPU dan menyeragamkan protokol. Batch size tetap dapat memengaruhi performa. |
| Epoch | Maksimal `50`; early stopping patience `10`; pulihkan bobot terbaik | Membatasi biaya pelatihan dan memilih checkpoint berdasarkan validation. Epoch terbaik dapat berbeda antar kandidat. |
| Optimizer | Tetap Adam | Menyeragamkan algoritme optimisasi agar pencarian berfokus pada learning rate dan dropout. |
| Hidden layer units | Tetap dua lapisan: `128`, lalu `64` | Mempertahankan kapasitas dan struktur head yang sama. |
| Weight decay | Tetap `0` | Mempertahankan konfigurasi awal; pencarian regularisasi difokuskan pada dropout. |
| Loss function | Tetap MSE | Mempertahankan fungsi objektif regresi AHI. |
| Aktivasi head | Tetap ReLU; output linear | Mempertahankan struktur head dan keluaran regresi kontinu. |
| Unfreeze layer/block | Tetap: layer4 ResNet50, block5 VGG16/VGG19; Batch Normalization ResNet50 tetap beku/eval | Menyeragamkan cakupan adaptasi dalam setiap arsitektur. Hanya berlaku pada FT; backbone FE dibekukan dan seluruh bobot scratch dilatih. |

## Aturan pelaksanaan

- Empat learning rate dikalikan tiga dropout menghasilkan **12 kombinasi per kondisi**. Tiga arsitektur dengan tiga skenario menghasilkan **108 pelatihan kandidat**, di luar smoke test.
- Learning rate pada tabel merupakan nilai awal. ReduceLROnPlateau tetap digunakan dengan konfigurasi yang sama untuk seluruh kandidat.
- Gunakan seed `42`, manifest split, dan preprocessing yang sama. Tidak ada pemilihan konfigurasi berdasarkan test.
- Pilih konfigurasi setiap kondisi berdasarkan RMSE validation terkecil, kemudian MAE jika seri. Search space sama tidak mengharuskan konfigurasi terpilih sama.
- Setiap kandidat dimulai dari inisialisasi baru sesuai skenarionya. FE dan FT memakai backbone ImageNet `IMAGENET1K_V1` dengan head baru. FT langsung melatih head dan blok akhir backbone sejak awal, tanpa checkpoint FE atau tahap pelatihan head pendahuluan. Batch Normalization ResNet50 tetap beku/eval pada FE dan FT.
- Simpan konfigurasi, checkpoint, metrik, waktu, dan status setiap kandidat, termasuk kegagalan. Jangan memperluas pencarian berdasarkan hasil test.

## Batasan

Hasil hanya menunjukkan konfigurasi terbaik di antara kandidat yang diuji. FT langsung merupakan strategi yang berdiri sendiri; ketentuan ini menggantikan rancangan FT dari checkpoint FE untuk eksperimen tuning baru. Hasil historis dipertahankan; eksperimen baru disimpan terpisah. Test telah digunakan pada PHASE 5, sehingga evaluasi ulang tidak boleh disebut evaluasi independen baru. Full training tetap memerlukan persetujuan setelah smoke test.
