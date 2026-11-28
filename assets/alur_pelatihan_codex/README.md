# Ilustrasi peran subset dalam regresi AHI

Tiga ilustrasi untuk memahami penelitian dan menjelaskannya kepada klien. Dibuat dengan tool imagegen bawaan; prompt lengkap tersedia di [PROMPTS.txt](PROMPTS.txt). Versi Gemini dipertahankan.

## Urutan penjelasan

1. [Peran tiga subset](01_peran_tiga_subset.png): train menjadi sumber pembelajaran bobot dan parameter skala fitur; validation mengarahkan pelatihan dan pemilihan; test mengukur hasil setelah pilihan dikunci. Pembagian data tetap sepanjang penelitian.
2. [Siklus train dan validation](02_siklus_train_validation.png): train memperbarui bobot yang dilatih, kemudian validation memeriksa model tanpa memperbarui bobot secara langsung. Hasil validation mengendalikan penyimpanan checkpoint, scheduler, dan penghentian dini. Setelah pelatihan, pulihkan bobot terbaik.
3. [Pencarian pengaturan](03_pencarian_pengaturan.png): setiap kombinasi learning rate awal dan dropout dilatih secara mandiri pada train, kemudian dibandingkan berdasarkan validation. Test tidak digunakan untuk memilih.

## Catatan pendamping

Bobot merupakan parameter yang dipelajari selama pelatihan. Learning rate awal dan dropout merupakan hyperparameter yang dicari. Empat learning rate awal dan tiga dropout menghasilkan 12 kandidat per kondisi, total 108 untuk sembilan kondisi. Validation memilih satu kandidat per kondisi serta satu CNN utama dari sembilan kandidat terpilih, berdasarkan RMSE terkecil dan MAE jika seri.

Checkpoint tiap kandidat dipilih melalui MSE validation terendah. Scheduler dapat menurunkan learning rate selama pelatihan berdasarkan MSE validation; nilai awalnya tetap berasal dari grid. Siklus pada gambar kedua berakhir saat early stopping atau batas maksimal 50 epoch tercapai.

Sesudah pilihan dibekukan, evaluasi test mencakup sembilan CNN terpilih dan dua baseline, tanpa melatih ulang CNN atau mengganti pilihan dari hasil test. Test pernah dipakai pada PHASE 5; evaluasi tuning merupakan evaluasi lanjutan setelah paparan.

Objek input, koneksi model, dan grafik merupakan ilustrasi konseptual, bukan data pasien, arsitektur rinci, atau hasil pengukuran. Warna dan nama membedakan subset: biru untuk train, jingga untuk validation, hijau kebiruan untuk test.

## Acuan dan pemeriksaan

Isi diperiksa terhadap [desain eksperimen](../../docs/desain_eksperimen.md) dan [search space](../../docs/search_space.md). Jumlah sampel mengikuti acuan: train 755, validation 157, test 179. Pemeriksaan meliputi pembukaan ketiga PNG, dimensi, hash salinan, keterbacaan label, arah panah, serta pemisahan pembelajaran bobot dari pemilihan hyperparameter.

PASS, hard gate: aset diminta pengguna; angka berasal dari dokumen proyek; tidak ada data pasien atau klaim performa.
PASS, purpose gate: warna membedakan subset; kunci menunjukkan bobot tetap atau pilihan beku; panah menjelaskan aliran dan pengulangan.
PASS, liveliness: ENERGY 2 / RHYTHM 2 / MOTION 0; objek model, alat penyetel, kaca pembesar, dan gerbang menjadi fokus sesuai fungsi.
PASS, craftsmanship: ketiga gambar diperiksa secara visual; grafik merupakan sketsa, tidak ada kontrol interaktif; berkas disalin tanpa menimpa aset lama.

Perubahan hanya menambah folder ilustrasi ini. Tidak ada perubahan kode, notebook, data, protokol, atau hasil eksperimen. Tidak ada pelatihan atau evaluasi ulang. Pengujian implementasi tidak diperlukan untuk penambahan aset ini. Rekomendasi: jelaskan gambar pertama, gunakan gambar kedua untuk pertanyaan tentang epoch, lalu gambar ketiga untuk pertanyaan tentang pencarian parameter.
