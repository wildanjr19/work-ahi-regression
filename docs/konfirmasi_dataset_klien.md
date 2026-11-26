# Konfirmasi hasil audit dataset

Audit awal ketiga dataset selesai. Sebelum pembagian data training, validation, dan testing, mohon konfirmasi temuan berikut.

## Temuan

- Tidak ditemukan nilai kosong, infinite, atau duplikasi kunci pasien-posisi-IMF. AHI konsisten untuk pasien dan posisi yang sama dalam maupun antar sensor.
- Sebanyak 160 sampel dari 62 ID memiliki fitur ketiga sensor identik pada posisi yang sama, membentuk 80 kelompok. Lima kelompok memiliki AHI berbeda antar-ID. Temuan ini belum membuktikan oversampling atau kesalahan data.
- AHI maksimum mencapai 600 pada satu sampel posisi Right. Nilai tersebut tetap dipertahankan.

## Mohon konfirmasi

1. Apakah dataset merupakan data asli atau sudah melalui oversampling, penggandaan, atau proses sintetis?
2. Apakah ID dengan fitur identik mewakili pasien berbeda atau salinan/pengukuran ulang? Mohon periksa AHI pada lima kelompok yang berbeda target.
3. Apakah AHI 600 benar sesuai sumber? Mohon konfirmasi satuan dan cara perhitungannya.

## Usulan penanganan

Pertahankan data dan target asli. Untuk mencegah kebocoran data, tempatkan ID yang terhubung oleh fitur identik dalam subset yang sama. Penanganan akhir mengikuti hasil konfirmasi sumber data.
