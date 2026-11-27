# Keputusan penggunaan dataset

Arahan klien disampaikan pada 27 November 2026: gunakan seluruh 411 ID dengan data dan target asli.

| Subset | ID | Sampel lengkap tiga sensor |
|---|---:|---:|
| Training | 287 | 755 |
| Validation | 59 | 157 |
| Testing | 65 | 179 |
| Total | 411 | 1.091 |

Seluruh ID sudah tercakup pada eksperimen yang selesai. Sebanyak 16 pasangan ID-posisi tanpa PulseRate dikeluarkan melalui aturan complete-case; tidak ada ID yang dikeluarkan.

Fitur identik pada 160 sampel (80 kelompok, 62 ID), lima kelompok dengan AHI berbeda, serta AHI maksimum 600 tetap dipertahankan. ID yang terhubung oleh fitur identik berada dalam subset yang sama. Persetujuan penggunaan data tidak menyatakan asal atau validitas klinis label telah diverifikasi.

Tidak diperlukan perubahan data, split, atau training ulang. Metadata dan protokol beku tetap menjadi rekam eksperimen; dokumen ini mencatat keputusan terbaru.
