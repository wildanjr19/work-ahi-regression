# Keadaan data awal

Data berupa tabel fitur Intrinsic Mode Function (IMF) dari tiga sensor, bukan sinyal mentah atau citra. Setiap workbook memiliki sheet Supine (telentang), Left (miring kiri), dan Right (miring kanan).

## Jumlah data

Jumlah baris berikut mencakup ketiga sheet, tanpa header.

| Workbook | Baris IMF | ID unik | Pasangan ID-posisi | Kolom |
|---|---:|---:|---:|---:|
| Flow Patient.xlsx | 19.632 | 411 | 1.107 | 15 |
| PulseRate.xlsx | 14.700 | 411 | 1.091 | 15 |
| Pleth.xlsx | 16.417 | 411 | 1.107 | 15 |

Total ketiga workbook adalah 50.749 baris IMF, dengan 411 ID unik yang sama antar sensor. Satu baris mewakili satu IMF untuk satu ID, posisi, dan sensor. Satu sampel penelitian adalah pasangan `(id, Posisi)`, yang memiliki sejumlah baris IMF per sensor.

Dari 1.107 pasangan ID-posisi, 1.091 memiliki ketiga sensor: 411 Supine, 330 Left, dan 350 Right. Sebanyak 16 pasangan tidak memiliki PulseRate (9 Left dan 7 Right). Seluruh 411 ID tetap tercakup dalam data lengkap tiga sensor.

## Kolom dan variabel

Ketiga workbook memiliki struktur yang sama: 11 fitur numerik, tiga kolom metadata, dan satu target.

| Kolom | Penjelasan | Peran |
|---|---|---|
| `id` | Identitas pasien; dibaca sebagai string agar nol di awal tetap terjaga. | Metadata |
| `Posisi` | Posisi tidur: Supine, Left, atau Right. | Metadata; input posisi terpisah |
| `IMF` | Indeks komponen Intrinsic Mode Function, dari IMF1 hingga IMF50; jumlah komponen bervariasi antar sampel dan sensor. | Metadata pengurutan |
| `Energi IMF` | Ukuran energi komponen IMF. | Fitur sensor |
| `Energi Relatif (%)` | Persentase kontribusi energi IMF terhadap energi acuan. | Fitur sensor |
| `Frekuensi Dominan (Hz)` | Frekuensi yang paling menonjol pada IMF. | Fitur sensor |
| `Rentang Frek (Hz)` | Ukuran rentang frekuensi IMF. | Fitur sensor |
| `Mean Period (s)` | Rata-rata periode osilasi IMF, dalam detik. | Fitur sensor |
| `Mean Amplitudo` | Rata-rata amplitudo IMF. | Fitur sensor |
| `STD Amplitudo` | Simpangan baku amplitudo IMF, sebagai ukuran variasinya. | Fitur sensor |
| `Instant Freq (Hz)` | Fitur frekuensi sesaat IMF. | Fitur sensor |
| `STD Instant Freq (Hz)` | Simpangan baku frekuensi sesaat IMF. | Fitur sensor |
| `Entropi` | Ukuran ketidakteraturan IMF. | Fitur sensor |
| `Korelasi dengan sinyal asli` | Ukuran hubungan IMF dengan sinyal asalnya. | Fitur sensor |
| `AHI` | Nilai Apnea-Hypopnea Index untuk pasangan ID-posisi; sama pada seluruh IMF dan ketiga sensor untuk pasangan tersebut. | Target regresi kontinu |

Penjelasan fitur bersifat konseptual berdasarkan nama kolom. Rumus ekstraksi, jenis entropi/korelasi, serta acuan energi relatif tidak tercantum dalam sumber proyek. ID, posisi, indeks IMF, dan AHI tidak termasuk dalam 11 fitur sensor.

## Catatan kualitas awal

- Audit tidak menemukan nilai kosong, numerik invalid/infinite, duplikasi kunci `(id, Posisi, IMF)` dalam tiap sensor, atau konflik AHI untuk pasangan ID-posisi yang sama.
- AHI complete-case berada pada rentang 0 hingga 600; nilai ekstrem tetap dipertahankan.
- Sebanyak 160 sampel membentuk 80 kelompok dengan fitur ketiga sensor identik antar-ID pada posisi yang sama. Lima kelompok memiliki AHI berbeda antar-ID. Temuan ini belum membuktikan kesamaan identitas atau oversampling; data dan target asli dipertahankan.
- Header `AHI ` pada Flow/Right memiliki spasi di akhir; pipeline menormalkannya di memori. Workbook asli tidak diubah.

Sumber: [laporan audit PHASE 0](phase_0_report.md), [konfigurasi kolom](../src/config.py), dan [brief dataset](../BRIEF.md). Hash SHA-256 ketiga workbook telah dicocokkan dengan audit tersimpan.
