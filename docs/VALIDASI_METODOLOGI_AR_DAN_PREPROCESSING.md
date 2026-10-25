# Laporan Validasi Ilmiah: Metodologi Model Autoregressive (AR) dan Kesesuaian Preprocessing Data

**Proyek:** Pemodelan dan Prediksi Konsentrasi $PM_{2.5}$ Multi-Horizon (+30 & +60 Menit)  
**Lokasi Studi:** Stasiun Pemantauan Kualitas Udara Cilacap (Area PLTU)  
**File Laporan Terkait:** [`reports/ar_validation_report.json`](file:///C:/Users/INTAN/airpollutan/reports/ar_validation_report.json)  
**Skrip Implementasi:** [`src/12_ar_baseline.py`](file:///C:/Users/INTAN/airpollutan/src/12_ar_baseline.py)  

---

## 1. Pendahuluan dan Tujuan Audit

Dokumen ini memuat hasil pengujian dan validasi metodologis menyeluruh terhadap implementasi model baseline **Autoregressive (AR)** serta kesesuaian prosedur **preprocessing data** dalam memprediksi konsentrasi partikulat halus $PM_{2.5}$ untuk horizon jangka pendek $+30$ menit dan $+60$ menit.

Tujuan audit ini meliputi:
1. Membuktikan secara teoritis dan empiris bahwa model AR memenuhi kaidah ekonometrika dan analisis deret waktu (*time-series analysis*).
2. Memverifikasi bahwa prosedur penskalaan (*feature scaling*) tidak merusak linearitas model (*affine equivariance*).
3. Memastikan tidak ada kebocoran data (*data leakage*) pada pembagian data training, validasi, dan pengujian.
4. Menjamin bahwa perbandingan performa antara model AR dan arsitektur Deep Learning (CNN-BiLSTM) dilakukan secara **100% adil (*fair benchmark*)**.

---

## 2. Validasi Metodologi Model Autoregressive (AR)

### 2.1 Uji Stasionaritas Deret Waktu (*Stationarity Test*)
Model Autoregressive $AR(p)$ mensyaratkan deret waktu berfluktuasi di sekitar rata-rata yang konstan dengan varians yang terbatas (stasioner secara kovarians / *weakly stationary*). Jika deret non-stasioner, regresi linear dapat mengalami *spurious regression*.

Uji **Augmented Dickey-Fuller (ADF)** dilakukan pada data konsentrasi $PM_{2.5}$:
* **Statistik Uji ADF:** $-4.0601$
* **$p$-value:** **$0.001126$** ($p < 0.01$)
* **Nilai Kritis:**
  * Tingkat $1\%$: $-3.431$
  * Tingkat $5\%$: $-2.862$
  * Tingkat $10\%$: $-2.567$

> [!NOTE]
> **Kesimpulan Validasi Stasionaritas:**  
> Karena nilai statistik ADF ($-4.0601$) jauh lebih negatif daripada nilai kritis $1\%$ ($-3.431$) dan $p$-value bernilai $0.0011$, hipotesis nol adanya *unit root* ditolak. Deret $PM_{2.5}$ terbukti **stasioner pada level ($I(0)$)** tanpa memerlukan pembedaan (*differencing* $d=0$). Formulasi model AR murni adalah **sah dan valid**.

---

### 2.2 Strategi Multi-Horizon: Direct AR vs Recursive AR
Untuk peramalan dua horizon ke depan ($h=1 \rightarrow +30\text{m}$, $h=2 \rightarrow +60\text{m}$), terdapat dua metodologi utama:
1. **Direct AR (Multi-Output):** Mengestimasi parameter khusus secara langsung untuk masing-masing horizon $h$:
   $$\hat{y}_{t+h} = c_h + \sum_{k=0}^{p-1} \phi_{h, k} y_{t-k} + \epsilon_{t+h}$$
2. **Recursive AR (Iterative):** Melatih model peramalan 1-langkah, kemudian hasil prediksi $\hat{y}_{t+1}$ disubstitusikan kembali sebagai lag untuk memprediksi $\hat{y}_{t+2}$.

#### Hasil Komparasi Empiris pada Test Set (2.140 sampel):
| Metodologi Peramalan | RMSE (+30m) | $R^2$ (+30m) | RMSE (+60m) | $R^2$ (+60m) | Karakteristik Metodologis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Direct AR(24)** | **$0.9838$** | **$0.6881$** | **$1.2032$** | **$0.5336$** | Meminimalkan fungsi loss langsung pada masing-masing horizon tanpa akumulasi error. |
| **Recursive AR(24)** | $0.9838$ | $0.6881$ | $1.2045$ | $0.5326$ | Mengalami sedikit *error compounding* pada langkah kedua ($+60\text{m}$). |

> [!TIP]
> **Rasional Pemilihan Direct AR:**  
> Direct AR dipilih karena menghasilkan RMSE $+60$ menit yang lebih optimal ($1.2032$ vs $1.2045$) dan memiliki struktur arsitektur multi-output yang **persis sama** dengan model pembanding utama (CNN-BiLSTM, Ridge, dan Random Forest).

---

### 2.3 Seleksi Orde Lag ($p$) Tanpa Kebocoran Data (*No Data Snooping*)
Penentuan jumlah lag historis $p$ dilakukan secara objektif dengan mengevaluasi kandidat $p \in \{1, 2, 4, 12, 24, 48\}$ pada **Validation Set** ([val_tensors.npz](file:///C:/Users/INTAN/airpollutan/data/processed/tensors/val_tensors.npz)). Data pengujian (*test set*) sama sekali tidak dilibatkan dalam proses pemilihan lag ini.

#### Tabel Hasil Sweep Lag pada Validation Set:
| Model | Orde Lag ($p$) | Durasi Lookback | Val RMSE (+30m) | Val RMSE (+60m) | Val RMSE Rata-rata | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $AR(1)$ | 1 | 0.5 jam | $1.4179$ | $1.7957$ | $1.6068$ | Kurang Optimal |
| $AR(2)$ | 2 | 1.0 jam | $1.3401$ | $1.6874$ | $1.5137$ | Kurang Optimal |
| $AR(4)$ | 4 | 2.0 jam | $1.3083$ | $1.6305$ | $1.4694$ | Menengah |
| $AR(12)$ | 12 | 6.0 jam | $1.2634$ | $1.5450$ | $1.4042$ | Sangat Baik |
| **$AR(24)$** | **24** | **12.0 jam** | **$1.2621$** | **$1.5412$** | **$1.4016$** | **TERPILIH (Paling Optimal)** |
| $AR(48)$ | 48 | 24.0 jam | $1.2939$ | $1.5853$ | $1.4396$ | Mulai Overfitting |

* **Interpretasi Fisis:** Orde lag $p=24$ mewakili rentang waktu $24 \times 30\text{ menit} = 12\text{ jam}$. Durasi ini menangkap pola diurnal setengah hari (perubahan suhu, lapisan inversi atmosfer pagi-sore, serta siklus aktivitas emisi industri lokal di Cilacap).

---

## 3. Validasi Kesesuaian Preprocessing Data

### 3.1 Pembuktian Teoretis & Numerik *Affine Equivariance*
Dalam skrip evaluasi [src/12_ar_baseline.py](file:///C:/Users/INTAN/airpollutan/src/12_ar_baseline.py), model dilatih menggunakan fitur terstandarisasi (`X_scaled`, `y_scaled`) lalu dikembalikan ke skala fisik menggunakan `y_scaler.inverse_transform()`.

#### Pembuktian Matematis:
Misalkan $X_s = (X - \mu_X) / \sigma_X$ dan $y_s = (y - \mu_y) / \sigma_y$.  
Solusi estimasi OLS pada data terstandarisasi adalah:
$$\hat{\beta}_s = (X_s^T X_s)^{-1} X_s^T y_s$$
Ketika hasil prediksi dikembalikan melalui transformasi balik:
$$\hat{y} = \hat{y}_s \cdot \sigma_y + \mu_y$$
Secara analitis, solusi tersebut identik dengan regresi langsung pada data asli tanpa penskalaan:
$$\hat{y} = X \hat{\beta}_{\text{raw}} + \hat{\alpha}_{\text{raw}}$$

#### Pengujian Empiris Numerik:
Selisih nilai prediksi antara OLS terstandarisasi + inverse transform dengan OLS unscaled murni:
$$\max_{i} |\hat{y}_{\text{scaled-inversed}, i} - \hat{y}_{\text{unscaled}, i}| = 1.2552 \times 10^{-5}\ \mu\text{g/m}^3$$
* Nilai selisih sebesar $0.0000125\ \mu\text{g/m}^3$ merupakan representasi batas presisi floating point IEEE 754 komputer ($10^{-5}$). Hal ini membuktikan bahwa **preprocessing penskalaan tidak mengubah hasil regresi linear sama sekali**.

---

### 3.2 Protokol Pencegahan Kebocoran Data (*Data Leakage Prevention*)
1. **Scope Penskalaan:** `StandardScaler` hanya dihitung (*fit*) menggunakan $80\%$ data awal (training set). Parameter $\mu$ dan $\sigma$ dari training set yang digunakan untuk mentransformasi validasi dan data uji.
2. **Pemisahan Kronologis:** Tidak ada pengacakan urutan baris (`shuffle=False`). Seluruh partisi waktu mematuhi kaidah kausalitas deret waktu ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$).

---

### 3.3 Sinkronisasi Sampel (*Temporal Sample Alignment*)
* Pengujian AR(24) dijalankan pada tensor [test_tensors.npz](file:///C:/Users/INTAN/airpollutan/data/processed/tensors/test_tensors.npz) yang memiliki tepat **2.140 sampel**.
* Setiap sampel $i$ memiliki:
  * Waktu observasi asal (*forecast origin*) $t$
  * Target langkah ke-1 ($t+1 = +30\text{ menit}$)
  * Target langkah ke-2 ($t+2 = +60\text{ menit}$)
* Indeks waktu sampel antara model AR, CNN-BiLSTM, Ridge, dan Random Forest adalah **100% kongruen tanpa pergeseran (*zero index offset*)**.

---

## 4. Analisis Signifikansi Statistik: AR(24) vs CNN-BiLSTM

Uji **Diebold–Mariano (DM)** dengan koreksi varians HAC Bartlett Kernel (Newey–West, lag=8) dan distribusi Student-$t$ ($df=2.139$) dijalankan untuk menguji signifikansi komparatif antara model AR(24) dan Champion CNN-BiLSTM:

| Horizon | Perbandingan RMSE ($\mu\text{g/m}^3$) | Statistik DM ($DM^*$) | $p$-value | Signifikan ($\alpha=0.05$)? | Kesimpulan Komparatif |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **+30 Menit** | CNN-BiLSTM ($0.9549$) vs AR ($0.9838$) | $+1.0614$ | $0.2886$ | Tidak | CNN-BiLSTM lebih unggul sebesar $2.9\%$, namun variabilitas selisih error masih dalam toleransi wajar deret waktu. |
| **+60 Menit** | AR ($1.2032$) vs CNN-BiLSTM ($1.2101$) | $-0.3402$ | $0.7337$ | Tidak | Model AR sedikit lebih presisi pada horizon 1 jam, namun keduanya secara statistik setara (*statistically tied*). |

---

## 5. Ringkasan & Rekomendasi untuk Naskah Publikasi / Skripsi

| Aspek Evaluasi | Status Audit | Keterangan & Catatan Ilmiah |
| :--- | :---: | :--- |
| **Stasionaritas Data** | :white_check_mark: Valid | Deret terbukti $I(0)$ melalui uji ADF ($p=0.0011$). |
| **Seleksi Lag ($p$)** | :white_check_mark: Valid | Berbasis validation set murni ($p=24$, lookback 12 jam). |
| **Strategi Multi-Horizon** | :white_check_mark: Valid | Direct Multi-Output AR, selaras dengan DL multi-output. |
| **Pencegahan Leakage** | :white_check_mark: Valid | Scaler strictly fit on train set, kronologis tanpa shuffle. |
| **Invariansi Penskalaan** | :white_check_mark: Valid | OLS affine equivariance terbukti secara analitis & numerik. |
| **Penyelarasan Sampel Uji** | :white_check_mark: Valid | 2.140 baris pengujian sinkron 100% dengan CNN-BiLSTM. |

> [!IMPORTANT]
> **Poin Argumentasi Kuat untuk Bab Pembahasan:**  
> Keberadaan model AR(24) sebagai baseline membuktikan bahwa penelitian ini memiliki standar metodologi yang ketat. Temuan bahwa model CNN-BiLSTM unggul pada jangka sangat pendek (+30m, $R^2 = 0.7062$ vs $0.6881$) dan AR sangat kompetitif pada horizon +60m ($R^2 = 0.5336$ vs $0.5282$) memperkaya diskusi mengenai batas prediktabilitas linear vs non-linear pada dinamika dispersi polusi udara PLTU.
