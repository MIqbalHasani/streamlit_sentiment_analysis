# 📊 Sistem Analisis Sentimen Ulasan Produk

Aplikasi ini adalah alat analisis sentimen interaktif yang dibangun menggunakan Streamlit. Tujuan utama aplikasi ini adalah untuk memungkinkan pengguna menganalisis dan memvisualisasikan data ulasan produk dengan cara yang intuitif dan mendalam. Aplikasi ini mendukung analisis sentimen baik untuk ulasan individu maupun untuk batch ulasan dari file CSV. Selain itu, aplikasi ini menyediakan berbagai jenis visualisasi untuk membantu pengguna mendapatkan wawasan yang lebih baik dari data yang mereka miliki.

## Fitur Utama

1. **Analisis Sentimen Ulasan Individu:**
   - Masukkan ulasan produk secara langsung di aplikasi untuk menganalisis sentimennya.
   - Dapatkan prediksi sentimen (positif atau negatif) berdasarkan model analisis sentimen yang telah dilatih.

2. **Upload dan Analisis Ulasan Massal:**
   - Unggah file CSV yang berisi ulasan produk.
   - Aplikasi akan memproses ulasan, melakukan analisis sentimen, dan menampilkan hasil analisis dalam bentuk tabel dan visualisasi.

3. **Visualisasi Data:**
   - **Distribusi Sentimen:** Grafik interaktif yang menunjukkan persentase masing-masing kategori sentimen dalam dataset.
   - **Distribusi Panjang Ulasan:** Histogram yang menggambarkan distribusi panjang ulasan dalam dataset.
   - **WordCloud:** Visualisasi frekuensi kata yang paling sering muncul dalam ulasan produk.
   - **Tren Sentimen:** Grafik yang menunjukkan perubahan sentimen dari waktu ke waktu (jika data mencakup kolom tanggal).
   - **Rata-rata Panjang Ulasan per Sentimen:** Grafik batang yang menunjukkan panjang rata-rata ulasan untuk setiap kategori sentimen.
   - **Heatmap Korelasi:** Heatmap yang menunjukkan korelasi antara fitur numerik dalam dataset.

4. **Pengunduhan Hasil Analisis:**
   - Unduh hasil analisis sentimen dan visualisasi dalam format CSV atau PNG.

## Teknologi yang Digunakan

- **Streamlit:** Untuk membuat antarmuka pengguna interaktif.
- **Scikit-learn:** Untuk model regresi logistik yang digunakan dalam analisis sentimen.
- **SpaCy:** Untuk tokenisasi dan pemrosesan bahasa alami.
- **Sastrawi:** Untuk stemming kata dalam bahasa Indonesia.
- **Plotly:** Untuk visualisasi grafik interaktif.
- **WordCloud & Matplotlib:** Untuk membuat visualisasi WordCloud.

## Cara Menjalankan Aplikasi

1. **Instalasi Dependencies:**
   Pastikan Anda memiliki Python dan pip terinstal. Kemudian, instal semua dependencies yang diperlukan dengan menjalankan:
   ```bash
   pip install streamlit joblib pandas spacy nltk Sastrawi plotly wordcloud matplotlib
   ```

2. **Unduh Model:**
   Pastikan Anda memiliki file `model_logistic_regression.pkl` yang berisi model regresi logistik yang telah dilatih. Model ini harus ditempatkan di direktori yang sama dengan skrip aplikasi atau sesuaikan path model di dalam kode.

3. **Menjalankan Aplikasi:**
   Jalankan aplikasi Streamlit dengan perintah berikut di terminal:
   ```bash
   streamlit run app.py
   ```
   Gantilah `app.py` dengan nama file Python yang berisi kode aplikasi.

## Petunjuk Penggunaan

- **Analisis Ulasan Individu:** Pilih opsi ini untuk memasukkan ulasan produk secara langsung dan mendapatkan hasil analisis sentimen.
- **Upload dan Analisis:** Unggah file CSV yang berisi ulasan produk untuk analisis massal dan dapatkan hasil analisis dalam bentuk tabel dan visualisasi.
- **Visualisasi:** Unggah file CSV yang sudah diproses dan pilih jenis visualisasi yang diinginkan untuk melihat grafik dan analisis yang berbeda.

## Contoh

1. **Analisis Ulasan Individu:**
   Masukkan ulasan produk di kolom teks dan klik "Analisis Sentimen" untuk mendapatkan hasil analisis.

2. **Upload dan Analisis Ulasan dari File CSV:**
   Unggah file CSV dengan kolom `Review` untuk menganalisis sentimen dan melihat hasil dalam bentuk tabel serta grafik.

3. **Visualisasi Data:**
   Unggah file CSV yang sudah diproses dan pilih jenis visualisasi (seperti distribusi sentimen, WordCloud, dsb.) untuk mendapatkan wawasan lebih lanjut dari data.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan buka issue tracker dan ajukan pull request dengan perubahan yang diusulkan.
