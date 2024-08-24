import streamlit as st
import joblib
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import concurrent.futures

# Pastikan NLTK resources di-download
nltk.download('stopwords')

# Muat model yang telah dilatih
model = joblib.load('model_logistic_regression.pkl')

# Muat model bahasa Indonesia dari spacy
nlp = spacy.load('xx_ent_wiki_sm')

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

# Fungsi untuk tokenisasi menggunakan spacy
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Fungsi untuk menghapus stop words
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

# Fungsi untuk stemming satu token
def stem_token(token):
    return stemmer.stem(token)

# Fungsi untuk stemming batch token
def stem_tokens(tokens):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stemmed_tokens = list(executor.map(stem_token, tokens))
    return stemmed_tokens

# Fungsi pra-pemrosesan teks
def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)

# Fungsi untuk memproses dan memprediksi sentimen
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    return model.predict([processed_text])[0]

# Fungsi untuk menampilkan grafik distribusi sentimen
def plot_sentiment_distribution(data):
    sentiment_counts = data['Sentiment'].value_counts()
    fig = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values,
                 title='Distribusi Sentimen Ulasan')
    st.plotly_chart(fig)

# Fungsi untuk menampilkan grafik distribusi panjang ulasan
def plot_review_length_distribution(data, bins):
    data['Review_Length'] = data['Review'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    fig = px.histogram(data, x='Review_Length', nbins=bins, title='Distribusi Panjang Ulasan')
    st.plotly_chart(fig)

# Fungsi untuk menampilkan frekuensi kata menggunakan WordCloud
def plot_word_cloud(data, max_words, colormap):
    if 'Processed_Review' in data.columns:
        data['Processed_Review'] = data['Processed_Review'].fillna('').astype(str)
        text = ' '.join(data['Processed_Review'])
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap, max_words=max_words).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Simpan ke BytesIO untuk unduh
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        st.pyplot(fig)
        
        st.download_button(
            label="Unduh WordCloud",
            data=buffer,
            file_name='wordcloud.png',
            mime='image/png'
        )
    else:
        st.write("Kolom 'Processed_Review' tidak ditemukan dalam data.")

# Fungsi untuk menampilkan detail analisis sentimen
def plot_detailed_sentiment_analysis(data):
    sentiment_summary = data['Sentiment'].value_counts().reset_index()
    sentiment_summary.columns = ['Sentiment', 'Count']
    sentiment_summary['Percentage'] = (sentiment_summary['Count'] / sentiment_summary['Count'].sum() * 100).round(2)

    # Tampilkan tabel detail analisis sentimen
    st.write("### Rincian Analisis Sentimen")
    st.write("Tabel berikut menunjukkan rincian tentang setiap kategori sentimen, termasuk jumlah ulasan dan persentase dari total ulasan. Ini memberikan gambaran umum tentang distribusi sentimen dalam dataset.")
    st.dataframe(sentiment_summary, use_container_width=True)

# Fungsi untuk menampilkan grafik tren sentimen dari waktu ke waktu
def plot_sentiment_trend(data):
    if 'Tanggal' in data.columns:
        sentiment_trend = data.groupby('Tanggal')['Sentiment'].value_counts().unstack().fillna(0)
        fig = px.line(sentiment_trend, x=sentiment_trend.index, y=sentiment_trend.columns,
                      title='Tren Sentimen dari Waktu ke Waktu')
        fig.update_layout(xaxis_title='Tanggal', yaxis_title='Jumlah Ulasan')
        st.write("### Tren Sentimen dari Waktu ke Waktu")
        st.write("Grafik ini menunjukkan perubahan jumlah ulasan berdasarkan sentimen dari waktu ke waktu. Ini berguna untuk mengidentifikasi pola sentimen musiman atau tren jangka panjang.")
        st.plotly_chart(fig)
    else:
        st.write("Data tidak memiliki kolom 'Tanggal'.")

# Fungsi untuk menampilkan grafik rata-rata panjang ulasan per sentimen
def plot_average_review_length_per_sentiment(data):
    data['Review_Length'] = data['Review'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    avg_length = data.groupby('Sentiment')['Review_Length'].mean().reset_index()
    fig = px.bar(avg_length, x='Sentiment', y='Review_Length', text='Review_Length',
                 title='Rata-rata Panjang Ulasan per Sentimen')
    fig.update_layout(xaxis_title='Sentimen', yaxis_title='Panjang Rata-rata Ulasan')
    st.plotly_chart(fig)

# Fungsi untuk menampilkan heatmap korelasi antara fitur
def plot_correlation_heatmap(data):
    # Pilih hanya kolom numerik untuk perhitungan korelasi
    numeric_data = data.select_dtypes(include=['number'])
    
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        fig = px.imshow(corr_matrix, text_auto=True, title='Heatmap Korelasi Antar Fitur')
        st.write("### Heatmap Korelasi Antar Fitur")
        st.write("Heatmap ini menunjukkan korelasi antar fitur numerik dalam dataset. Warna yang lebih gelap menunjukkan korelasi yang lebih kuat antara fitur-fitur tersebut.")
        st.plotly_chart(fig)
    else:
        st.write("Data tidak memiliki cukup fitur numerik untuk membuat heatmap.")

# Fungsi untuk menyaring dan menampilkan data berdasarkan kategori atau rentang waktu
def filter_data(data):
    # Menyaring berdasarkan kategori atau waktu jika ada kolom kategori dan waktu
    if 'Kategori' in data.columns:
        category_filter = st.multiselect("Pilih Kategori", options=data['Kategori'].unique(), default=data['Kategori'].unique())
        data = data[data['Kategori'].isin(category_filter)]

    if 'Tanggal' in data.columns:
        start_date = st.date_input("Tanggal Mulai", value=pd.to_datetime(data['Tanggal'].min()))
        end_date = st.date_input("Tanggal Akhir", value=pd.to_datetime(data['Tanggal'].max()))
        data = data[(data['Tanggal'] >= start_date) & (data['Tanggal'] <= end_date)]

    return data

# Fungsi untuk menampilkan grafik distribusi sentimen dengan Plotly
def plot_interactive_sentiment_distribution(data):
    sentiment_counts = data['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Distribusi Sentimen Ulasan')
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1])
    st.plotly_chart(fig)

# Fungsi untuk menampilkan grafik distribusi panjang ulasan dengan Plotly
def plot_interactive_review_length_distribution(data, bins):
    data['Review_Length'] = data['Review'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    fig = px.histogram(data, x='Review_Length', nbins=bins, title='Distribusi Panjang Ulasan')
    fig.update_layout(xaxis_title='Panjang Ulasan', yaxis_title='Jumlah Ulasan')
    st.plotly_chart(fig)

# Fungsi untuk memproses dan memprediksi sentimen dari file CSV
def process_and_predict(file):
    try:
        data = pd.read_csv(file, delimiter=',', quotechar='"', on_bad_lines='skip', encoding='utf-8')
        if 'Review' in data.columns:
            data['Review'] = data['Review'].fillna('')
            data['Processed_Review'] = data['Review'].apply(preprocess_text)
            data['Processed_Review'] = data['Processed_Review'].fillna('').astype(str)
            data['Sentiment'] = data['Processed_Review'].apply(lambda x: predict_sentiment(x))

            st.write("### Hasil Analisis Sentimen")
            st.write("Tabel berikut menunjukkan ulasan dan hasil analisis sentimen. Setiap ulasan dikategorikan dalam salah satu dari beberapa sentimen berdasarkan model analisis.")
            st.dataframe(data[['Review', 'Sentiment']], use_container_width=True, height=300)

            # Tampilkan detail analisis sentimen
            plot_detailed_sentiment_analysis(data)
            
            # Tampilkan grafik distribusi sentimen
            st.write("### Distribusi Sentimen Ulasan")
            st.write("Grafik berikut memberikan pandangan interaktif tentang distribusi sentimen dalam dataset. Dengan menggunakan grafik pie, Anda dapat melihat persentase dan jumlah ulasan untuk setiap kategori sentimen.")
            plot_interactive_sentiment_distribution(data)
            
            # Tampilkan grafik distribusi panjang ulasan
            st.write("### Distribusi Panjang Ulasan")
            st.write("Grafik ini menunjukkan variasi panjang ulasan dalam dataset, memberikan wawasan tentang bagaimana panjang ulasan tersebar di seluruh data.")
            plot_interactive_review_length_distribution(data, bins=30)  # Default value for bins
            
            # Tampilkan grafik frekuensi kata
            st.write("### WordCloud dari Ulasan")
            st.write("WordCloud ini menampilkan kata-kata yang paling sering muncul dalam ulasan. Kata-kata yang lebih besar menunjukkan frekuensi yang lebih tinggi. Ini dapat membantu mengidentifikasi tema atau topik utama yang sering dibahas dalam ulasan.")
            plot_word_cloud(data, max_words=100, colormap='viridis')  # Default values
            
            # Tambahkan tombol untuk mengunduh hasil analisis
            st.download_button(
                label="Unduh Hasil Analisis",
                data=data.to_csv(index=False, encoding='utf-8'),
                file_name='hasil_analisis.csv',
                mime='text/csv'
            )
        else:
            st.write("File CSV tidak memiliki kolom 'Review'.")
    except pd.errors.EmptyDataError:
        st.error("File CSV kosong atau tidak valid.")
    except pd.errors.ParserError as e:
        st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Streamlit aplikasi
st.title('📊 Sistem Analisis Sentimen Ulasan Produk')

option = st.sidebar.selectbox('Pilih Opsi', ['Analisis Ulasan', 'Upload dan Analisis', 'Visualisasi'])

# Menambahkan sidebar untuk navigasi
st.sidebar.header('Navigasi')
st.sidebar.write("Pilih opsi dari menu di sebelah kiri untuk menggunakan aplikasi ini.")
st.sidebar.write("### Opsi:")
st.sidebar.write("1. **Analisis Ulasan**: Masukkan ulasan produk untuk menganalisis sentimennya.")
st.sidebar.write("2. **Upload dan Analisis**: Unggah file CSV yang berisi ulasan produk untuk analisis sentimen massal.")
st.sidebar.write("3. **Visualisasi**: Pilih jenis visualisasi untuk melihat grafik dan analisis yang berbeda dari data yang telah diproses.")
st.sidebar.write("### Petunjuk Penggunaan:")
st.sidebar.write("1. Pilih opsi dari menu navigasi.")
st.sidebar.write("2. Ikuti instruksi yang muncul untuk masing-masing opsi.")
st.sidebar.write("3. Untuk analisis batch, pastikan file CSV memiliki kolom 'Review'.")
st.sidebar.write("4. Gunakan filter dan opsi visualisasi untuk mendapatkan wawasan lebih lanjut dari data.")

if option == 'Analisis Ulasan':
    st.subheader('Analisis Ulasan Individu')
    st.write("Masukkan ulasan produk di bawah ini untuk menganalisis sentimennya. Sistem akan memberikan prediksi sentimen berdasarkan ulasan yang Anda masukkan.")
    
    user_input = st.text_area("Masukkan Ulasan Produk:", height=200)
    
    if st.button('Analisis Sentimen'):
        if user_input:
            try:
                sentiment = predict_sentiment(user_input)
                st.write(f"**Sentimen Ulasan:** {sentiment}")
                st.write("### Penjelasan Sentimen Ulasan")
                st.write("Hasil prediksi sentimen ini menunjukkan kategori sentimen untuk ulasan yang diberikan. Kategori ini berdasarkan model analisis sentimen yang telah dilatih dengan data ulasan. Jika hasilnya positif, berarti ulasan cenderung positif, dan sebaliknya.")
            except Exception as e:
                st.write(f"Terjadi kesalahan: {e}")
        else:
            st.write("Silakan masukkan ulasan untuk dianalisis.")
    
elif option == 'Upload dan Analisis':
    st.subheader('Unggah dan Analisis Ulasan dari File CSV')
    st.write("Unggah file CSV yang berisi ulasan produk untuk menganalisis sentimennya. Pastikan file CSV memiliki kolom 'Review'.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        process_and_predict(uploaded_file)

elif option == 'Visualisasi':
    st.subheader('Visualisasi Data Sentimen')
    st.write("Pilih jenis visualisasi yang Anda inginkan dan unggah file CSV yang sudah diproses untuk melihat grafik yang sesuai.")
    
    # Pilih jenis visualisasi
    visualization_option = st.selectbox("Pilih Jenis Visualisasi", ["Distribusi Sentimen", "Distribusi Panjang Ulasan", "WordCloud", "Tren Sentimen", "Rata-rata Panjang Ulasan per Sentimen", "Heatmap Korelasi"])
    
    # Pilih file CSV yang sudah diproses
    file = st.file_uploader("Unggah File CSV yang Sudah Diproses", type="csv")
    
    if file is not None:
        data = pd.read_csv(file, encoding='utf-8')
        
        if 'Sentiment' in data.columns:
            # Filter data jika ada opsi
            data = filter_data(data)

            # Visualisasi berdasarkan pilihan
            if visualization_option == "Distribusi Sentimen":
                st.write("### Distribusi Sentimen Ulasan")
                st.write("Grafik ini menunjukkan persentase dari setiap kategori sentimen dalam ulasan. Ini memberikan gambaran visual yang jelas tentang distribusi sentimen.")
                plot_interactive_sentiment_distribution(data)
            
            elif visualization_option == "Distribusi Panjang Ulasan":
                bins = st.slider("Jumlah Bin untuk Histogram", min_value=10, max_value=100, value=30, step=1)
                st.write("### Distribusi Panjang Ulasan")
                st.write("Grafik ini menunjukkan distribusi panjang ulasan dalam dataset. Anda dapat menyesuaikan jumlah bin untuk histogram agar lebih sesuai dengan data.")
                plot_interactive_review_length_distribution(data, bins)
            
            elif visualization_option == "WordCloud":
                max_words = st.slider("Maksimum Jumlah Kata di WordCloud", min_value=10, max_value=200, value=100, step=10)
                colormap = st.selectbox("Pilih Skema Warna", ["viridis", "plasma", "inferno", "magma", "cividis"])
                st.write("### WordCloud dari Ulasan")
                st.write("WordCloud ini menampilkan kata-kata yang paling sering muncul dalam ulasan. Kata-kata yang lebih besar menunjukkan frekuensi yang lebih tinggi. Anda dapat menyesuaikan jumlah kata maksimum dan skema warna untuk mendapatkan visualisasi yang sesuai.")
                plot_word_cloud(data, max_words, colormap)
            
            elif visualization_option == "Tren Sentimen":
                st.write("### Tren Sentimen dari Waktu ke Waktu")
                st.write("Grafik ini menunjukkan perubahan jumlah ulasan berdasarkan sentimen dari waktu ke waktu. Ini berguna untuk mengidentifikasi pola sentimen musiman atau tren jangka panjang.")
                plot_sentiment_trend(data)
            
            elif visualization_option == "Rata-rata Panjang Ulasan per Sentimen":
                st.write("### Rata-rata Panjang Ulasan per Sentimen")
                st.write("Grafik ini menunjukkan panjang rata-rata ulasan untuk setiap kategori sentimen. Ini dapat memberikan wawasan tentang bagaimana panjang ulasan berkorelasi dengan sentimen.")
                plot_average_review_length_per_sentiment(data)
            
            elif visualization_option == "Heatmap Korelasi":
                st.write("### Heatmap Korelasi Antar Fitur")
                st.write("Heatmap ini menunjukkan korelasi antar fitur numerik dalam dataset. Warna yang lebih gelap menunjukkan korelasi yang lebih kuat antara fitur-fitur tersebut.")
                plot_correlation_heatmap(data)

            st.download_button(
                label="Unduh Visualisasi",
                data=data.to_csv(index=False, encoding='utf-8'),
                file_name='visualisasi_data.csv',
                mime='text/csv'
            )
        else:
            st.write("File CSV tidak memiliki kolom 'Sentiment'.")