# -*- coding: utf-8 -*-
"""Spotify-Recommendation-Systems.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NnCbzqJav-yHADbxKbOF2iHd3tYmjtBP

# Spotify Recommendation Systems

## A. Business Understanding
Spotify merupakan salah satu platform streaming musik terbesar di dunia dengan jutaan pengguna aktif. Agar tetap kompetitif dan meningkatkan pengalaman pengguna, Spotify harus memberikan rekomendasi musik yang relevan dan personal. Sistem rekomendasi yang efektif membantu menjaga pengguna tetap terlibat dengan aplikasi, memperpanjang waktu mendengarkan musik, dan mendorong mereka untuk mengeksplorasi konten baru.

Spotify menghadapi tantangan dalam menyaring jumlah besar data yang berasal dari jutaan lagu, artis, dan preferensi pengguna. Sistem rekomendasi saat ini mungkin kurang optimal dalam menyesuaikan preferensi musik yang selalu berubah. Kegagalan untuk memberikan rekomendasi yang tepat dapat mengakibatkan pengguna tidak puas, meninggalkan platform, atau tidak terlibat sebanyak mungkin. Oleh karena itu, pengembangan sistem rekomendasi yang lebih baik dan lebih akurat sangat dibutuhkan.

## B. Data Understanding
Mengatur dan menghubungkan Kaggle dengan Google Colaboratory. Langkah ini bertujuan untuk mengunduh dataset yang tersedia di Kaggle agar bisa digunakan dalam pengembangan model di Colaboratory.

### Connect to Kaggle
"""

from google.colab import drive
drive.mount('/content/drive')

# kaggle API Token
!mkdir -p ~/.kaggle
!cp '/content/drive/MyDrive/PYTHON/Kaggle API/kaggle.json' ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

# API Command from Anemia Dataset
!kaggle datasets download -d paradisejoy/top-hits-spotify-from-20002019

# Unzipping Datasets
!unzip -o /content/top-hits-spotify-from-20002019.zip -d /content/drive/MyDrive/PYTHON/Dataset

"""### Data Loading"""

# libraries to be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# read_csv
df = pd.read_csv('/content/drive/MyDrive/PYTHON/Dataset/songs_normalize.csv')
df.head()

"""### Data Cleaning"""

df.info()

# check missing values
(df.isnull() | df.empty | df.isna()).sum()

df.duplicated().sum()

# remove the duplicates data
df.drop_duplicates(inplace=True)
if df.duplicated().sum() == 0:
  print('No Duplicates')

# check dimensinality
df.shape

# check the decriptive statistics
df.describe(include='all').transpose()

"""Dataset ini terdiri dari berbagai kolom, seperti artist, song, duration_ms, explicit, year, popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, dan genre. Kolom artist dan song menampilkan nama artis serta judul lagu, di mana artis yang paling sering muncul adalah Drake (23 kali) dan lagu yang paling sering muncul adalah "Sorry" (4 kali). Kolom explicit menunjukkan apakah lagu mengandung konten eksplisit, dengan mayoritas (1404 entri) tidak eksplisit. Genre paling dominan adalah "pop" dengan 416 lagu.

Pada tahap pembersihan data, teridentifikasi adanya **59 entri duplikat** dalam dataset yang digunakan. Duplikasi ini harus dihapus untuk memastikan keakuratan model dalam melakukan prediksi. Meskipun jumlah duplikat cukup signifikan, dataset tetap dapat digunakan karena masih tersedia **1941 entri yang valid**, yang dianggap cukup memadai untuk analisis lebih lanjut.

Setelah itu, fungsi `describe()` digunakan untuk memberikan ringkasan statistik yang mencakup informasi terkait tendensi sentral, distribusi, serta jangkauan data dalam dataset. Pada beberapa kolom, seperti mean, std, dan min, terdapat nilai NaN (Not a Number), terutama untuk kolom artist, song, dan explicit. Hal ini disebabkan karena kolom-kolom tersebut berisi data object, bukan data numerik. Nilai NaN muncul karena fungsi deskriptif seperti mean, standar deviasi, dan lainnya tidak relevan untuk data non-numerik. Oleh karena itu, statistik seperti rata-rata dan deviasi standar tidak dapat dihitung untuk kolom yang berisi teks atau boolean. Sehingga perlu dikonversi ke tipe data kategorikal guna memfasilitasi proses Exploratory Data Analysis (EDA).

## C. Eploratory Data Analysis (EDA)
Analisis data eksploratif (Exploratory Data Analysis/EDA) adalah proses awal yang penting untuk memahami dataset, menganalisis karakteristiknya, menemukan pola, mendeteksi anomali, dan memeriksa asumsi data dengan memanfaatkan metode statistik serta visualisasi grafis. Pada tahap pembersihan data sebelumnya, sebenarnya sudah dilakukan proses EDA dengan menggali informasi terkait dataset, atribut, kelas, jumlah instance, dan statistik deskriptif.

Di sini, eksplorasi lanjutan akan dilakukan untuk menganalisis lebih dalam dan mengetahui wawasan (*insight*) yang dapat diperoleh dari dataset ini. Sebelumnya, akan dilakukan konversi data numerik menjadi kategorikal.

### Univariate Analysis
"""

plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=df, palette='viridis')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Distribution of Songs by Year')
plt.xticks(rotation=45)
plt.show()

"""Visualisasi ini menunjukkan bahwa tidak terdapat hubungan yang kuat antara popularitas lagu dengan fitur audio seperti danceability, energy, dan tempo. Sebagian besar lagu, terlepas dari variasi dalam fitur-fitur tersebut, memiliki tingkat popularitas yang serupa, berkisar antara 40 hingga 80. Oleh karena itu, fitur-fitur audio ini mungkin tidak cukup untuk memprediksi popularitas lagu secara akurat dalam sistem rekomendasi. Pendekatan yang lebih efektif mungkin memerlukan penggunaan faktor lain, seperti preferensi pengguna, lirik, atau metadata lainnya, untuk meningkatkan relevansi rekomendasi."""

# Mengurutkan dan mengambil 10 artis teratas berdasarkan popularitas
top_artists_popularity = df.sort_values(by='popularity', ascending=False).head(10)

# Membuat ukuran figure yang lebih besar
plt.figure(figsize=(12, 8))

# Membuat grafik bar dengan palet warna 'viridis'
plot = sns.barplot(x='popularity', y='artist', palette='viridis', data=top_artists_popularity)

# Menonaktifkan garis sumbu y
plot.spines['left'].set_visible(False)

# Mengatur label sumbu
plt.xlabel('Popularity Score', fontsize=14)
plt.ylabel('Artist', fontsize=14)
plt.title('Top 10 Artists by Popularity', fontsize=16)

# Menambahkan nilai pada setiap bar
for i in plot.patches:
    plot.text(i.get_width() + 1, i.get_y() + i.get_height()/2,
              str(round(i.get_width(), 2)),
              fontsize=12, color='black', ha='left', va='center')

# Menampilkan plot dengan layout yang lebih baik
plt.tight_layout()
plt.show()

"""Grafik ini menampilkan 10 artis terpopuler berdasarkan skor popularitas. The Neighbourhood menempati posisi pertama dengan skor 87.0, diikuti oleh Tom Odell dengan skor 88.0, dan Eminem di posisi ketiga dengan 86.5. Billie Eilish dan WILLOW memiliki skor yang sama, yaitu 86.0, sementara Post Malone, Bruno Mars, dan Ed Sheeran memiliki skor 85.0. Perbedaan skor antara artis-artis ini sangat tipis, menunjukkan bahwa mereka semua berada pada tingkat popularitas yang hampir seimbang.

### Bivariate Analysis
"""

sns.set(style="whitegrid")

# Analisis Bivariate: Korelasi antara popularitas dengan danceability, energy, dan tempo
plt.figure(figsize=(15, 5))

# Scatter plot Popularity vs Danceability
plt.subplot(1, 3, 1)
sns.scatterplot(x='danceability', y='popularity', data=df)
plt.title('Danceability vs Popularity')

# Scatter plot Popularity vs Energy
plt.subplot(1, 3, 2)
sns.scatterplot(x='energy', y='popularity', data=df)
plt.title('Energy vs Popularity')

# Scatter plot Popularity vs Tempo
plt.subplot(1, 3, 3)
sns.scatterplot(x='tempo', y='popularity', data=df)
plt.title('Tempo vs Popularity')

plt.suptitle('Bivariate Analysis: Popularity vs Audio Features', size=16)
plt.tight_layout()
plt.show()

"""Visualisasi ini menunjukkan bahwa tidak terdapat hubungan yang kuat antara popularitas lagu dengan fitur audio seperti danceability, energy, dan tempo. Sebagian besar lagu, terlepas dari variasi dalam fitur-fitur tersebut, memiliki tingkat popularitas yang serupa, berkisar antara 40 hingga 80. Oleh karena itu, fitur-fitur audio ini mungkin tidak cukup untuk memprediksi popularitas lagu secara akurat dalam sistem rekomendasi. Pendekatan yang lebih efektif mungkin memerlukan penggunaan faktor lain, seperti preferensi pengguna, lirik, atau metadata lainnya, untuk meningkatkan relevansi rekomendasi.

### Multivariate Analysis
"""

# Analisis Multivariate: Korelasi matriks
plt.figure(figsize=(12, 8))

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

corr_matrix = numeric_df.corr()  # Calculate correlation on numeric data only
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Multivariate Analysis: Correlation Matrix of Features', size=16)
plt.show()

"""Visualisasi korelasi matriks fitur dalam konteks sistem rekomendasi musik, terlihat bahwa fitur seperti energy dan loudness memiliki korelasi positif kuat, menunjukkan lagu energik cenderung lebih keras. Sebaliknya, ada korelasi negatif antara acousticness dan energy, sehingga lagu akustik cenderung memiliki energi rendah. Selain itu, lagu yang lebih ceria (valence) sering kali lebih mudah untuk menari (danceability). Korelasi ini membantu sistem rekomendasi musik dalam menyesuaikan lagu berdasarkan preferensi pengguna, seperti memilih lagu energik atau ceria yang cocok untuk suasana tertentu.

### Outlier and Distribution Analysis
"""

# Mengatur style visualisasi
sns.set(style="whitegrid")

# Visualisasi boxplot untuk semua kolom numerik
plt.figure(figsize=(15, 10))

# Get numerical columns from the DataFrame
numerical_columns = df.select_dtypes(include=['number']).columns # Define numerical_columns

# Adjust the grid layout to accommodate all numerical columns
num_rows = 4  # Adjust as needed
num_cols = 4  # Adjust as needed

plt.suptitle('Distribution Analysis and Outlier Detection', size=16)

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(num_rows, num_cols, i)  # Mengatur grid 3x4
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot {col}')

plt.tight_layout()
plt.show()

"""## D. Data Preparation
Data preparation sangat penting dalam pengembangan model machine learning, karena proses ini mencakup transformasi data yang diperlukan agar pemodelan dapat berlangsung dengan optimal.

Dalam dataset ini, terdapat sejumlah variabel yang akan dipakai untuk membangun model machine learning, di antaranya adalah artist, song, year, popularity, danceability, energy, speechiness, acousticness, instrumentalness, valence, tempo, dan genre. Dengan demikian, fitur-fitur lain yang tidak relevan untuk proyek ini akan dihapus atau di-drop.
"""

# delete unessesary features from dataset
features_to_drop = ['duration_ms', 'explicit', 'key', 'loudness', 'mode']
df.drop(features_to_drop, axis=1, inplace=True)

df.head()

"""Untuk memahami seberapa bervariasi data dalam setiap fitur, kita bisa memeriksa jumlah entri unik yang dimiliki oleh masing-masing fitur. Hal ini penting dilakukan karena memberikan gambaran mengenai distribusi data dan membantu dalam proses analisis lebih lanjut. Langkah-langkah untuk menghitung jumlah entri unik pada setiap fitur dapat dilakukan dengan cara berikut."""

print('Jumlah entri unik pada setiap fitur:')
for column in df.columns:
    unique_count = df[column].nunique()
    print(f'Kolom {column}: {unique_count}')

# Menampilkan nilai unik dari kolom 'year' dengan mengurutkannya
unique_year = df['year'].unique()
unique_year.sort()
unique_year

"""Musik yang terdapat data dataset antara tahun **1998 - 2020**"""

# Menampilkan nilai unik dari kolom 'popularity' dengan mengurutkannya
unique_popularity = df['popularity'].unique()
unique_popularity.sort()
unique_popularity

"""Popularity tersebar dari yang paling rendah yaitu 0 hingga paling tinggi yaitu 89"""

# Menampilkan nilai unik dari kolom 'genre'
unique_genre = df['genre'].unique()
unique_genre.sort()
unique_genre

"""Jika diperhatikan, terdapat beberapa entri lagu yang memiliki lebih dari satu genre, dan genre tersebut muncul berulang kali. Kondisi ini dapat memengaruhi kinerja model, sehingga perlu adanya penanganan khusus untuk mengatasi masalah tersebut. Sebagai contoh, terdapat entri dengan genre "rock, pop" atau "rock, metal". <br>
Dalam proyek ini, data yang akan digunakan hanya akan mengambil genre pertama pada entri yang memiliki lebih dari satu genre. Pendekatan ini bertujuan untuk menyederhanakan representasi genre dan mencegah kompleksitas berlebih dalam model.
"""

# Memperbaiki kode untuk menangani kasus genre dengan tanda '/'
df['genre'] = df[~(df.genre.isna())]['genre'].apply(lambda x: x.split(',')[0] if x.split(',')[0] != "Hip Hop" else "Hip Hop" if '/' not in x else x)

print(df['genre'].value_counts())

"""Berdasarkan analisis yang telah dilakukan, ditemukan genre dengan nama set(). Karena saat ini tidak memungkinkan untuk mengidentifikasi genre dari musik tersebut, langkah sementara yang akan diambil adalah menghapus data tersebut dari dataset. Hal ini dilakukan untuk menjaga konsistensi dan kualitas data yang digunakan dalam model, hingga solusi yang lebih tepat dapat diterapkan di masa mendatang."""

# Menghapus data dengan genre "set()"
df = df[df['genre'] != 'set()']
df.head()

"""## E. Content Based Filtering Model

### Indentifikasi Representasi Feature
"""

data = df
data.head()

# Insialisasi
tf = TfidfVectorizer()

# Perhitungan idf pada data genre
tf.fit(data['genre'])

# Mapping array feature index int ke feature utama
tf.get_feature_names_out()

# transform ke bentuk matrix
tfidf_matrix = tf.fit_transform(data['genre'])

# output
tfidf_matrix.shape

# Mengubah vektor tf-idf dalam bentuk matrix
tfidf_matrix.todense()

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan jenis masakan
# Baris diisi dengan nama resto

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.song
).sample(10, axis=1).sample(10, axis=0)

"""### Cosine Similarity"""

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""Pada tahap ini, dilakukan perhitungan cosine similarity menggunakan dataframe `tfidf_matrix` yang sudah dihasilkan pada tahap sebelumnya. Dengan hanya satu baris kode untuk memanggil fungsi cosine similarity dari library `sklearn`, perhitungan kesamaan antar lagu berhasil dilakukan. Hasil dari perhitungan ini adalah sebuah matriks kesamaan yang disajikan dalam bentuk array.

Langkah selanjutnya adalah melihat hasil matriks kesamaan antar lagu. Untuk itu, ditampilkan sampel data dari matriks tersebut dengan mengambil 5 sampel kolom (axis = 1) yang merepresentasikan judul lagu, serta 10 sampel baris (axis = 0) untuk melihat perbandingan kesamaan antar beberapa lagu.







"""

# merancang dataframe dari var cosine_sim dengna baris dan kolom judul lagu
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['song'], columns=data['song'])
print('Shape:', cosine_sim_df.shape)

# menampilkan similarity matrix pada tiap lagu
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""Berdasarkan cosine similarity di atas, dapat diidentifikasi tingkat kesamaan antar lagu. Matriks dengan ukuran (1919, 1919) merepresentasikan ukuran similarity matrix dari data yang digunakan. Artinya, matriks ini menunjukkan kesamaan antara 1919 lagu, baik pada sumbu X (horizontal) maupun Y (vertikal). Dengan kata lain, telah dilakukan penghitungan kesamaan antar 1919 lagu.

Namun, karena tidak mungkin menampilkan semua hasil dalam satu waktu, hanya 10 lagu di sumbu vertikal dan 5 lagu di sumbu horizontal yang ditampilkan untuk representasi sederhana.

Jika diperhatikan, kesamaan antara lagu di sumbu X dan Y bisa diidentifikasi. Sebagai contoh, lagu **"If You Come Back"** pada sumbu X terdeteksi memiliki kesamaan dengan lagu **"Milkshake"** pada sumbu Y yang menunjukkan bahwa kedua lagu tersebut memiliki karakteristik yang mirip menurut penghitungan cosine similarity.

### Top-N Recommendation
"""

def recommend_song(song_title, similarity_data=cosine_sim_df, items=data[['song', 'genre']], k=5):
    """
    Rekomendasi lagu berdasarkan kemiripan dataframe

    Parameter:
    ---
    song_title : tipe data string (str)
                Nama lagu (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan lagu sebagai
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---


    Pada index ini, kita mengambil k dengan nilai similarity terbesar
    pada index matrix yang diberikan (i).
    """
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,song_title].to_numpy().argpartition(
        range(-1, -k, -1))
    # Mengambil similarity terbesar dai index
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    # Drop nama lagu yang dicari
    closest = closest.drop(song_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""Pada kode di atas, digunakan fungsi `argpartition` untuk mengambil sejumlah nilai tertinggi (k) dari data similarity, yaitu dalam hal ini dari dataframe `cosine_sim_df`. Argpartition membantu mengidentifikasi k nilai kesamaan tertinggi dari data yang tersedia. Setelah itu, data dengan bobot atau tingkat kesamaan tertinggi hingga terendah diurutkan dan disimpan ke dalam variabel `closest`.

Langkah berikutnya adalah menghapus lagu yang sedang dicari, agar lagu tersebut tidak muncul dalam daftar rekomendasi. Dalam contoh ini, kita mencari lagu yang mirip dengan lagu "Excuse Me Miss" oleh Taylor Swift. Dengan demikian, setelah proses pengurutan, rekomendasi lagu yang mirip dengan "Excuse Me Miss" akan muncul, kecuali "Excuse Me Miss" itu sendiri.
"""

# Menampilkan daftar lagu dengan artis yang diinginkan
based_songs = df[df['artist'] == 'JAY-Z'].head()
based_songs

data[data.song.eq('Excuse Me Miss')]

# Mendapatkan rekomendasi lagu yang mirip dengan lagu Jay-Z - Excuse me miss
recommend_song('Excuse Me Miss')