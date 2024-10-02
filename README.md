# Laporan Proyek Machine Learning - Reyhan Ezra Bimantara
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg" />
</p>

## Project Overview
Spotify merupakan salah satu platform streaming musik terbesar di dunia dengan jutaan pengguna aktif. Agar tetap kompetitif dan meningkatkan pengalaman pengguna, Spotify harus memberikan rekomendasi musik yang relevan dan personal. Sistem rekomendasi yang efektif membantu menjaga pengguna tetap terlibat dengan aplikasi, memperpanjang waktu mendengarkan musik, dan mendorong mereka untuk mengeksplorasi konten baru.

Spotify menghadapi tantangan dalam menyaring jumlah besar data yang berasal dari jutaan lagu, artis, dan preferensi pengguna. Sistem rekomendasi saat ini mungkin kurang optimal dalam menyesuaikan preferensi musik yang selalu berubah. Kegagalan untuk memberikan rekomendasi yang tepat dapat mengakibatkan pengguna tidak puas, meninggalkan platform, atau tidak terlibat sebanyak mungkin. Oleh karena itu, pengembangan sistem rekomendasi yang lebih baik dan lebih akurat sangat dibutuhkan.

**Referensi:**
- [The Netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS)](https://dl.acm.org/doi/10.1145/2843948) 
- [Deep Learning Based Recommender System: A Survey and New Perspectives](https://dl.acm.org/doi/10.1145/3285029) 

## Business Understanding
### Problem Statements
Berdasarkan penjelasan dalam latar belakang, dapat dirumuskan beberapa permasalahan utama sebagai berikut:
- Bagaimana cara memberikan rekomendasi musik atau lagu berdasarkan genre menggunakan algoritma machine learning?
- Bagaimana mengembangkan model machine learning yang dapat merekomendasikan musik kepada pengguna berdasarkan genre yang ingin didengar?
- Bagaimana mengevaluasi performa model machine learning yang telah dikembangkan untuk memberikan rekomendasi musik kepada pengguna?
### Goals
Dari rumusan masalah di atas, penelitian ini bertujuan untuk:
- Memahami penerapan algoritma machine learning dalam memberikan rekomendasi musik berdasarkan genre.
- Memahami seluruh proses pengembangan model machine learning dari awal hingga akhir dalam memberikan rekomendasi musik berdasarkan genre.
- Mengetahui performa atau evaluasi dari model machine learning yang telah dikembangkan untuk memberikan rekomendasi musik berdasarkan genre.
### Solution statements
Untuk mencapai tujuan yang ditetapkan, penelitian ini mengadopsi beberapa langkah berikut:
- Melaksanakan proses pengembangan model machine learning dari awal hingga akhir, yang mencakup data wrangling, eksplorasi data (EDA), pemodelan, dan evaluasi.
- Mengembangkan model machine learning dengan menggunakan algoritma yang sesuai untuk masalah ini, yaitu algoritma content-based filtering dengan dataset yang telah ditentukan. Adaapun tahapan yang dilakukan seperti berikut:
  - **Identifikasi Representasi Fitur (TfidfVectorizer)**
  Dalam sistem rekomendasi musik, langkah pertama adalah mengubah data tekstual menjadi representasi numerik agar model machine learning dapat memprosesnya. Salah satu metode yang umum digunakan adalah TfidfVectorizer. Tfidf (Term Frequency-Inverse Document Frequency) adalah teknik yang menghitung seberapa penting suatu kata dalam sebuah dokumen relatif terhadap seluruh kumpulan dokumen. Dalam konteks rekomendasi musik, setiap lagu dapat direpresentasikan sebagai vektor berdasarkan lirik, genre, atau deskripsi lainnya. TfidfVectorizer akan menghitung bobot dari setiap fitur ini, sehingga fitur yang lebih relevan dan penting dalam mendeskripsikan lagu akan memiliki nilai yang lebih tinggi. Representasi fitur ini memungkinkan model untuk memahami hubungan antar lagu berdasarkan konten mereka.
  - **Cosine Similarity**
  Setelah mendapatkan representasi fitur dari setiap lagu, langkah berikutnya adalah mengukur kesamaan antara lagu-lagu tersebut. Cosine similarity adalah salah satu metode yang digunakan untuk menghitung seberapa mirip dua vektor satu sama lain. Dalam konteks rekomendasi musik, cosine similarity akan digunakan untuk menentukan kesamaan antara vektor representasi fitur dari lagu yang diinginkan pengguna dan lagu-lagu lain dalam dataset. Nilai cosine similarity berkisar antara -1 hingga 1, di mana 1 menunjukkan kesamaan yang sangat tinggi, 0 menunjukkan ketidakberhubungan, dan -1 menunjukkan ketidakcocokan. Dengan menghitung cosine similarity, sistem dapat mengidentifikasi lagu-lagu yang memiliki karakteristik serupa dengan lagu yang dipilih oleh pengguna.
  - **Top-N Recommendations**
  Setelah menghitung cosine similarity untuk setiap lagu dalam dataset, langkah terakhir adalah memberikan rekomendasi kepada pengguna. Top-N recommendations merujuk pada proses pemilihan sejumlah N lagu teratas yang paling mirip dengan lagu yang diinginkan pengguna berdasarkan hasil perhitungan cosine similarity. Dalam implementasi ini, sistem akan mengurutkan lagu-lagu berdasarkan nilai kesamaan tertinggi dan memilih N lagu teratas untuk ditampilkan kepada pengguna sebagai rekomendasi. Pendekatan ini memberikan pengguna pilihan lagu yang relevan dan sesuai dengan preferensi mereka, sehingga meningkatkan pengalaman mendengarkan musik.
- Melakukan evaluasi model untuk mengukur performa model dengan menggunakan metrik evaluasi yang telah ditetapkan. Evaluasi ini bertujuan untuk menentukan seberapa baik model yang telah dikembangkan dalam memberikan rekomendasi musik berdasarkan genre, sehingga dapat diidentifikasi model machine learning yang paling efektif untuk digunakan dalam memberikan rekomendasi musik.
## Data Understanding
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/dataset.png" />
</p>

Dataset yang digunakan dalam proyek machine learning ini merupakan dataset anemia yang terdiri dari 2000 entri data atau record. Dataset ini bersifat open-source, yang berarti tersedia secara bebas untuk digunakan oleh publik, dan telah dipublikasikan oleh Mark Korveha melalui platform Kaggle dengan judul [Top Hits Spotify from 2000-2019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019). Dataset ini berisi statistik audio dari 2000 lagu teratas di Spotify yang dirilis antara tahun 2000 hingga 2019. Dalam dataset ini, terdapat sekitar 18 kolom yang masing-masing memberikan informasi detail mengenai lagu-lagu tersebut serta berbagai karakteristiknya. Setiap kolom dirancang untuk menjelaskan aspek tertentu dari lagu, mulai dari durasi, genre, popularitas, hingga fitur-fitur teknis lainnya yang mencakup elemen musik seperti tempo, kunci, dan tingkat energi.

Statistik audio yang terdapat dalam dataset ini sangat berguna bagi para peneliti, pengembang, dan analis musik, karena memungkinkan mereka untuk mengeksplorasi dan memahami tren dalam industri musik selama dua dekade terakhir. Dengan data ini, pengguna dapat melakukan analisis mendalam mengenai bagaimana kualitas audio, genre, dan elemen lainnya berkontribusi pada popularitas suatu lagu. Selain itu, dataset ini juga dapat digunakan sebagai basis untuk membangun model machine learning yang dapat memberikan rekomendasi musik, sehingga meningkatkan pengalaman pengguna dalam menemukan lagu-lagu baru yang sesuai dengan preferensi mereka.

**Variabel-variabel pada Spotify dataset adalah sebagai berikut:**
- **artist**: Nama Artis.
- **song**: Nama Lagu.
- **duration_ms**: Durasi lagu dalam milidetik.
- **explicit**: Lirik atau konten dari sebuah lagu atau video musik mengandung satu atau lebih kriteria yang dapat dianggap ofensif atau tidak sesuai untuk anak-anak.
- **year**: Tahun rilis lagu.
- **popularity**: Semakin tinggi nilai, semakin populer lagu tersebut.
- **danceability**: Danceability menggambarkan seberapa cocok sebuah lagu untuk menari berdasarkan kombinasi elemen musik, termasuk tempo, stabilitas ritme, kekuatan ketukan, dan regularitas keseluruhan. Nilai 0,0 adalah yang paling tidak cocok untuk menari dan 1,0 adalah yang paling cocok.
- **energy**: Energi adalah ukuran dari 0,0 hingga 1,0 yang mewakili ukuran persepsi tentang intensitas dan aktivitas.
- **key**: Kunci lagu. Bilangan bulat dipetakan ke nada menggunakan notasi Kelas Nada standar. Contoh: 0 = C, 1 = C♯/D♭, 2 = D, dan seterusnya. Jika tidak ada kunci yang terdeteksi, nilainya adalah -1.
- **loudness**: Tingkat kebisingan keseluruhan sebuah lagu dalam desibel (dB). Nilai kebisingan dirata-ratakan di seluruh lagu dan berguna untuk membandingkan kebisingan relatif antar lagu. Kebisingan adalah kualitas suara yang merupakan korespondensi psikologis utama dari kekuatan fisik (amplitudo). Nilai biasanya berkisar antara -60 hingga 0 dB.
- **mode**: Mode menunjukkan modalitas (mayor atau minor) dari sebuah lagu, jenis skala dari mana konten melodik diambil. Mayor diwakili oleh 1 dan minor oleh 0.
- **speechiness**: Speechiness mendeteksi keberadaan kata-kata yang diucapkan dalam sebuah lagu. Semakin eksklusif rekaman mirip ucapan (misalnya, talk show, buku audio, puisi), semakin mendekati 1,0 nilai atribut tersebut. Nilai di atas 0,66 menggambarkan lagu-lagu yang mungkin terdiri sepenuhnya dari kata-kata yang diucapkan. Nilai antara 0,33 dan 0,66 menggambarkan lagu-lagu yang mungkin mengandung musik dan ucapan, baik dalam bagian-bagian atau tumpang tindih, termasuk kasus seperti musik rap. Nilai di bawah 0,33 kemungkinan besar mewakili musik dan lagu-lagu lain yang tidak mirip ucapan.
- **acousticness**: Ukuran kepercayaan dari 0,0 hingga 1,0 apakah lagu tersebut akustik. 1,0 mewakili kepercayaan tinggi bahwa lagu tersebut akustik.
- **instrumentalness**: Memprediksi apakah sebuah lagu tidak mengandung vokal. Suara "ooh" dan "aah" dianggap sebagai instrumen dalam konteks ini. Lagu rap atau kata yang diucapkan jelas dianggap "vokal". Semakin mendekati 1,0 nilai instrumentalness, semakin besar kemungkinan lagu tersebut tidak mengandung konten vokal. Nilai di atas 0,5 dimaksudkan untuk mewakili lagu-lagu instrumental, tetapi kepercayaan lebih tinggi saat nilai mendekati 1,0.
- **liveness**: Mendeteksi keberadaan audiens dalam rekaman. Nilai liveness yang lebih tinggi menunjukkan kemungkinan meningkat bahwa lagu tersebut dipertunjukkan secara langsung. Nilai di atas 0,8 memberikan kemungkinan kuat bahwa lagu tersebut adalah pertunjukan langsung.
- **valence**: Ukuran dari 0,0 hingga 1,0 yang menggambarkan positiveness musik yang disampaikan oleh sebuah lagu. Lagu-lagu dengan valence tinggi terdengar lebih positif (misalnya, bahagia, ceria, euforia), sementara lagu-lagu dengan valence rendah terdengar lebih negatif (misalnya, sedih, depresi, marah).
- **tempo**: Tempo keseluruhan yang diperkirakan dari sebuah lagu dalam ketukan per menit (BPM). Dalam terminologi musik, tempo adalah kecepatan atau ritme dari sebuah karya dan berasal langsung dari rata-rata durasi ketukan.
- **genre**: Genre dari lagu tersebut.

### Hasil Visualiasi dan Analisis Data
1. **Pengecekan Duplikasi Data**
Pada tahap pembersihan data, teridentifikasi adanya **59 entri duplikat** dalam dataset yang digunakan. Duplikasi ini harus dihapus untuk memastikan keakuratan model dalam melakukan prediksi. Meskipun jumlah duplikat cukup signifikan, dataset tetap dapat digunakan karena masih **tersedia 1941 entri yang valid**, yang dianggap cukup memadai untuk analisis lebih lanjut.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/duplicated.png" />
</p>


1. **Univariate Analysis**
Visualisasi data menunjukkan bahwa jumlah lagu per tahun meningkat tajam dari tahun 1999 hingga mencapai puncaknya pada tahun 2001. Setelah itu, jumlah lagu yang dirilis per tahun relatif stabil antara 80 hingga 100 lagu dari tahun 2002 hingga 2017. Setelah 2017, terlihat sedikit penurunan, dengan penurunan drastis pada tahun 2020, yang mencatat jumlah lagu paling sedikit sepanjang periode yang ditampilkan.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/uni_analysis.png" />
</p>

2. **Bivariate Analysis**
Visualisasi ini menunjukkan bahwa tidak terdapat hubungan yang kuat antara popularitas lagu dengan fitur audio seperti danceability, energy, dan tempo. Sebagian besar lagu, terlepas dari variasi dalam fitur-fitur tersebut, memiliki tingkat popularitas yang serupa, berkisar antara 40 hingga 80. Oleh karena itu, fitur-fitur audio ini mungkin tidak cukup untuk memprediksi popularitas lagu secara akurat dalam sistem rekomendasi. Pendekatan yang lebih efektif mungkin memerlukan penggunaan faktor lain, seperti preferensi pengguna, lirik, atau metadata lainnya, untuk meningkatkan relevansi rekomendasi.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/bivar_analysis.png" />
</p>

3. **Multivariate Analysis**
Visualisasi korelasi matriks fitur dalam konteks sistem rekomendasi musik, terlihat bahwa fitur seperti energy dan loudness memiliki korelasi positif kuat, menunjukkan lagu energik cenderung lebih keras. Sebaliknya, ada korelasi negatif antara acousticness dan energy, sehingga lagu akustik cenderung memiliki energi rendah. Selain itu, lagu yang lebih ceria (valence) sering kali lebih mudah untuk menari (danceability). Korelasi ini membantu sistem rekomendasi musik dalam menyesuaikan lagu berdasarkan preferensi pengguna, seperti memilih lagu energik atau ceria yang cocok untuk suasana tertentu.

<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/multi_analysis.png" />
</p>

4. **Outlier & Distribution Analysis**
Dari hasil analisis distribusi dan deteksi outlier, dapat disimpulkan bahwa sebagian besar fitur dalam dataset memiliki distribusi yang seimbang, dengan rentang nilai yang tidak terlalu ekstrem. Namun, terdapat beberapa fitur yang menunjukkan keberadaan outlier yang signifikan, seperti durasi, popularitas, speechiness, dan instrumentalness. Hal ini mengindikasikan adanya variasi yang cukup tinggi di beberapa lagu, terutama terkait elemen popularitas, vokal, dan instrumental. Secara umum, distribusi fitur seperti danceability, energy, tempo, dan valence menunjukkan variasi yang lebih terkendali. Adanya outlier ini penting untuk dipertimbangkan, terutama jika analisis lanjutan seperti modeling atau clustering dilakukan, karena dapat mempengaruhi hasil dan interpretasi lebih lanjut.

<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/distribution-and-outlier.png" />
</p>

## Data Preparation
Persiapan data adalah tahap penting dalam mengolah data mentah menjadi format yang sesuai untuk analisis atau pemrosesan lebih lanjut. Dalam proyek ini, beberapa teknik dan metode yang diterapkan dalam proses persiapan data adalah sebagai berikut:

1. **Handling Missing Value**
Nilai yang hilang adalah salah satu tantangan umum dalam analisis data di industri. Permasalahan ini muncul ketika terdapat data yang tidak lengkap, yang sering kali direpresentasikan sebagai nilai NaN di dalam pustaka pandas. Penyebabnya bisa beragam, termasuk kesalahan manusia, isu privasi, serta masalah saat melakukan penggabungan data. Tujuan dari langkah ini adalah untuk memastikan bahwa data yang digunakan dalam analisis atau pemodelan memiliki akurasi dan keandalan yang tinggi. Nilai yang hilang dapat menyebabkan bias serta kesalahan dalam analisis, sehingga penting untuk mengidentifikasi dan menangani masalah ini agar hasil analisis menjadi lebih akurat dan dapat dipercaya.

2. **Handling Dupluicated Data**
Data duplikat juga merupakan masalah yang sering ditemui dalam industri. Masalah ini terjadi ketika terdapat observasi yang memiliki nilai yang persis sama di setiap kolomnya. Langkah ini bertujuan untuk menjaga integritas data. Kehadiran data duplikat dapat mempengaruhi hasil analisis dan menghasilkan informasi yang tidak akurat. Oleh karena itu, penting untuk mengidentifikasi dan menghapus data yang terduplikasi agar data yang digunakan dalam analisis atau pemodelan tetap valid dan representatif. Salah satu teknik yang dapat diterapkan untuk mengatasi masalah ini adalah dengan menghapus data yang terduplikasi.

3. **Feature Engineering**
Merupakan proses untuk mengembangkan dan memilih atribut atau fitur yang akan digunakan dalam analisis data atau dalam pembuatan model *machine learning*. Dalam proyek ini, tahap rekayasa fitur dilakukan pada kolom genre. Terdapat beberapa entri dalam kolom genre yang memiliki lebih dari satu genre. Oleh karena itu, perlu dilakukan penanganan dengan memilih genre dari kategori pertama. Hal ini bertujuan untuk memudahkan pengembangan model dan memastikan model yang dihasilkan memiliki performa yang baik.

4. **Vektorisasi dengan TF-IDF**
Pada tahap ini, data yang telah dibersihkan dan siap untuk digunakan akan dikonversi menjadi format vektor dengan memanfaatkan fungsi `TfidfVectorizer()` dari library scikit-learn. Proses ini berhasil mengidentifikasi representasi fitur melalui fungsi `TfidfVectorizer()`.

## Modeling
Pada proyek ini, pendekatan yang dipakai untuk mengembangkan model dalam sistem rekomendasi adalah `Content-Based Filtering`.

### Content Based Filtering
Content Based Filtering adalah metode yang digunakan dalam sistem rekomendasi dan analisis data dengan fokus pada karakteristik atau konten dari item yang ingin direkomendasikan atau dianalisis. Pendekatan ini memanfaatkan atribut atau fitur dari item untuk menentukan kesamaan antara item yang ada dan preferensi pengguna. Dengan kata lain, sistem ini merekomendasikan item berdasarkan kesamaan antara konten item yang sudah diketahui pengguna dan konten item yang akan direkomendasikan.

<p align='center'>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/content-based-filtering.png"  width="500">
</p>

<div align="center">
    <strong>Tabel 1.</strong> Kekurangan dan Kelibihan Content Base Filtering
</div>

<div align="center">

| **Kelebihan**                                 | **Kekurangan**                                 |
|-----------------------------------------------|------------------------------------------------|
| Personalisasi yang Tinggi                     | Keterbatasan dalam Variasi Rekomendasi         |
| Transparansi dalam Rekomendasi                | Ketergantungan pada Kualitas Fitur             |
| Kemampuan untuk Beradaptasi                   | Masalah Overspecialization                     |
| Kemandirian dari Data Pengguna Lain           | Keterbatasan dalam Memahami Konteks Sosial     |

</div>

### Implementasi
Tahapan pemodelan menggunakan algoritma `Content Based Filtering` dalam proyek ini terdiri dari beberapa langkah, yaitu:

1. **Data Prepatation**
Pada tahap sebelumnya, data telah melalui proses pembersihan, dan langkah berikutnya adalah menyiapkan dataframe yang sudah bebas dari kesalahan atau anomali untuk digunakan dalam tahap persiapan data (data preparation). Proses ini mencakup penghapusan data yang tidak relevan, penanganan nilai yang hilang, serta normalisasi atau standarisasi agar data siap digunakan untuk analisis lebih lanjut atau pelatihan model machine learning.

2. **TF-IDF Vectorizer**
Proses term frequency-inverse document frequency (`TF-IDF`) digunakan untuk mengidentifikasi representasi kata-kata penting dalam kolom genre. Dalam proyek ini, proses vektorisasi dilakukan menggunakan fungsi `TfidfVectorizer()` yang tersedia di library scikit-learn. Berikut adalah hasil `TF-IDF` dalam bentuk matriks, di mana matriks tersebut memperlihatkan hubungan antara musik dan genre yang terkait.
<p align='center'>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/tf-idf.png"  width="800">
</p>

3. **Perhitungan Derajat Kesamaan (`Cosine Similarity`)**
Di tahap ini, dilakukan penghitungan derajat kesamaan menggunakan fungsi `cosine_similarity` pada dataframe `tfidf_matrix` yang telah diperoleh sebelumnya. Proses ini bertujuan untuk menghitung kesamaan (*similarity*) antara musik atau lagu berdasarkan genre.
<p align='center'>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/cosine-similarity.png"  width="800">
</p>

4. **Create Custom Functions**
Langkah terakhir adalah membangun fungsi kustom untuk menghasilkan rekomendasi berdasarkan data input yang diinginkan. Fungsi ini bekerja dengan mengambil nilai similarity dari musik yang ingin dicari, kemudian memasukkan musik yang paling mirip ke dalam variabel closest. Parameter `N` ditentukan untuk menghasilkan `top-N recommendation` berdasarkan tingkat kesamaan tertinggi. Musik yang dicari akan dihapus dari daftar agar tidak muncul dalam rekomendasi. Pada langkah akhir, return digunakan untuk mengembalikan hasil rekomendasi dalam bentuk dataframe, di mana nilai yang dikembalikan adalah judul-judul musik berdasarkan tingkat similarity.
<p align='center'>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/custom-functions.png"  width="800">
</p>

5. **Pembuatan Rekomendasi Musik atau Lagu**
Pada tahap ini, fungsi `recommend_song` dibuat dengan menggunakan argpartition. Fungsi ini mengambil sejumlah nilai k tertinggi dari data kesamaan (dalam proyek ini: `cosine_sim_df`). Selanjutnya, data diurutkan dari tingkat kesamaan tertinggi hingga terendah dan dimasukkan ke dalam variabel closest. Untuk memastikan bahwa lagu yang dicari tidak muncul dalam daftar rekomendasi, lagu tersebut dihapus dari hasil akhir.
<p align='center'>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/recommendations.png"  width="800">
</p>

### Output
Proses pengembangan model machine learning untuk rekomendasi musik atau lagu telah berhasil diselesaikan. Langkah berikutnya adalah memeriksa hasil rekomendasi yang dihasilkan oleh model tersebut.

Pada kasus ini, akan dilakukan pencarian lagu-lagu yang serupa dengan lagu **"Excuse Me Miss"** milik **JAY-Z**, yang memiliki genre Hip-Hop.

<div align="center">
    <strong>Tabel 2.</strong> Informasi musik atau lagu untuk uji coba
</div>
<div align="center">

| artist       | song       | year | popularity | genre |
|--------------|------------|------|------------|-------|
| JAY-Z        | Excuse Me Miss | 2002 | 56         | hip-hop   |

</div>

<div align="center">
    <strong>Tabel 3.</strong> Hasil Rekomendasi Musik Berdasarkan Genre 
</div>

<div align="center">

|  index  | song                  | genre     |
|---------|-----------------------|-----------|
| 0       | Circles               | hip-hop   |
| 1       | Titanium (feat. Sia)  | hip-hop   |
| 2       | Slow Jamz             | hip-hop   |
| 3       | Lost                  | hip-hop   |
| 4       | My Place              | hip-hop   |

</div>




## Evaluation
### Perhitungan Evaluasi
Proyek machine learning ini memanfaatkan algoritma `Content Based Filtering`, dan untuk evaluasi performa model, metrik yang digunakan adalah Precision. Precision mengukur seberapa relevan rekomendasi yang diberikan oleh model dan dapat dinyatakan dengan rumus sebagai berikut:

<p align='center'>
<img src="https://latex.codecogs.com/svg.image?{\color{White}Precision=\frac{r}{i}}">
</p>

Di mana:
- <img src="https://latex.codecogs.com/svg.image?{\color{white}(r)}"> adalah jumlah rekomendasi yang relevan.
- <img src="https://latex.codecogs.com/svg.image?{\color{white}(i)}"> adalah total rekomendasi yang diberikan.

Berdasarkan pengujian yang dilakukan di bagian Hasil, diperoleh 5 rekomendasi lagu berdasarkan genre. Jika dilakukan perhitungan dengan rumus di atas, maka nilai Precision yang dihasilkan adalah:

<p align='center'>
  <img style src="https://latex.codecogs.com/svg.image?{\color{White}Precision=\frac{5}{5}=100%}">
</p>

### Hasil Evluasi:
1. Algoritma `content-based filtering` dengan `TfidfVectorizer` dan `cosine similarity` merupakan pendekatan paling efektif dalam proyek ini, dengan kemampuan untuk memberikan rekomendasi musik berdasarkan genre yang diinginkan oleh pengguna.
2. Penelitian ini berhasil menjawab problem statement dengan mengembangkan model machine learning yang dapat merekomendasikan musik berdasarkan genre, menggunakan representasi fitur musik dengan TfidfVectorizer dan mengukur kesamaan antar lagu dengan `cosine similarity`. Proses pengembangan model mencakup *data loading*, eksplorasi data (EDA), pemodelan, dan evaluasi kinerja model yang tepat sesuai dengan tujuan rekomendasi musik berbasis genre.

3. Penelitian ini juga berhasil mencapai seluruh tujuan yang diharapkan, yaitu:
    - Memahami penerapan algoritma machine learning dalam memberikan rekomendasi musik.
    - Mengembangkan model rekomendasi musik yang sesuai dengan preferensi pengguna berdasarkan genre.
    - Mengevaluasi kinerja model menggunakan metrik yang relevan.

4. Solusi yang diterapkan, termasuk penggunaan representasi fitur dengan `TfidfVectorizer`, penghitungan kesamaan menggunakan `cosine similarity`, dan penerapan `Top-N recommendations`, memberikan dampak signifikan terhadap performa model. Evaluasi model menunjukkan bahwa sistem rekomendasi mampu memberikan rekomendasi yang relevan dan sesuai dengan preferensi pengguna, mendukung pengalaman mendengarkan musik yang lebih personal dan tepat sasaran.

## Conclusion
Pengembangan model machine learning untuk rekomendasi musik atau lagu menggunakan algoritma `Content Based Filtering` melalui beberapa tahapan yang bersifat iteratif. Tahapan tersebut dimulai dari pemahaman bisnis (*business understanding*), pemahaman data (*data understanding*), hingga proses pemodelan (*modelling*) dan evaluasi (*evaluation*). Dalam proyek ini, metrik evaluasi yang digunakan adalah precision, yang dipilih berdasarkan konteks data, rumusan masalah, dan solusi yang diterapkan. Precision digunakan untuk mengukur seberapa akurat prediksi atau rekomendasi yang diberikan oleh model.

Setelah melalui tahapan evaluasi, model rekomendasi musik berdasarkan genre menunjukkan hasil yang memuaskan, dengan nilai precision mencapai 100%. Model ini, yang dikembangkan dengan algoritma `Content Based Filtering`, dinilai berhasil menjawab pertanyaan-pertanyaan masalah yang telah dirumuskan serta mencapai tujuan proyek. Dengan adanya model ini, diharapkan sistem rekomendasi musik berdasarkan genre dapat membantu layanan aplikasi musik online memberikan rekomendasi yang lebih relevan kepada penggunanya.