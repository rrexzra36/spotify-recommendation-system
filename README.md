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
  <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/dataset.png" />
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

## Data Preparation
Persiapan data adalah tahap penting dalam mengolah data mentah menjadi format yang sesuai untuk analisis atau pemrosesan lebih lanjut. Dalam proyek ini, beberapa teknik dan metode yang diterapkan dalam proses persiapan data adalah sebagai berikut:

1. **Handling Missing Value**
Nilai yang hilang adalah salah satu tantangan umum dalam analisis data di industri. Permasalahan ini muncul ketika terdapat data yang tidak lengkap, yang sering kali direpresentasikan sebagai nilai NaN di dalam pustaka pandas. Penyebabnya bisa beragam, termasuk kesalahan manusia, isu privasi, serta masalah saat melakukan penggabungan data. Tujuan dari langkah ini adalah untuk memastikan bahwa data yang digunakan dalam analisis atau pemodelan memiliki akurasi dan keandalan yang tinggi. Nilai yang hilang dapat menyebabkan bias serta kesalahan dalam analisis, sehingga penting untuk mengidentifikasi dan menangani masalah ini agar hasil analisis menjadi lebih akurat dan dapat dipercaya.

2. **Handling Dupluicated Data**
Data duplikat juga merupakan masalah yang sering ditemui dalam industri. Masalah ini terjadi ketika terdapat observasi yang memiliki nilai yang persis sama di setiap kolomnya. Langkah ini bertujuan untuk menjaga integritas data. Kehadiran data duplikat dapat mempengaruhi hasil analisis dan menghasilkan informasi yang tidak akurat. Oleh karena itu, penting untuk mengidentifikasi dan menghapus data yang terduplikasi agar data yang digunakan dalam analisis atau pemodelan tetap valid dan representatif. Salah satu teknik yang dapat diterapkan untuk mengatasi masalah ini adalah dengan menghapus data yang terduplikasi.

3. **Feature Engineering**
Merupakan proses untuk mengembangkan dan memilih atribut atau fitur yang akan digunakan dalam analisis data atau dalam pembuatan model *machine learning*. Dalam proyek ini, tahap rekayasa fitur dilakukan pada kolom genre. Terdapat beberapa entri dalam kolom genre yang memiliki lebih dari satu genre. Oleh karena itu, perlu dilakukan penanganan dengan memilih genre dari kategori pertama. Hal ini bertujuan untuk memudahkan pengembangan model dan memastikan model yang dihasilkan memiliki performa yang baik.

## Modeling
Pada proyek ini, pendekatan yang dipakai untuk mengembangkan model dalam sistem rekomendasi adalah `Content-Based Filtering`.

### Content Based Filtering
Content Based Filtering adalah metode yang digunakan dalam sistem rekomendasi dan analisis data dengan fokus pada karakteristik atau konten dari item yang ingin direkomendasikan atau dianalisis. Pendekatan ini memanfaatkan atribut atau fitur dari item untuk menentukan kesamaan antara item yang ada dan preferensi pengguna. Dengan kata lain, sistem ini merekomendasikan item berdasarkan kesamaan antara konten item yang sudah diketahui pengguna dan konten item yang akan direkomendasikan.

<p align='center'><img src="https://github.com/SyarifulMsth/Spotify-Music-Recommendation-System-/blob/main/images/content_based_filltering.png?raw=true"  width="500"></p>

<div align="center">

| **Kelebihan**                                 | **Kekurangan**                                 |
|-----------------------------------------------|------------------------------------------------|
| Personalisasi yang Tinggi                     | Keterbatasan dalam Variasi Rekomendasi         |
| Transparansi dalam Rekomendasi                | Ketergantungan pada Kualitas Fitur             |
| Kemampuan untuk Beradaptasi                   | Masalah Overspecialization                     |
| Kemandirian dari Data Pengguna Lain           | Keterbatasan dalam Memahami Konteks Sosial     |

</div>

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.
Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.
**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
**---Ini adalah bagian akhir laporan---**
_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.