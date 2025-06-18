# Klasifikasi Kematangan Pisang Menggunakan Segmentasi HSV dan K-NN

<p align="center">
  <img src="https://via.placeholder.com/600x300?text=Contoh+Tampilan+Aplikasi" alt="Contoh Tampilan Aplikasi">
  <br>
  <i>(Ganti placeholder ini dengan screenshot aplikasi Anda)</i>
</p>

Proyek ini adalah aplikasi desktop sederhana yang dikembangkan menggunakan PyQt5 dan OpenCV untuk mengklasifikasikan tingkat kematangan pisang. Klasifikasi dilakukan berdasarkan analisis warna pisang menggunakan ruang warna HSV (Hue, Saturation, Value) dan model Machine Learning K-Nearest Neighbors (K-NN).

## Fitur Utama

* **Input Gambar:** Memungkinkan pengguna untuk mengunggah gambar pisang dari perangkat mereka.
* **Preprocessing Citra:** Melakukan serangkaian tahapan preprocessing pada gambar pisang, termasuk:
    * Resizing gambar ke ukuran standar (200x200 piksel).
    * Konversi ruang warna dari BGR ke HSV.
    * Segmentasi objek pisang menggunakan masking HSV.
    * Penyempurnaan masker melalui operasi morfologi (dilasi dan erosi).
* **Ekstraksi Fitur:** Mengekstrak fitur warna pisang berupa nilai rata-rata Hue, Saturation, dan Value dari area pisang yang tersegmentasi.
* **Klasifikasi K-NN:** Menggunakan model K-Nearest Neighbors (K-NN) yang telah dilatih untuk memprediksi tingkat kematangan pisang (misalnya, "Belum Matang", "Matang", "Sangat Matang").
* **Visualisasi Proses:** Menampilkan citra asli, citra hasil masking, serta memungkinkan pengguna melihat detail setiap tahapan preprocessing dalam jendela terpisah.
* **Analisis Tambahan:** Memberikan informasi mengenai nilai fitur HSV yang diekstraksi dan persentase kemiripan fitur citra uji dengan rata-rata HSV dari setiap kategori kematangan.

## Cara Menggunakan

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/yourusername/nama-repo-anda.git]
    cd nama-repo-anda
    ```
2.  **Instal Dependensi:**
    Pastikan Anda memiliki Python terinstal. Instal library yang dibutuhkan:
    ```bash
    pip install opencv-python numpy scikit-learn PyQt5
    ```
3.  **Siapkan Data Latih:**
    Buat folder bernama `train` di direktori utama proyek. Di dalam folder `train`, buat subfolder berikut:
    * `belum_matang`
    * `matang`
    * `sangat_matang`
    Letakkan gambar-gambar pisang yang sesuai di setiap subfolder. Pastikan gambar-gambar tersebut berformat `.jpg`.

    Contoh struktur folder:
    ```
    .
    ├── program.py
    └── train/
        ├── belum_matang/
        │   ├── pisang_bm_01.jpg
        │   └── ...
        ├── matang/
        │   ├── pisang_m_01.jpg
        │   └── ...
        └── sangat_matang/
            ├── pisang_sm_01.jpg
            └── ...
    ```
4.  **Jalankan Aplikasi:**
    ```bash
    python program.py
    ```
5.  Setelah aplikasi terbuka, klik tombol "Pilih Gambar Pisang" untuk mengunggah gambar dan melihat hasil klasifikasi.

## Konsep Teknis

* **Pemrosesan Citra Digital:** Pemanfaatan OpenCV untuk operasi dasar seperti resizing, konversi ruang warna, segmentasi, dan morfologi citra.
* **Ruang Warna HSV:** Digunakan karena lebih efektif dalam memisahkan informasi warna dari intensitas cahaya, membuatnya robust terhadap variasi pencahayaan.
* **K-Nearest Neighbors (K-NN):** Algoritma klasifikasi berbasis instans yang memprediksi kelas objek baru berdasarkan mayoritas kelas dari tetangga-tetangga terdekatnya dalam ruang fitur.
* **Normalisasi Fitur (StandardScaler):** Penting untuk K-NN agar semua fitur memiliki bobot yang setara dalam perhitungan jarak.
* **PyQt5:** Digunakan untuk membangun antarmuka pengguna grafis (GUI) aplikasi desktop.
