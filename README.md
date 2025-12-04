# Capstone-Project-3-Priadi-Jatmiko-JCDSOLSK-003
Membuat Prediksi Harga Apartemen di Daegu Korea Selatan


Dokumen ini menjelaskan alur lengkap proyek Machine Learning untuk
memprediksi harga apartemen di Kota Daegu, Korea Selatan. Proyek
mencakup proses mulai dari eksplorasi data hingga pemilihan model
terbaik.

1. Latar Belakang

Harga apartemen di kota besar seperti Daegu dipengaruhi oleh banyak
faktor, seperti tipe bangunan, fasilitas publik, akses transportasi, dan
tahun pembangunan. Untuk memahami faktor-faktor tersebut dan membuat
prediksi harga yang akurat, digunakan model Machine Learning berdasarkan
dataset real estate Daegu.

2. Tujuan Proyek

-   Membangun model regresi untuk memprediksi harga apartemen.
-   Mengetahui fitur-fitur apa yang paling berpengaruh terhadap harga.
-   Melakukan evaluasi performa model menggunakan metrik regresi.
-   Membuat model yang siap digunakan untuk prediksi ke depan.

3. Data Understanding

Dataset berisi kolom-kolom seperti: - Size(sqf) - YearBuilt -
HallwayType - ProvAmenities (fasilitas sekitar) - Parking lot basement -
PublicOffice, University, Subway - SalePrice (target)

Total data sekitar 2.900 baris dan ~15 kolom.

4. Data Preparation

-   Missing value diimputasi (modus/median).
-   Encoding kategori menggunakan Label Encoding.
-   Scaling dilakukan pada model berbasis jarak.
-   Train-test split 70:30.

5. Exploratory Data Analysis (EDA)

Analisis awal menunjukkan: - Harga apartemen lebih tinggi pada hallway
type tertentu. - Fasilitas sekitar berhubungan positif dengan harga. -
Ukuran rumah dan tahun pembangunan berpengaruh signifikan.

6. Modeling

Model yang dicoba: 
1. Linear Regression 
2. KNN Regression 
3. RandomForest 
4. HistGradientBoosting 
5. XGBoost (tuned)

Evaluasi menggunakan: 
- MAE 
- RMSE 
- R² Score

7. Hasil Model Terbaik

Model terbaik: XGB_tuned 
- R²   : 0.8497 (mendekati 1)
- MAE  : 32813.09 (terendah)
- RMSE : 41078.46 (paling kecil)

8. Feature Importance

Top 5 fitur paling berpengaruh: 
1. HallwayType_enc 
2. N_FacilitiesNearBy(ETC) 
3. YearBuilt 
4. Size(sqf) 
5. N_Parkinglot(Basement)

Interpretasi:
- Hallway type sangat memengaruhi kenyamanan dan harga. 
- Semakin banyak fasilitas sekitar → harga meningkat. 
- Rumah baru → harga lebih mahal. 
- Unit lebih besar → harga lebih tinggi.

9. Residual Analysis

-   Residual tersebar acak → model tidak bias.
-   Ada underprediction untuk apartemen harga tinggi.
-   Pola error masih wajar untuk model real estate.

10. Actual vs Prediction

Sebagian besar titik berada dekat garis diagonal → model cukup akurat.
Untuk harga premium, prediksi sedikit lebih rendah.

11. Deployment

File penting: 
- best_daegu_model.pkl 
- label_encoders.pkl 
- model_features.pkl

Cara pakai: 
1. Encode fitur kategori. 
2. Susun input sesuai model_features. 
3. Gunakan model.predict().

12. Kesimpulan

-   XGB_tuned adalah model terbaik.
-   Fitur hallway type dan fasilitas sekitar adalah penentu harga
    terbesar.
-   Model dapat membantu developer, pembeli, dan analis real estate.

13. Rekomendasi

-   Tambahkan fitur eksternal seperti data kriminalitas.
-   Gunakan model boosting lain untuk membandingkan performa.
-   Tambahkan data spasial (GIS mapping).
