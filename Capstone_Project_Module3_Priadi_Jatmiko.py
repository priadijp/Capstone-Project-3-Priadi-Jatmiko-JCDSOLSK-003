#!/usr/bin/env python
# coding: utf-8

# ### **Prediksi Harga Apartemen di Kota Daegu Menggunakan Model Regresi**

# ### **Permasalahan Bisnis**

# **Latar Belakang**
# 
# Kota Daegu merupakan salah satu kota metropolitan besar di Korea Selatan dengan pertumbuhan sektor properti yang sangat pesat. Sebagai pusat pendidikan, transportasi, dan aktivitas bisnis, harga apartemen di Daegu terus mengalami peningkatan dan menjadi perhatian penting bagi pembeli, investor, maupun pengembang properti.
# 
# Mengetahui faktor apa saja yang memengaruhi harga apartemen serta membuat model prediksi harga yang akurat menjadi kebutuhan penting, yaitu dengan menggunakan pendekatan analitis Regresi, karena target prediksi harga jual adalah berupa angka. Dengan adanya model prediksi harga yang reliabel, berbagai pihak dapat mengambil keputusan lebih tepat seperti:
# 
# * Menentukan harga jual optimal
# * Menganalisis faktor paling berpengaruh
# * Menilai kewajaran harga pada suatu area
# 
# Karena itu, memprediksi harga apartemen berdasarkan berdasarkan variabel lokasi, fasilitas, ukuran apartemen, dan usia bangunan sangat diperlukan.

# **Rumusan Masalah**
# 
# * Variabel apa saja yang memengaruhi harga apartemen di Daegu?
# * Bagaimana distribusi masing-masing fitur dalam dataset?
# * Bagaimana melakukan pembersihan dan persiapan data sebelum modeling?
# * Model regresi apa yang menghasilkan performa terbaik?
# * Seberapa besar tingkat akurasi model dalam memprediksi harga apartemen?

# **Tujuan**
# 
# * Melakukan eksplorasi dataset Daegu Apartment secara menyeluruh.
# * Melakukan data preparation sesuai prosedur machine learning.
# * Mengembangkan model prediksi harga apartemen menggunakan regresi.
# * Mengukur performa model menggunakan metrik standar regresi (MAE, MSE, RMSE, R²).
# * Mengidentifikasi fitur yang paling berpengaruh dalam menentukan harga.

# **Pendekatan Analisis**
# 
# Untuk dapat menghasilkan prediksi harga jual apartemen yang andal, langkah pertama yang perlu dilakukan adalah menganalisis data untuk menemukan pola, hubungan, dan karakteristik penting dari setiap fitur yang tersedia seperti ukuran unit, fasilitas dalam kompleks apartemen, kedekatan dengan stasiun subway, jumlah fasilitas publik di sekitar, serta tahun pembangunan gedung.
# 
# Dengan memahami pola tersebut, kita dapat mengetahui faktor-faktor apa saja yang memberikan pengaruh besar terhadap harga jual apartemen di Daegu.
# 
# Selanjutnya, kita akan membangun sebuah model regresi yang mampu memberikan prediksi harga secara akurat berdasarkan fitur-fitur tersebut. Model prediksi ini nantinya dapat digunakan oleh:
# * developer properti
# * agen real estate
# * investor
# * calon pembeli
# untuk memperkirakan nilai pasar dari sebuah unit apartemen baru maupun existing, sehingga proses pengambilan keputusan dapat menjadi lebih cepat, informatif, dan kompetitif.
# 
# Dengan adanya alat prediksi seperti ini, pihak-pihak terkait dapat mengoptimalkan strategi harga, meningkatkan daya tarik properti, serta mengurangi ketidakpastian dalam menentukan nilai yang tepat.

# ### **Pemahaman Data**

# * Dataset yang digunakan adalah dataset Daegu Apartment mulai tahun 1978 sampai dengan 2015.
# * Setiap baris data merupakan informasi jenis apartemen, karakteristik properti, lokasi dan harga.
# 
# **Informasi Atribut**
# 
# | **Atribut** | **Tipe Data** | **Penjelasan** |
# | --- | --- | --- |
# | HallwayType | Object | Jenis apartemen (Terraced, Corridor, Mixed) |
# | TimeToSubway | Object | Waktu yang dibutuhkan untuk mencapai stasiun kereta bawah tanah terdekat (0-5min, 5~10min, 10~15min, 15-20min, no_bus_stop_nearby) |
# | SubwayStation | Object | Nama stasiun kereta bawah tanah terdekat |
# | N_FacilitiesNearBy(ETC) | Float | Jumlah fasilitas di sekitar (0 s.d. 5) |
# | N_FacilitiesNearBy(PublicOffice) | Float | Jumlah fasilitas lembaga publik di sekitar (0 s.d. 7) |
# | N_SchoolNearBy(University) | Float | Jumlah universitas di sekitar (0 s.d. 5) |
# | N_Parkinglot(Basement) | Float | Jumlah area parkir |
# | YearBuilt | Integer | Tahun pembangunan apartemen (1978 s.d. 2015) |
# | N_FacilitiesInApt | Integer | Jumlah fasilitas di dalam apartemen (1 s.d. 10) |
# | Size(sqft) | Integer | Ukuran apartemen (dalam kaki persegi) |
# | SalePrice | Integer | Harga jual apartemen (Won) |
# 
# 
# <br>

# **Import Library dan Membaca Data**

# In[1]:


# Import library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# Membaca Dataset dan menampilkan 5 baris teratas
df = pd.read_csv('data_daegu_apartment.csv')
df.head()



# **Duplikat DataFrame untuk Modeling**

# In[2]:


# Duplikat dari DataFrame asli untuk modeling
df_model = df.copy()

# summary table per fitur (df_model_desc)
listItem = []

for col in df_model.columns:
    uniq_vals = df_model[col].drop_duplicates()
    if len(uniq_vals) >= 2:
        sample_vals = uniq_vals.sample(min(2, len(uniq_vals))).values
    else:
        sample_vals = uniq_vals.values
    listItem.append([
        col,
        str(df_model[col].dtype),
        int(df_model[col].isna().sum()),
        round((df_model[col].isna().sum()/len(df_model[col]))*100, 2),
        int(df_model[col].nunique()),
        list(sample_vals)
    ])
df_model_desc = pd.DataFrame(listItem, columns=['dataFeatures','dataType','null','nullPct','unique','uniqueSample'])
df_model_desc



# * Membuat salinan dari DataFrame asli sebelum proses pemodelan berupa df_model menjadi data kerja, agar data asli tetap aman.
# * df_model_desc membantu menentukan antara lain: imputasi, encoding variable kategorikal, training model (RandomForest, XGBoost, Linear Regression, dll), evaluasi model.
# * Terdapat 3 kolom tipe object (kategorikal): HallwatType, TimeToSubway, SubwayStation perlu encoding.

# **Penanganan Missing Value dan Pembersihan Data**

# In[3]:


# Imputasi - mengisi nilai kosong (NaN)
# Strategi Missing Value (numerik -> median, kategorikal -> mode)
num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_model.select_dtypes(include=['object','category']).columns.tolist()

# Tidak termasuk SalePrice untuk imputasi target handling
num_cols_model = [c for c in num_cols if c != 'SalePrice']

# Imputasi numerik dengan median
for c in num_cols_model:
    if df_model[c].isna().sum() > 0:
        med = df_model[c].median()
        df_model[c] = df_model[c].fillna(med)
        print(f"Isi numerik {c} dengan median={med}")

# Imputasi kategorikal dengan mode
for c in cat_cols:
    if df_model[c].isna().sum() > 0:
        modev = df_model[c].mode(dropna=True)[0]
        df_model[c] = df_model[c].fillna(modev)
        print(f"Isi kategorikal {c} dengan mode='{modev}'")

print("Data kosong setelah imputasi:")
print(df_model.isnull().sum())



# Tidak ada Data Kosong/Missing Value (NaN) pada Dataset.

# In[4]:


# Skeweness
skew_table = df_model.skew(numeric_only=True).to_frame(name='Skewness')
print(skew_table)


# * Skewness dilakukan untuk membantu memahami suatu data, apakah berdistribusi normal (simetris) atau sangat condong ke kiri/kanan, untuk data bertipe numerik.
# * Sebagian besar numerik right-skewed (skewness positif).
# * Size(sqf) skew = 0.875, ada beberapa unit apartemen memiliki ukuran yang besar.
# * YearBuilt skew = -0.807, mayoritas bangunan relatif baru.
# * SalePrice cenderung right-skewed juga (terdapat beberapa unit premium).

# **Distribusi Data Numerik**
# 
# Distribusi data numerik sangat penting untuk Analisis Data Eksploratori, karena memberikan gambaran awal tentang bentuk, pola, kecenderungan nilai dan potensi masalah pada setiap variabel numerik:
# * Mengetahui pola sebaran data
# * Mendeteksi outlier
# * Menentukan kebutuhan transformasi
# * Membantu memilih algoritma
#   - Model linear regression sensitif terhadap distribusi dan outlier
#   - Model tree-based (Random Forest dan XGB) tidak mudah terpengaruh terhadap distribusi tidak normal 

# In[5]:


# Ambil kolom kategorikal
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

n_cols = 4   # jumlah grafik per baris
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

plt.figure(figsize=(18, 5 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribusi Berdasarkan {col}', size=8)
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.suptitle('Distribusi Data Numerik', fontsize=20, weight='bold')
plt.show()



# Fitur yang berpengaruh besar terhadap harga:
# * Size(sqf)
# * N_FacilitiesInApt
# * N_Parkinglot(Basement)
# * YearBuilt
# 
# Fitur yang berpengaruh kecil terhadap harga:
# * N_SchoolNearBy(University)
# * N_FacilitiesNearBy(PublicOffice)
# * N_FacilitiesNearBy(ETC)
# 
# 
# 
# 
# 

# **Distribusi Data Kategorikal**
# 
# Distribusi Data Kategorikal sangat penting untuk memberi wawasan tentang kondisi lingkungan, preferensi pasar dan karakteristik unit apartemen yang tidak bisa ditangkap oleh data numerik antara lain:
# * Memahami profil lingkungan dan karakter apartemen
# * Menilai faktor lokasi yang memengaruhi harga
# * Menentukan jenis encoding terbaik
# * Menghindari bias analisis
# * Memahami target pasar tiap kategori
# * Menyiapkan analisis harga yang lebih akurat
# 

# In[6]:


# Ambil kolom kategorikal
cat_cols = df.select_dtypes(include=['object', 'category']).columns

n_cols = 1
n_rows = int(np.ceil(len(cat_cols) / n_cols))

plt.figure(figsize=(20, 5 * n_rows))

for i, col in enumerate(cat_cols, 1):

    # Buat frekuensi dan sort secara ascending
    freq = df[col].value_counts(ascending=False).index

    plt.subplot(n_rows, n_cols, i)
    sns.countplot(data=df, y=col, order=freq)
    plt.title(f'Distribusi {col} (sorted)', size=18, weight='bold')
    plt.xlabel('Jumlah', size=16)
    plt.ylabel(col, size=16)

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.suptitle('Distribusi Data Kategorikal (Descending)', fontsize=22, weight='bold')
plt.show()



# * HallwayType : Terraced paling dominan, sulit untuk melihat pengaruhnya terhadap harga.
# * TimeToSubway : 0-5min paling dominan, kemungkinan memiliki harga tinggi.
# * SubwayStation : Stasiun Kyungbuk_uni_hospital paling dominan, kemungkinan harga lebih murah, karena dekat kampus dan segmentasinya untuk mahasiswa.
#   

# **Hitung Korelasi Numerik Antar Fitur**
# 
# Menghitung Korelasi Numerik Antar Fitur sangat penting dalam analisis data, karena korelasi memberi gambaran seberapa kuat hubungan antar variabel numerik di dalam dataset.
# * Mengetahui fitur mana yang paling mempengaruhi harga
# * Menentukan fitur yang redundant (multicollinearity)
# * Memahami hubungan alami antar variabel
# * Melakukan feature selection
# * Membuat visualisasi heatmap yang membantu EDA
# * Mempersiapkan data untuk machine learning

# In[7]:


# Hitung korelasi hanya untuk kolom numerik
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='white')

plt.title("Heatmap Korelasi Antar Fitur", fontsize=18, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.show()



# **1. Size(sqft) : 0.70**
# * Sangat berpengaruh terhadap harga jual
# * Ukuran makin besar, maka semakin mahal
# 
# **2. N_FacilitiesInApt : 0.51**
# * Jumlah fasilitas di dalam apartemen (contohnya: gym, keamanan, lift) sangat berpengaruh
# * Semakin banyak fasilitas, maka harga cenderung semakin mahal
# 
# **3. N_Parkinglot(Basement) : 0.47**
# * Banyaknya slot parkir basement juga berpengaruh terhadap harga jual
# 
# **4. YearBuilt : 0.45**
# * Apartemen baru cenderung lebih mahal
# 
# **5. N_FacilitiesNearBy(PublicOffice) : -0.44**
# * Korelasi negatif, semakin banyak fasilitas di sekitar yang bersifat ramai/padat, maka harga cenderung turun
# 
# **6. N_FacilitiesNearBy(ETC) : -0.45**
# * Korelasi negatif, semakin banyak fasilitas di sekitar yang bersifat ramai/padat, maka harga cenderung turun
# 
# **7. N_FacilitiesNearBy(University) : -0.39**
# * Korelasi negatif, semakin banyak fasilitas di sekitar yang bersifat ramai/padat, maka harga cenderung turun
# 
# Size(sqf) jelas faktor utama menaikkan harga. Nilai negatif pada fasilitas sekitar bisa menunjukkan area padat yang menurunkan harga (domain knowledge: beberapa fasilitas publik besar bisa menurunkan harga area karena kepadatan/kemacetan).
# 
# 

# ### **Persiapan Data**

# **Label Encoding Fitur Kategorikal**
# 
# Label Encoding pada fitur kategorikal digunakan untuk mengubah kategori (teks) menjadi angka agar bisa diproses oleh model Machine Learning.
# Model tidak bisa membaca teks, sehingga fitur kategorikal harus dikonversi.
# 

# In[8]:


# Label encoding 3 fitur kategorikal
encode_cols = ['TimeToSubway','HallwayType','SubwayStation']
label_maps = {}

for col in encode_cols:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))
        label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        print(f'{col} tidak ada dalam kolom, lewati.')

# Tampilkan mapping + jumlah
for col, mapping in label_maps.items():
    print(f'\nMapping {col} (label -> kategori -> jumlah):')

    # ambil jumlah kemunculan kategori
    counts = df_model[col].value_counts()

    # tampilkan label, kategori, dan jumlah dalam urutan label descending
    for cat, lbl in sorted(mapping.items(), key=lambda x:x[1], reverse=True):
        jumlah = counts[cat]
        print(f'  {lbl} -> {cat} = {jumlah}')




# Label encoding menghasilkan label integer. Mapping disimpan untuk laporan. TimeToSubway tetap dianggap kategori (encoded only)

# In[9]:


df_model.info()


# **Pemilihan Fitur dan Pengisian Defensif**
# 
# **Pemilihan Fitur**
# Proses memilih fitur (kolom) mana yang paling relevan dan penting untuk digunakan dalam analisis atau model Machine Learning yang bertujuan untuk:
# * Menyederhanakan model
# * Menghilangkan noise
# * Mencegah overfitting
# * Meningkatkan interpretasi
# * Mengurangi fitur yang saling tumpang tindih
# 
# **Pengisian Defensif**
# Mengisi missing value dengan cara yang aman agar:
# * Tidak merusak data
# * Tidak mempengaruhi distribusi
# * Tidak membuat model bias
# 

# In[10]:


# Menentukan Fitur
candidate_features = [
    'TimeToSubway_enc','SubwayStation_enc','HallwayType_enc',
    'N_FacilitiesNearBy(ETC)','N_FacilitiesNearBy(PublicOffice)',
    'N_SchoolNearBy(University)','N_Parkinglot(Basement)',
    'N_FacilitesInApt','Size(sqf)','YearBuilt'
]
# Menyesuaikan nama fitur jika menggunakan suffix _enc
features = [f for f in candidate_features if f in df_model.columns]
print('Fitur yang digunakan:', features)

X = df_model[features].copy()
y = df_model['SalePrice'].copy()

# Mengisi sisa ruang
for c in X.columns:
    if X[c].isna().sum() > 0:
        X[c] = X[c].fillna(X[c].median())
        print(f'Mengisi sisa NaN {c} dengan median')



# **Train-Test Split (70:30)**
# 
# Untuk memisahkan data menjadi 2 bagian:
# * Melatih Model (Training Data – 70%)
# * Menguji Model Secara Adil (Test Data – 30%)
# 
# 
# 

# In[11]:


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)


# * Menggunakan 70:30 (train : test). Model menggunakan random state agar dapat menghasilkan output yang konsisten.
# * Jumlah data untuk training model : 2886, jumlah fitur training : 9
# * Jumlah data untuk testing model : 1237, jumlah fitur testing : 9

# ### **Pemodelan dan Evaluasi**

# **Pemodelan**
# 
# Pemodelan bertujuan untuk:
# * Menangkap pola hubungan antar fitur
# * Membuat prediksi untuk data baru (Linear Regression, Random Forest, XGBoost, dll)
# * Menguji hipotesis dalam data
# 
# **Evaluasi**
# 
# Evaluasi bertujuan untuk:
# * Mengetahui apakah model akurat
# * Mencegah overfitting atau underfitting
# * Membandingkan beberapa model
# * Mengecek apakah model layak digunakan di dunia nyata
# 
# 
# 
# 
# **Membuat Pipeline**
# 
# Menyusun seluruh langkah preprocessing dan pemodelan Machine Learning ke dalam satu rangkaian kerja (workflow) yang rapi, otomatis, dan konsisten.

# In[12]:


# Membuat Pipeline
# pipe_lr (StandardScaler + LinearRegression)
# pipe_knn (StandardScaler + KNN)
# pipe_rf (RandomForest tanpa Scaler)
# pipe_hgb (HistGradientBoosting tanpa Scaler)
# pipe_xgb (XGB)

# Deteksi apakah XGBoost tersedia
try:
    xgb_available = True
except ImportError:
    xgb_available = False

# Deteksi apakah CatBoost tersedia
try:
    cat_available = True
except ImportError:
    cat_available = False

# Kolom numerik untuk scaling (semua numerik)
numeric_features = X.columns.tolist()

pipe_lr = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
pipe_knn = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())])
pipe_rf = Pipeline([('model', RandomForestRegressor(random_state=10, n_estimators=200))])
pipe_hgb = Pipeline([('model', HistGradientBoostingRegressor(random_state=10))])

if xgb_available:
    pipe_xgb = Pipeline([('model', XGBRegressor(random_state=10, verbosity=0))])
else:
    pipe_xgb = None

print('Pipelines siap')
print ('XGBoost tersedia:', xgb_available)



# Scaler digunakan untuk model berbasis jarak / linear. Tree models tidak butuh scaling.

# In[13]:


# Baseline Training
models = {
    'LinearRegression': pipe_lr,
    'KNN': pipe_knn,
    'RandomForest': pipe_rf,
    'HistGradientBoosting': pipe_hgb
}
if pipe_xgb:
    models['XGBoost'] = pipe_xgb

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results[name] = {'model': model, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f'{name}: \nMAE = {mae:.2f} \nRMSE = {rmse:.2f} \nR² = {r2:.4f}\n')

    


# MAE (rata-rata selisih absolut antara prediksi dan nilai asli)
#   - Semakin kecil, semakin akurat
# 
# RMSE (rata-rata jarak prediksi terhadap nilai asli)
#   - Semakin kecil, semakin akurat
#   
# R² (R-Squared)
#   - Semakin mendekati 1, semakin baik
#   

# **Hyperparameter Tuning**
# 
# Proses mencari kombinasi pengaturan terbaik untuk sebuah model machine learning agar menghasilkan performa yang paling optimal.
# 
# 

# In[14]:


# RandomForest param grid (moderate)
rf_param = {
    'model__n_estimators': [100, 200, 400],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

rs_rf = RandomizedSearchCV(pipe_rf, rf_param, n_iter=10, cv=5, scoring='r2', random_state=10, n_jobs=-1)
rs_rf.fit(X_train, y_train)
print('RF best params:', rs_rf.best_params_, 'best R² (cv):', rs_rf.best_score_)

# HistGradientBoosting param grid
hgb_param = {
    'model__max_iter': [100, 200, 400],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [None, 10, 20]
}
rs_hgb = RandomizedSearchCV(pipe_hgb, hgb_param, n_iter=8, cv=5, scoring='r2', random_state=10, n_jobs=-1)
rs_hgb.fit(X_train, y_train)
print('HGB best params:', rs_hgb.best_params_, 'best R² (cv):', rs_hgb.best_score_)

# XGBoost tuning jika tersedia
if pipe_xgb:
    xgb_param = {
        'model__n_estimators': [100,200,400],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3,6,10]
    }
    rs_xgb = RandomizedSearchCV(pipe_xgb, xgb_param, n_iter=10, cv=5, scoring='r2', random_state=10, n_jobs=-1)
    rs_xgb.fit(X_train, y_train)
    print('XGB best params:', rs_xgb.best_params_, 'best R² (cv):', rs_xgb.best_score_)



# **Evaluasi Model Tuned vs Baseline**
# 
# Proses membandingkan performa model sebelum dan sesudah dilakukan hyperparameter tuning.
# 
# 

# In[15]:


# Evaluasi model yang telah disesuaikan pada kumpulan data uji
tuned_models = {
    'RF_tuned': rs_rf.best_estimator_,
    'HGB_tuned': rs_hgb.best_estimator_
}
if pipe_xgb and 'rs_xgb' in globals():
    tuned_models['XGB_tuned'] = rs_xgb.best_estimator_

# Evaluate all tuned models + baseline
eval_rows = []
# add baseline models already fitted in results dict
for name in results:
    pred = results[name]['model'].predict(X_test)
    eval_rows.append([name, results[name]['MAE'], results[name]['RMSE'], results[name]['R2']])

# evaluate tuned
for name, model in tuned_models.items():
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    eval_rows.append([name, mae, rmse, r2])
    print(f'{name}: \nMAE = {mae:.2f} \nRMSE = {rmse:.2f} \nR² = {r2:.4f}\n')

eval_df = pd.DataFrame(eval_rows, columns=['Model','MAE','RMSE','R2']).sort_values('R2', ascending=False).reset_index(drop=True)
eval_df



# XGB_tuned merupakan model terbaik di semua metrik:
# 
# MAE  : error rata-rata terkecil
# RMSE : kesalahan lebih sedikit
# R²   : tertinggi, variasi harga paling baik
# 
# Tuning XGBoost memang efektif.

# **Fitur Importance**
# 
# Ukuran yang menunjukkan seberapa besar pengaruh setiap fitur (kolom) terhadap hasil prediksi model machine learning.
# 
# 

# In[16]:


# Fitur importance dari model tree-based
# pick best tuned tree model by R²
best_row = eval_df.iloc[0]
best_name = best_row['Model']
print('Model terbaik:', best_name)
print()

# Mengambil objek model
best_model_obj = None
if best_name in tuned_models:
    best_model_obj = tuned_models[best_name]
else:
    best_model_obj = results.get(best_name, {}).get('model', None)

# pipeline untuk akses model dasar
if isinstance(best_model_obj, Pipeline):
    underlying = best_model_obj.named_steps['model']
else:
    underlying = best_model_obj

# Menggunakan importances jika tersedia
if hasattr(underlying, 'feature_importances_'):
    importances = pd.Series(underlying.feature_importances_, index=X.columns).sort_values(ascending=False)
    display(importances)
    plt.figure(figsize=(8,5))
    importances.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f'Fitur importance ({best_name})')
    plt.show()
else:
    print("Model dasar tidak memiliki feature_importances_ (misalnya LinearRegression).")


# HallwayType_enc = 0.65
# * Jenis apartemen sangat berpengaruh terhadap harga unit di Daegu.
# * Tipe Terraced paling banyak diminati pembeli.
# 
# N_FacilitiesNearBy(ETC) = 0.18
# * Fasilitas umum sekitar apartemen (toko, restoran, fasilitas kebutuhan harian) sangat mempengaruhi harga.
# 
# YearBuilt = 0.059
# * Semakin baru bangunan → harga cenderung lebih tinggi.
# 
# Size(sqf) = 0.053
# * Biasanya ukuran adalah faktor terbesar, tapi di dataset ini lebih kecil dari HallwayType.
# 
# N_Parkinglot(Basement) = 0.0466
# * Jumlah parkir basement cukup berpengaruh terhadap prediksi harga.
# 
# Fitur dengan pengaruh sangat kecil (<0.005):
# * N_FacilitiesNearBy(PublicOffice)
# * SubwayStation_enc
# * N_SchoolNearBy(University)
# * TimeToSubway_enc
# 

# **Residual Analisis dan Aktual vs Prediksi**
# 
# Residual analysis digunakan untuk:
# * Mengecek apakah model sudah baik
# * Menemukan bias
# * Menemukan outlier
# 

# In[17]:


# Residuals untuk model terbaik
pred_best = best_model_obj.predict(X_test)
residuals = y_test - pred_best

plt.figure(figsize=(6,6))
plt.scatter(pred_best, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Prediksi')
plt.ylabel('Residual (Aktual - Prediksi)')
plt.title(f'Residual plot ({best_name})')
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('SalePrice Aktual')
plt.ylabel('Prediksi SalePrice')
plt.title(f'Aktual vs Prediksi ({best_name})')
plt.show()

# Ringkasan residuals
print("Residuals mean:", residuals.mean(), "std:", residuals.std())



# Residual Plot:
# * Sebaran residual berada di sekitar garis 0 --> model cukup baik, karena prediksi tidak terlalu menyimpang.
# * Terdapat sebaran residual yang cukup besar (hingga ±100.000) --> beberapa prediksi masih meleset jauh, karena variasi harga sangat besar.
# * Residual menyebar tidak membentuk pola yang jelas --> error bersifat acak, tidak ada pola tertentu, model tidak bias.
# * Sedikit bentuk spread mengembang --> semakin tinggi harga rumah, semakin besar error prediksinya.
# 
# Plot Aktual vs Prediksi (XGB_tuned):
# * Titik-titik mengikuti garis diagonal dengan cukup rapat --> model memiliki performa prediksi yang baik.
# * Sebaran yang cukup lebar pada harga yang lebih tinggi --> model lebih sulit memprediksi harga yang tinggi karena variasinya lebih besar.
# * Beberapa prediksi lebih rendah dan lebih tinggi dari nilai sebenarnya --> ada titik yang berada di bawah garis dan di atas garis, menunjukkan variasi error dan masih normal.
# 
# 
# 

# In[18]:


# Simpan Mode/Prediksi
preds_df = X_test.copy().reset_index(drop=True)
preds_df['Actual'] = y_test.reset_index(drop=True)
preds_df[f'Pred_{best_name}'] = pred_best
preds_df.head()



# ### **Kesimpulan**

# * Model terbaik: __XGB_tuned__ dengan R² = __0.8497__. Dari hasil baseline & tuning, model tree-based (RandomForest / HGB / XGBoost / CatBoost) menghasilkan performa terbaik dibanding Linear Regression dan KNN.
# * Fitur penentu harga: Size(sqf), YearBuilt, N_Parkinglot(Basement), TimeToSubway, N_FacilitiesInApt.
# * Keterbatasan: model cenderung bekerja kurang baik untuk harga ekstrem (outliers). Beberapa variabel lingkungan yang berpengaruh (kualitas sekolah, crime rate, walking score) tidak tersedia.
# * Validitas: model dapat menjelaskan sebagian besar variasi harga. 

# ### **Rekomendasi**

# * Untuk Pembeli: pilih unit dengan ukuran optimal dan kedekatan stasiun untuk investasi jangka panjang.
# * Untuk Penjual/Developer: meningkatkan fasilitas parkir dan akses transportasi, renovasi untuk menaikkan YearBuilt-equivalent value.
# * Untuk Data Team/Next Iteration:
#   - Tambah fitur eksternal (crime, school quality, walking score).
#   - Lakukan tuning lebih intensif (Bayesian Optimization) untuk XGBoost/CatBoost.
#   - Pertimbangkan log-transform target jika residual heteroskedastis.
#   - Lakukan A/B test penerapan harga berdasarkan prediksi.

# ****

# #### **Simpan Model**

# In[19]:


# SIMPAN MODEL & ENCODER

import joblib

joblib.dump(best_model_obj, 'best_daegu_model.pkl')

joblib.dump(label_maps, 'label_encoders.pkl')

joblib.dump(features, 'model_features.pkl')

print('Model terbaik, encoder, dan fitur berhasil disimpan!')



# #### **Load Model untuk Prediksi Baru**

# In[20]:


model = joblib.load('best_daegu_model.pkl')
encoders = joblib.load('label_encoders.pkl')
features = joblib.load('model_features.pkl')

# Contoh input baru
data_baru = {
    'TimeToSubway': '5min~10min',
    'HallwayType': 'mixed',
    'SubwayStation': 'Bangoge',
    'N_FacilitiesNearBy(ETC)': 5,
    'N_FacilitiesNearBy(PublicOffice)': 5,
    'N_SchoolNearBy(University)': 4,
    'N_Parkinglot(Basement)': 798,
    'N_FacilitesInApt': 7,
    'Size(sqf)': 914,
    'YearBuilt': 2005
}

# Encode kategori
for col in ['TimeToSubway','HallwayType','SubwayStation']:
    data_baru[col + '_enc'] = encoders[col][data_baru[col]]

# Hapus yang tidak dipakai
data_baru = {k: v for k, v in data_baru.items() if k.endswith('_enc') or k in [
    'N_FacilitiesNearBy(ETC)','N_FacilitiesNearBy(PublicOffice)',
    'N_SchoolNearBy(University)','N_Parkinglot(Basement)',
    'N_FacilitesInApt','Size(sqf)','YearBuilt'
]}

# Konversi ke DataFrame sesuai fitur model
df_input = pd.DataFrame([data_baru])[features]

# Prediksi
prediksi = model.predict(df_input)
print('Prediksi Harga:', prediksi[0])


# In[ ]:




