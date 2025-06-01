# -*- coding: utf-8 -*-
import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from zipfile import ZipFile
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

"""## Data loading

Selanjutnya dilakukan pengambilan dataset dari kaggle, dengan cara memanggil dataset tersebut dari kaggle dataset
"""

!mkdir -p ~/.kaggle
!echo '{"username":"fajriharyanto","key":"998acd734e359f906329500715f20f7e"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d aprabowo/indonesia-tourism-destination

"""Kemudian dataset yang sudah dipanggil, akan diajadikan zip, dan di ekstrak menjadi file csv untuk digunakan dalam analisa ini."""

!unzip -q indonesia-tourism-destination.zip -d /content

!ls

"""Dataset 1, digunakan dataset tourism_with_id.csv: berisi informasi detail mengenai tempat wisata."""

Destinations_df = pd.read_csv("/content/tourism_with_id.csv")
Destinations_df.head()

"""Dataset 2, digunakan dataset tourism_rating.csv: berisi data rating dari pengguna terhadap tempat wisata."""

Reviews_df = pd.read_csv("/content/tourism_rating.csv")
Reviews_df.head()

"""## Data Type Information

Pada tahap ini, akan melihat bagimana tipe data pada dataset, jumlah informasi Statistik awal pada dataset
"""

Destinations_df.info()

Reviews_df.info()

print('Jumlah data wisata yang tersedia', len(Destinations_df.Place_Id.unique()))
print('Jumlah tempat wisata yang tersedia', len(Destinations_df.Place_Name.unique()))
print('Kategori tempat wisata yang tersedia: ', len(Destinations_df.Category.unique()))
print('jumlah daerah tempat wisata yang tersedia', len(Destinations_df.City.unique()))
print('Jumlah data user yang melakukan review', len(Reviews_df.User_Id.unique()))

"""Terlihat, lumayan bervariasi pada setiap kolom di dataset tersebut, hal ini sangat bagus preferensi informasi bisa lebih informatif

# **📈 Univariate Exploratory Data Analysis**

## Destinations Variabel
"""

Destinations_df.describe()

Destinations_df.shape


Destinations_df.head()

Destinations_df.tail()

print('Jumlah data wisata yang tersedia', len(Destinations_df.Place_Id.unique()))
print('Jumlah tempat wisata yang tersedia', len(Destinations_df.Place_Name.unique()))
print('Kategori tempat wisata yang tersedia: ', len(Destinations_df.Category.unique()))
print('jumlah daerah tempat wisata yang tersedia', len(Destinations_df.City.unique()))

plt.figure(figsize=(5, 3))
sns.countplot(y=Destinations_df['Category'], order=Destinations_df['Category'].value_counts().index)
plt.title('Distribusi kategori tempat wisata yang tersedia')
plt.show()

plt.figure(figsize=(5, 3))
sns.countplot(data=Destinations_df, x='City')
plt.title('Distribusi jumlah daerah tempat wisata yang tersedia')
plt.show()

## Reviews Variabel

Reviews_df.describe()

Reviews_df.shape


Reviews_df.head()

Reviews_df.tail()

print('Jumlah data tempat yang tersedia', len(Reviews_df.Place_Id.unique()))
print('Jumlah data user yang tersedia', len(Reviews_df.User_Id.unique()))
print('Jumlah data place rating', len(Reviews_df.Place_Ratings.unique()))
print('Jumlah data reviews', len(Reviews_df))

plt.figure(figsize=(5, 3))
sns.countplot(data=Reviews_df, x=Reviews_df['Place_Ratings'])
plt.show()

# **🧹 Data Preprocessing**

Pada tahap ini, akan dilakukan pengecekan awal missing value dan duplikasi pada data.

## Combining Tourist Attraction Data
"""

Tour_all = np.concatenate((
    Destinations_df.Place_Id.unique(),
    Reviews_df.Place_Id.unique()
), axis=0)

Tour_all = np.sort(np.unique(Tour_all))

print('Jumlah data wisata yang tersedia', len(Tour_all))

"""## Missing Value Cross Check"""

Destinations_df.isnull().sum()

"""Pada dataset **Destinations_df** ditemukan ada 2 indikasi missing value yaitu pada kolom Time_Minutes yang berjumlah 232 missing value dan juga kolom Unnamed sebanyak 437 missing value."""

Reviews_df.isnull().sum()

"""Pada dataset **Reviews_df** tidak ditemukan indikasi missing value sama sekali, yang berarti dataset ini lumayan aman, akan tetapi akan dilakukan kembali kedepannya apakah dataset ini beneran bagus atau tidak.

## Duplicated Cross Check
"""

print("Jumlah data duplikat pada destinasi: ", Destinations_df.duplicated().sum())
print("Jumlah data duplikat pada rewiew: ", Reviews_df.duplicated().sum())

"""Bisa diketahui dari output tersebut bahwa dataset **Destinations_df** tidak meiliki dupliaksi data sama sekali, sedang kolom **Reviews_df** meimiliki indikasi 79 duplikasi infoemasi data."""

duplicated = Reviews_df[Reviews_df.duplicated(keep=False)]
print(duplicated)

"""# **🍳 Data Preparation**

Pada tahap ini, membersihkan data dari nilai-nilai yang hilang, duplikat, dan anomali. Serta pada tahap ini akan dilakukan pembuatan *new distionary data* yang berguna untuk mempermudah analisa data kedepannya dan juga dilakukan teknik TF -IDF pada dataset yang digunakan Langkah ini penting untuk memastikan kualitas data yang akan dianalisis lebih lanjut

## Overcoming Missing Value

Pada tahap ini diputuskan untuk menghapus/menghilangkan kolom yang ada **missing value** karena terlalu banyak missing dan tidak bisa diimputasi secara akurat, atau tidak terlalu berdampak pada rekomendasi berbasis kategori.

Selain itu diputuskan juga untuk menghapus beberapa kolom yang tidak relevan agar bisa lebih sesuai tujuan penelitian untuk tidak digunakan dalam analisis rekomendasi, dan penghapusan kolom ini bisa membuat model lebih optimal kedepannya.
"""

Destinations_df = Destinations_df.drop(columns=['Time_Minutes', 'Coordinate', 'Price', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'], errors='ignore')
Destinations_df.isnull().sum()

"""Setelah proses pembersihan dataset Destinations_df, bisa kita lihat hanya tersisa beberapa kolom nah kolom ini yang akan kita gunakan untuk analisa kedepannya, dan juga kolom yang mengandung missing value sudah tidak ada.

"""

Destinations_df.head()

"""## Overcoming data duplication"""

Reviews_df.drop_duplicates(inplace=True)
Reviews_df.duplicated().sum()

"""Duplikasi pada dataset pada kolom **Reviews_df** berhasil tertatasi.

## Convert a series to a list

Pada tahap ini beberapa kolom pada dataset **Destinations_df** akan diubah menjadi *tolist()*, hal ini bertujuan agar mempermudah manipulasi data dan pembuatan struktur baru.
"""

tour_id = Destinations_df['Place_Id'].tolist()
tour_name = Destinations_df['Place_Name'].tolist()
tour_category = Destinations_df['Category'].tolist()

print('\nJumlah ID wisata:', len(tour_id))
print('Jumlah Nama wisata:', len(tour_name))
print('Jumlah Kategori wisata:', len(tour_category))

"""## Create a dictionary for the dataset

Diatahap ini, Dictionary baru dibuat untuk menyederhanakan proses pemetaan dan pengelompokan data wisata berdasarkan ID, nama, dan kategori
"""

tour_df = pd.DataFrame({
    'id': tour_id,
    'tour_name': tour_name,
    'category': tour_category
})

data = tour_df
data.sample(5)

"""## TF-IDF Vectorizer

Pada tahap ini dilakukan teknik untuk mempersiapkan atau mengubah fitur teks menjadi format numerik sebelum digunakan dalam pemodelan.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
tf.fit(data['category'])
tf.get_feature_names_out()

tfidf_matrix = tf.fit_transform(data['category'])
tfidf_matrix.shape

tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.tour_name
).sample(22, axis=1, replace=True).sample(10, axis=0)

"""# **🧭 Model Development dengan Content Based Filtering**

Langkap pertama dalam tahap pemodelan dengan CBF ini adalah menghitung tingkat kemiripan antar wisata menggunakan Cosine Similarity.

## Cosine Similarity
"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['tour_name'], columns=data['tour_name'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Recommendation testing"""

def tour_recommendation(place_name, top_k=5, similarity_data=cosine_sim_df, items=data[['tour_name', 'category']]):
    # Ambil rekomendasi tempat wisata berdasarkan nama tempat
    place_index = items[items['tour_name'].str.lower() == place_name.lower()].index
    if len(place_index) == 0:
        print(f"Tempat wisata dengan nama '{place_name}' tidak ditemukan.")
        return pd.DataFrame()

    sim_scores = list(enumerate(similarity_data.iloc[place_index[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k + 1]
    place_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Simpan similarity_score tapi tidak ditampilkan
    recommendations = items.iloc[place_indices].copy()
    recommendations['similarity_score'] = similarity_scores

    # Ambil kategori dari tempat wisata input
    query_category = data[data['tour_name'].str.lower() == place_name.lower()]['category'].values[0]
    query_categories = set(query_category.lower().replace(", ", ",").split(","))

    total_overlap = 0
    category_counts = []

    # Evaluasi kategori
    for _, row in recommendations.iterrows():
        rec_categories = set(row['category'].lower().replace(", ", ",").split(","))
        overlap = query_categories.intersection(rec_categories)
        total_overlap += len(overlap)
        category_counts.append(len(rec_categories))

    precision_at_k = total_overlap / sum(category_counts) if category_counts else 0

    print(f"Evaluasi untuk rekomendasi tempat yang mirip '{place_name}':")
    print(f"Tempat wisata yang pernah dikunjungi: {place_name} (Kategori: {query_categories})")
    print(f"\nRekomendasi Tempat Wisata Lainnya {top_k}:")

    # Tampilkan hanya kolom tour_name dan category
    display(recommendations[['tour_name', 'category']])

tour_recommendation("Rabbit Town")

"""# **🤝 Model Development dengan Collaborative Filtering**

## Data loading

Memanggil dataset kemabali untuk dilakukan analisa
"""

user = Reviews_df
user

"""## Data Preparation

Mengubah User_Id menjadi list tanpa nilai yang sama
"""

user = Reviews_df['User_Id'].unique().tolist()
print('list User_Id: ', user)

"""Melakukan encoding User_Id"""

user_to_user_encoded = {x: i for i, x in enumerate(user)}
print('encoded userID : ', user_to_user_encoded)

"""Melakukan proses encoding angka ke ke User_Id"""

user_encoded_to_user = {i: x for i, x in enumerate(user)}
print('encoded angka ke userID: ', user_encoded_to_user)

"""Mengubah Place_Id menjadi list tanpa nilai yang sama

"""

tour = Reviews_df['Place_Id'].unique().tolist()
print('list Place_Id: ', tour)

"""Melakukan proses encoding Place_Id"""

tour_to_tour_encoded = {x: i for i, x in enumerate(tour)}
print('encoded Place_Id: ', tour_to_tour_encoded)

"""Melakukan proses encoding angka ke Place_Id"""

tour_encoded_to_tour = {i: x for i, x in enumerate(tour)}
print('encoded angka ke Place_Id: ', tour_encoded_to_tour)

"""Mapping User_Id ke dataframe USER

"""

Reviews_df['USER'] = Reviews_df['User_Id'].map(user_to_user_encoded)

"""Mapping Place_Id ke dataframe TOUR

"""

Reviews_df['TOUR'] = Reviews_df['Place_Id'].map(tour_to_tour_encoded)

"""Menampilkan sample data setelah mapping"""

display(Reviews_df.head())

"""## Training data and data validation

### Training Process
"""

Reviews_df = Reviews_df.sample(frac=1, random_state=42)
Reviews_df

x = Reviews_df[['USER', 'TOUR']].values

# Mengubah rating menjadi nilai float (normalisasi)
min_rating = Reviews_df['Place_Ratings'].min()
max_rating = Reviews_df['Place_Ratings'].max()
y = Reviews_df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * Reviews_df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""### Building a Neural Collaborative Filtering Model

Pada tahap ini, dilakukan pemodelan sistem rekomendasi menggunakan pendekatan Content-Based Filtering, yaitu dengan merekomendasikan item yang memiliki kemiripan konten (kategori) dengan item yang dipilih oleh pengguna.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_tours, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-3)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.tour_embedding = layers.Embedding(
            input_dim=num_tours,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-4)
        )
        self.tour_bias = layers.Embedding(num_tours, 1)

        self.dropout = layers.Dropout(0.5)
        self.batchnorm = layers.BatchNormalization()

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        tour_vector = self.tour_embedding(inputs[:, 1])
        tour_bias = self.tour_bias(inputs[:, 1])

        dot_user_place = tf.reduce_sum(user_vector * tour_vector, axis=1, keepdims=True)
        x = dot_user_place + user_bias + tour_bias

        x = self.batchnorm(x)
        x = self.dropout(x)

        return x

num_users = len(user_to_user_encoded)
num_tours = len(tour_to_tour_encoded)
model = RecommenderNet(num_users, num_tours, embedding_size=20)


model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=20,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop]
)

"""## Tourist Attraction Recommendation Function"""

# Mengambil sample user
user_id = Reviews_df['User_Id'].sample(70).iloc[0]
print(f"User ID yang dipilih: {user_id}")

tours_visited_by_user = Reviews_df[Reviews_df['User_Id'] == user_id]
tour_ids_visited = tours_visited_by_user['Place_Id'].values
print(f"User ini telah mengunjungi {len(tour_ids_visited)} tempat.")

tours_not_visited = Destinations_df[~Destinations_df['Place_Id'].isin(tour_ids_visited)]
tours_not_visited_encoded = [
    tour_to_tour_encoded[x] for x in tours_not_visited['Place_Id'].values
    if x in tour_to_tour_encoded
]

user_encoded = user_to_user_encoded[user_id]
user_place_array = np.hstack([
    np.array([[user_encoded]] * len(tours_not_visited_encoded)),
    np.array(tours_not_visited_encoded).reshape(-1, 1)
])

predicted_ratings = model.predict(user_place_array).flatten()

top_n = 10
top_indices = predicted_ratings.argsort()[-top_n:][::-1]
recommended_encoded_ids = [tours_not_visited_encoded[i] for i in top_indices]
recommended_place_ids = [tour_encoded_to_tour[i] for i in recommended_encoded_ids]

print('\n' + '=' * 30)
print('Places with HIGH ratings by the user:')
print('-' * 30)

top_rated_tours = tours_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5)
top_tour_ids = top_rated_tours['Place_Id'].values
top_tour_info = tour_df[tour_df['id'].isin(top_tour_ids)]

for row in top_tour_info.itertuples():
    print(f"{row.tour_name} : {row.category}")

print('\n' + '-' * 30)
print(f'Top {top_n} place recommendations:')
print('-' * 30)

recommended_places_info = tour_df[tour_df['id'].isin(recommended_place_ids)]
for row in recommended_places_info.itertuples():
  print(f"{row.tour_name} : {row.category}")

"""# **🧪 Evaluation**

## Evaluation Content-based Filtering
"""

def evaluate_precision_recall(place_name, top_k=5, similarity_data=cosine_sim_df, items=data[['tour_name', 'category']]):
    place_index = items[items['tour_name'].str.lower() == place_name.lower()].index
    if len(place_index) == 0:
        print(f"Tempat wisata dengan nama '{place_name}' tidak ditemukan.")
        return

    sim_scores = list(enumerate(similarity_data.iloc[place_index[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k + 1]
    place_indices = [i[0] for i in sim_scores]

    recommendations = items.iloc[place_indices].copy()

    # Kategori input
    query_category = data[data['tour_name'].str.lower() == place_name.lower()]['category'].values[0]
    query_categories = set(query_category.lower().replace(", ", ",").split(","))

    # Kategori rekomendasi
    recommended_categories = []
    for cat_str in recommendations['category']:
        cats = set(cat_str.lower().replace(", ", ",").split(","))
        recommended_categories.append(cats)

    true_positives = 0
    total_recommended = len(recommended_categories)
    total_relevant = len(query_categories)

    for rec_cats in recommended_categories:
        if len(rec_cats.intersection(query_categories)) > 0:
            true_positives += 1

    precision = true_positives / total_recommended if total_recommended > 0 else 0
    recall = true_positives / total_relevant if total_relevant > 0 else 0

    print('\n' + '='*30)
    print(f"Evaluasi Precision & Recall untuk rekomendasi berdasarkan '{place_name}':")
    print('-'*30)
    print(f"Precision @ {top_k}: {precision:.2f}")
    print(f"Recall @ {top_k}: {recall:.2f}")
    print('='*30)

evaluate_precision_recall("Rabbit Town", top_k=5)

"""## Evaluation Collaborative Filtering

### Metric Visualization
"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np

def precision_recall_at_k(model, x_val, y_val, user_encoded_to_user, tour_encoded_to_tour, Reviews_df, k=10, rating_threshold=0.7):
    val_df = Reviews_df.iloc[len(x_train):].copy()
    val_df['USER_ENCODED'] = val_df['USER']
    val_df['TOUR_ENCODED'] = val_df['TOUR']
    val_df['RATING_NORMALIZED'] = y_val

    precision_scores = []
    recall_scores = []

    unique_users_val = val_df['USER_ENCODED'].unique()

    for user_enc in unique_users_val:
        true_items = val_df[(val_df['USER_ENCODED'] == user_enc) & (val_df['RATING_NORMALIZED'] >= rating_threshold)]['TOUR_ENCODED'].tolist()
        if len(true_items) == 0:
            continue

        # Semua tempat untuk prediksi rekomendasi user ini (baik yang sudah dikunjungi atau belum)
        all_tours = np.array(list(tour_encoded_to_tour.keys()))
        user_array = np.array([user_enc] * len(all_tours))
        user_tour_pairs = np.vstack([user_array, all_tours]).T

        # Prediksi rating semua tempat user ini
        preds = model.predict(user_tour_pairs, verbose=0).flatten()

        # Ambil top-k indeks tempat rekomendasi
        top_k_indices = preds.argsort()[-k:][::-1]
        recommended_items = all_tours[top_k_indices]

        # Hitung Precision dan Recall
        recommended_set = set(recommended_items)
        true_set = set(true_items)

        n_relevant_and_recommended = len(recommended_set.intersection(true_set))
        precision = n_relevant_and_recommended / k
        recall = n_relevant_and_recommended / len(true_set)

        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_precision, avg_recall

# Panggil fungsi evaluasi
precision, recall = precision_recall_at_k(model, x_val, y_val, user_encoded_to_user, tour_encoded_to_tour, Reviews_df, k=10, rating_threshold=0.7)

print(f"\nEvaluasi Model - Precision@10: {precision:.4f}")
print(f"Evaluasi Model - Recall@10   : {recall:.4f}")