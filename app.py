import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Clustering.csv')
    return data

# Halaman 1: About
def about_page():
    st.title("About Aplikasi Clustering")
    st.write("""
        Aplikasi ini menggunakan algoritma K-Means untuk melakukan clustering pada dataset yang berisi informasi demografis.
        Dataset ini mencakup berbagai atribut seperti ID, jenis kelamin, status pernikahan, usia, pendidikan, pendapatan, 
        pekerjaan, dan ukuran pemukiman. Aplikasi ini bertujuan untuk membantu pengguna memahami pola dalam data 
        dan bagaimana clustering dapat digunakan untuk analisis data.
    """)

    st.subheader("Apa itu Clustering?")
    st.write("""
        Clustering adalah teknik dalam analisis data yang digunakan untuk mengelompokkan sekumpulan objek ke dalam 
        kelompok (cluster) yang memiliki kesamaan. Objek dalam satu cluster lebih mirip satu sama lain dibandingkan 
        dengan objek di cluster lain. Clustering sering digunakan dalam berbagai bidang, termasuk pemasaran, 
        pengenalan pola, dan pengolahan citra.
    """)

    st.subheader("Algoritma K-Means")
    st.write("""
        K-Means adalah salah satu algoritma clustering yang paling populer. Algoritma ini bekerja dengan cara:
        
        1. **Inisialisasi**: Memilih K titik pusat (centroid) secara acak dari dataset.
        2. **Penugasan**: Setiap titik data ditugaskan ke centroid terdekat berdasarkan jarak Euclidean.
        3. **Pembaruan**: Menghitung ulang posisi centroid dengan mengambil rata-rata dari semua titik data yang 
           ditugaskan ke centroid tersebut.
        4. **Iterasi**: Mengulangi langkah 2 dan 3 hingga posisi centroid tidak berubah atau mencapai batas iterasi maksimum.

        K-Means berfungsi dengan baik pada dataset yang memiliki bentuk bulat dan ukuran cluster yang serupa.
    """)

    st.subheader("Fitur Aplikasi")
    st.write("""
        Aplikasi ini memiliki beberapa fitur utama:
        
        - **Halaman Dataset**: Menampilkan dataset yang digunakan dan memungkinkan pengguna untuk melihat data secara langsung.
        - **Visualisasi Clustering**: Menyediakan visualisasi yang menunjukkan bagaimana data dikelompokkan berdasarkan 
          usia dan pendapatan.
        - **Jalankan Algoritma**: Memungkinkan pengguna untuk memilih jumlah cluster yang diinginkan dan melihat 
          hasil clustering serta centroid dari setiap cluster.
    """)

    st.subheader("Mengapa Menggunakan Aplikasi Ini?")
    st.write("""
        Aplikasi ini dirancang untuk membantu pengguna memahami bagaimana algoritma K-Means bekerja dan bagaimana 
        data dapat dikelompokkan. Dengan menggunakan aplikasi ini, pengguna dapat:
        
        - Mempelajari konsep dasar clustering dan algoritma K-Means.
        - Melihat bagaimana data dikelompokkan berdasarkan fitur tertentu.
        - Mengubah jumlah cluster dan melihat bagaimana hasilnya berubah.
    """)

# Halaman 2: Dataset dan Chart
def dataset_page(data):
    st.title("Dataset dan Visualisasi")
    st.write("Berikut adalah dataset yang digunakan:")
    st.dataframe(data)

    # Visualisasi
    st.subheader("Visualisasi Clustering")
    if st.checkbox("Tampilkan Visualisasi Clustering"):
        # Preprocessing
        features = data[['Age', 'Income']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_features)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Age', y='Income', hue='Cluster', palette='viridis', s=100)
        plt.title("Clustering K-Means berdasarkan Usia dan Pendapatan")
        plt.xlabel("Usia")
        plt.ylabel("Pendapatan")
        st.pyplot(plt)

# Halaman 3: Jalankan Algoritma
def run_algorithm(data):
    st.title("Jalankan Algoritma Clustering")
    st.write("Pilih jumlah cluster yang diinginkan:")
    n_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)

    # Preprocessing
    features = data[['Age', 'Income']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_features)

    st.write("Hasil Clustering:")
    st.dataframe(data)

    # Penjelasan tentang K-Means
    st.subheader("Penjelasan tentang Algoritma K-Means")
    st.write("""
        Algoritma K-Means adalah metode clustering yang membagi data ke dalam K kelompok berdasarkan kesamaan fitur.
        Berikut adalah langkah-langkah dasar dari algoritma K-Means:
        
        1. **Inisialisasi**: Pilih K titik pusat (centroid) secara acak dari dataset.
        2. **Penugasan**: Setiap titik data ditugaskan ke centroid terdekat berdasarkan jarak Euclidean.
        3. **Pembaruan**: Hitung ulang posisi centroid dengan mengambil rata-rata dari semua titik data yang ditugaskan ke centroid tersebut.
        4. **Iterasi**: Ulangi langkah 2 dan 3 hingga posisi centroid tidak berubah atau mencapai batas iterasi maksimum.

        K-Means berfungsi dengan baik pada dataset yang memiliki bentuk bulat dan ukuran cluster yang serupa.
    """)

    # Visualisasi Clustering
    st.subheader("Visualisasi Hasil Clustering")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Age', y='Income', hue='Cluster', palette='viridis', s=100)
    plt.title(f"Clustering K-Means dengan {n_clusters} Cluster")
    plt.xlabel("Usia")
    plt.ylabel("Pendapatan")
    st.pyplot(plt)

    # Menampilkan centroid
    st.subheader("Centroid dari Cluster")
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Age', 'Income'])
    st.write(centroids)

# Main function
def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["About", "Dataset", "Run Clustering"])

    data = load_data()

    if page == "About":
        about_page()
    elif page == "Dataset":
        dataset_page(data)
    elif page == "Run Clustering":
        run_algorithm(data)

if __name__ == "__main__":
    main()