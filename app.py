import streamlit as st
import pandas as pd
from knn_native import KNNNative 

# --- CONFIG ---
st.set_page_config(page_title="SPK Kamera", layout="centered")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/data_final.csv') 
    
    # --- LOGIKA PENGURUTAN (Agar Tampilan di Tabel Rapi) ---
    urutan_kasta = ["Entry Level", "Mid Range", "High End"]
    # Paksa kolom Label mengikuti urutan kasta ini
    df['Label'] = pd.Categorical(df['Label'], categories=urutan_kasta, ordered=True)
    # Sortir barisnya
    df = df.sort_values('Label')
    
    # Ambil Data untuk Training
    X = df.iloc[:, 0:4].values.tolist() 
    y = df.iloc[:, 4].astype(str).values.tolist() # Pastikan label jadi string kembali
    return X, y, df

try:
    X_train, y_train, df_show = load_data()
except FileNotFoundError:
    st.error("File CSV tidak ditemukan.")
    st.stop()

# --- SIDEBAR (INPUT) ---
st.sidebar.header("🔧 Panel Kontrol")
k_input = st.sidebar.slider("Nilai K (Tetangga)", 1, 15, 3, step=2)

st.sidebar.markdown("---")
st.sidebar.header("📸 Input Spesifikasi")

in_pixel = st.sidebar.number_input("Effective Pixels (MP)", 0.0, 50.0, 12.0)
in_zoom = st.sidebar.number_input("Optical Zoom (x)", 0.0, 50.0, 3.0)
in_weight = st.sidebar.number_input("Berat (gram)", 0.0, 2500.0, 300.0)
in_dim = st.sidebar.number_input("Dimensi Terbesar (mm)", 0.0, 300.0, 100.0)

# --- MAIN PAGE ---
st.title("🔍 Klasifikasi Segmen Kamera")
st.markdown("Sistem Pendukung Keputusan menggunakan **KNN (Native)**")

# --- TAMPIL SAMPEL DATA (SUDAH URUT) ---
if st.checkbox("Tampilkan Sampel Dataset"):
    st.write("Berikut sampel data (Diurutkan dari Entry Level -> High End):")
    
    display_cols = ['Model', 'Release date', 'Label', 'Effective pixels', 'Optical_Zoom', 'Weight (inc. batteries)']
    
    try:
        # Groupby akan otomatis mengikuti urutan Categorical yang sudah kita set di load_data
        sample_view = df_show[display_cols].groupby('Label', observed=False).apply(
            lambda x: x.sample(2) if len(x) > 2 else x
        ).reset_index(drop=True)
        
        st.dataframe(sample_view)
        st.caption("Catatan: Perhatikan 'Release date'. Kamera Entry Level baru bisa lebih canggih dari kamera High End lama.")
    except Exception as e:
        st.dataframe(df_show.head())

# --- PROSES UTAMA ---
if st.sidebar.button("Klasifikasi Sekarang"):
    model = KNNNative(k=k_input)
    model.fit(X_train, y_train)
    
    input_user = [[in_pixel, in_zoom, in_weight, in_dim]]
    hasil_prediksi = model.predict(input_user)
    label_final = hasil_prediksi[0] 
    
    # Output
    st.success(f"Hasil Klasifikasi: **{label_final}**")
    
    # Detail
    st.write("---")
    st.subheader("📝 Detail Analisis")
    st.write(f"**Fitur Input:** {input_user[0]}")
    st.caption("(Urutan: Mega Pixel, Optical Zoom, Berat, Dimensi)")
    st.write(f"Model mencari **{k_input} kamera** dengan spesifikasi paling mirip...")