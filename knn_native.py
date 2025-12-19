import math

class KNNNative:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []      # Data latih asli (optional, buat backup)
        self.X_train_norm = [] # Data latih YANG SUDAH DINORMALISASI
        self.y_train = []      # Label
        self.min_vals = []     # Menyimpan nilai minimum tiap kolom
        self.max_vals = []     # Menyimpan nilai maksimum tiap kolom

    # --- FUNGSI BANTUAN: Normalisasi Min-Max (Native) ---
    def _normalize_row(self, row):
        norm_row = []
        for i in range(len(row)):
            val = row[i]
            min_v = self.min_vals[i]
            max_v = self.max_vals[i]
            
            # Rumus: (x - min) / (max - min)
            # Cegah pembagian dengan nol jika max == min
            if max_v - min_v == 0:
                norm_val = 0.0
            else:
                norm_val = (val - min_v) / (max_v - min_v)
            norm_row.append(norm_val)
        return norm_row

    # --- 1. TRAINING (Simpan Data & Hitung Skala) ---
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_train_norm = [] # Reset

        # A. Cari Min dan Max untuk setiap kolom (Fitur)
        # Teknik Zip(*X) mengubah baris jadi kolom
        columns = list(zip(*X)) 
        self.min_vals = [min(col) for col in columns]
        self.max_vals = [max(col) for col in columns]

        # B. Normalisasi semua data training dan simpan
        for row in X:
            self.X_train_norm.append(self._normalize_row(row))

    # --- 2. HITUNG JARAK (Euclidean) ---
    def _euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    # --- 3. PREDIKSI ---
    def predict(self, X_test):
        predictions = []
        
        for row_test in X_test:
            # A. PENTING: Input User juga harus dinormalisasi
            # menggunakan Min/Max dari data TRAINING
            row_test_norm = self._normalize_row(row_test)

            # B. Hitung jarak ke semua data training (yang sudah normal)
            all_distances = []
            for i in range(len(self.X_train_norm)):
                # Bandingkan Input Normal vs Training Normal
                dist = self._euclidean_distance(row_test_norm, self.X_train_norm[i])
                all_distances.append((dist, self.y_train[i]))
            
            # C. Sorting jarak terdekat
            all_distances.sort(key=lambda x: x[0])
            
            # D. Ambil K tetangga
            neighbors = all_distances[:self.k]
            
            # E. Voting
            votes = {}
            for neighbor in neighbors:
                label = neighbor[1]
                votes[label] = votes.get(label, 0) + 1
            
            # F. Pemenang
            result = max(votes, key=votes.get)
            predictions.append(result)
            
        return predictions