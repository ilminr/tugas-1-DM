# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Memuat data
data = pd.read_csv('dataset tugas dm.csv')

# 1. Identifikasi dan Penanganan Missing Values
print("Missing values per column:\n", data.isnull().sum())
data.fillna(data.median(), inplace=True)

# 2. Normalisasi atau Standarisasi Data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-1]), columns=data.columns[:-1])
data_scaled['Outcome'] = data['Outcome']

# 3. Penanganan Outliers (Metode IQR)
Q1 = data_scaled.quantile(0.25)
Q3 = data_scaled.quantile(0.75)
IQR = Q3 - Q1
data_cleaned = data_scaled[~((data_scaled < (Q1 - 1.5 * IQR)) | (data_scaled > (Q3 + 1.5 * IQR))).any(axis=1)]

# 4. Feature Engineering
data_cleaned['AgeGroup'] = pd.cut(data_cleaned['Age'], bins=[20, 30, 40, 50, 60, 100], labels=['20s', '30s', '40s', '50s', '60+'])

# 5. Encoding Variabel Kategorikal
data_encoded = pd.get_dummies(data_cleaned, columns=['AgeGroup'], drop_first=True)

# 6. Sampling Data (SMOTE untuk Data Tidak Seimbang)
X = data_encoded.drop('Outcome', axis=1)
y = data_encoded['Outcome']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Distribusi setelah SMOTE:\n", y_resampled.value_counts())

# 7. Splitting Data (Pelatihan & Pengujian)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Ukuran Data Training:", X_train.shape, "Data Testing:", X_test.shape)
