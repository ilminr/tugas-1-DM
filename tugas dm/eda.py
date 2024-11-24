# Import libraries untuk visualisasi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Memuat data
data = pd.read_csv('dataset tugas dm.csv')

# 1. Histogram Distribusi Data
def plot_histograms(data):
    data.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.tight_layout()
    plt.show()  # Menambahkan plt.show() di sini

# 2. Korelasi Antar Variabel (Heatmap)
def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap Korelasi Antar Variabel')
    plt.show()  # Menambahkan plt.show() di sini

# 3. Distribusi Target (Outcome)
def plot_outcome_distribution(data):
    sns.countplot(x='Outcome', data=data, palette='Set2')
    plt.title('Distribusi Outcome')
    plt.xlabel('Outcome (0: Tidak Diabetes, 1: Diabetes)')
    plt.ylabel('Jumlah')
    plt.show()  # Menambahkan plt.show() di sini

# 4. Hubungan BMI dan Glucose terhadap Outcome
def plot_bmi_glucose_relationship(data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=data, palette='coolwarm')
    plt.title('Hubungan BMI dan Glucose terhadap Outcome')
    plt.xlabel('BMI')
    plt.ylabel('Glucose')
    plt.show()  # Menambahkan plt.show() di sini

# 5. Pairplot untuk Melihat Hubungan Antar Variabel
def plot_pairplot(data):
    # Membersihkan data dari nilai NaN sebelum pairplot
    data_cleaned = data.dropna(subset=data.select_dtypes(include=['float64', 'int64']).columns)
    numerical_data = data_cleaned.select_dtypes(include=['float64', 'int64'])
    sns.pairplot(numerical_data, hue='Outcome', diag_kind='kde', palette='husl')
    plt.show()  # Menambahkan plt.show() di sini

# Menjalankan semua visualisasi
plot_histograms(data)
plot_correlation_heatmap(data)
plot_outcome_distribution(data)
plot_bmi_glucose_relationship(data)
plot_pairplot(data)
