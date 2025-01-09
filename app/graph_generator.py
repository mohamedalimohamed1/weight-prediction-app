import matplotlib.pyplot as plt

# Epoch değerleri
epochs = list(range(1, 51))

# Eğitim ve test doğruluk değerleri
egitim_dogruluk = [0.9853] * 50
test_dogruluk = [0.9624] * 50
guven_skoru = [96.24] * 50

# Scatter plot çizimi
plt.figure(figsize=(12, 6))

plt.scatter(epochs, egitim_dogruluk, color='blue', label='Eğitim Doğruluk (R²)', marker='o')
plt.scatter(epochs, test_dogruluk, color='green', label='Test Doğruluk (R²)', marker='x')
plt.scatter(epochs, guven_skoru, color='purple', label='Güven Skoru (%)', marker='s')

# Grafik başlıkları ve etiketleri
plt.title('Model Performansı - Scatter Plot')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.grid(True)

# Grafiği göster
plt.show()
