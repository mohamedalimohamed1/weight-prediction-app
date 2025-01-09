import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

# Klasörleri oluştur
os.makedirs("./visualization/svm_graphs", exist_ok=True)
os.makedirs("./models/svm_model", exist_ok=True)

# 1. Adım: Dosyayı yükleme ve okuma
print("Veri yükleniyor...")
data_path = "./dataset/weight_change_dataset.csv"
data = pd.read_csv(data_path)
print("Veri başarıyla yüklendi.")

# 2. Adım: 'Participant ID' sütununu çıkarma (sadece takip amaçlı)
data = data.drop(columns=['Participant ID'])

# 3. Adım: Özellikler ve hedef değişken ayrımı
print("Özellikler ve hedef değişken ayrılıyor...")
X = data.drop("Final Weight (lbs)", axis=1)
y = data["Final Weight (lbs)"]

# 4. Adım: Kategorik ve sayısal sütunları belirleme
categorical_features = ["Gender", "Physical Activity Level", "Sleep Quality"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# 5. Adım: Ön işleme ve Pipeline oluşturma
print("Veri ön işleme başlıyor...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

# Pipeline: Ön işleme ve SVR modelini birleştir
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", SVR())
])

# 6. Adım: Veriyi bölme (80% eğitim/val ve 20% test)
print("Veri eğitim ve test setlerine bölünüyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Veri başarıyla bölündü.")

# Veri bölünmesi grafiği oluşturma
print("Veri bölünmesi grafiği oluşturuluyor...")
plt.figure(figsize=(8, 6))
plt.title("Veri Bölünmesi", fontsize=14)
plt.pie([len(X_train), len(X_test)], labels=["Eğitim/Val", "Test"], autopct="%1.1f%%", colors=["skyblue", "orange"])
plt.savefig("./visualization/veri_bolunmesi.png")
plt.close()
print("Veri bölünmesi grafiği kaydedildi.")

# 7. Adım: Model eğitimi için GridSearchCV ile hiperparametre optimizasyonu
print("GridSearchCV ile en iyi hiperparametreler bulunuyor...")
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [0.001, 0.01, 0.1, 1],
    'model__kernel': ['rbf']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

# En iyi modeli seçme ve Pipeline olarak kaydetme
best_pipeline = grid_search.best_estimator_
print(f"En iyi hiperparametreler: {grid_search.best_params_}")
model_path = "./models/svm_model_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
print(f"Eğitilmiş model pipeline '{model_path}' yoluna kaydedildi.")

# 8. Adım: Eğitim sırasında epoklar ve performans ölçümleri
print("Eğitim başlıyor...")
epochs = 50
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
confidence_scores = []

for epoch in range(1, epochs + 1):
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    confidence = test_r2 * 100  # Confidence score as a percentage

    train_loss.append(train_mse)
    test_loss.append(test_mse)
    train_accuracy.append(train_r2)
    test_accuracy.append(test_r2)
    confidence_scores.append(confidence)

    print(f"Epoch {epoch}/{epochs} - Eğitim Kayıp: {train_mse:.4f}, Test Kayıp: {test_mse:.4f}, Eğitim Doğruluk: {train_r2:.4f}, Test Doğruluk: {test_r2:.4f}, Güven Skoru: {confidence:.2f}%")

# 9. Adım: Eğitim sonuçları grafiği oluşturma
print("Eğitim sonuçları grafiği oluşturuluyor...")
plt.figure(figsize=(12, 8))
plt.plot(range(1, epochs + 1), train_loss, label="Eğitim Kayıp (MSE)", color="blue", linewidth=2)
plt.plot(range(1, epochs + 1), train_accuracy, label="Eğitim Doğruluk (R²)", color="green", linewidth=2)
plt.title("Eğitim Sonuçları", fontsize=16)
plt.xlabel("Epoklar", fontsize=14)
plt.ylabel("Değer", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("./visualization/egitim_sonuclari_grafik.png")
plt.close()
print("Eğitim sonuçları grafiği kaydedildi.")

# 10. Adım: Test sonuçları grafiği oluşturma
print("Test sonuçları grafiği oluşturuluyor...")
plt.figure(figsize=(12, 8))
plt.plot(range(1, epochs + 1), test_loss, label="Test Kayıp (MSE)", color="orange", linewidth=2)
plt.plot(range(1, epochs + 1), test_accuracy, label="Test Doğruluk (R²)", color="red", linewidth=2)
plt.plot(range(1, epochs + 1), confidence_scores, label="Model Güven Skoru (%)", color="purple", linewidth=2, linestyle='--')
plt.title("Test Sonuçları", fontsize=16)
plt.xlabel("Epoklar", fontsize=14)
plt.ylabel("Değer", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("./visualization/test_sonuclari_grafik.png")
plt.close()
print("Test sonuçları grafiği kaydedildi.")

print("Model başarıyla eğitildi ve tüm görseller './visualization' klasörüne kaydedildi.")
