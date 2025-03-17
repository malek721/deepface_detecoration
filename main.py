import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Eğitilmiş modeli yükle
model = load_model('deepfake_detection_model.h5')

# Dış videonun yolu (Doğru video yolunu buraya girin)
video_path = r"path_to_external_video.mp4"

# OpenCV ile videoyu aç
cap = cv2.VideoCapture(video_path)

predictions = []  # Her kare için yapılan tahminleri saklamak üzere liste
frame_count = 0

# Örnekleme oranı: Hesaplama yükünü azaltmak için her 30. kareyi işle
sample_rate = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Örnekleme oranına göre kare seçimi (her 30. kareyi işleme al)
    if frame_count % sample_rate != 0:
        continue

    # Kareyi 256x256 boyutuna yeniden boyutlandır
    frame_resized = cv2.resize(frame, (256, 256))
    # Piksel değerlerini normalleştir (0 ile 1 arasında)
    frame_normalized = frame_resized.astype('float32') / 255.0
    # Model girişine uygun hale getirmek için boyutunu genişlet (1, 256, 256, 3)
    frame_input = np.expand_dims(frame_normalized, axis=0)

    # Model ile tahmin yap
    pred = model.predict(frame_input)
    predictions.append(pred[0][0])

# Videoyu serbest bırak
cap.release()

# Tüm karelerin tahminlerinin ortalamasını hesapla
if predictions:
    mean_pred = np.mean(predictions)
    print("Ortalama Tahmin:", mean_pred)
    # Ortalama 0.5'ten büyükse video deepfake olarak sınıflandırılır
    if mean_pred > 0.5:
        print("Video deepfake olarak sınıflandırıldı")
    else:
        print("Video gerçek olarak sınıflandırıldı")
else:
    print("Videodan kare alınamadı")
