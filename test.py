import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mtcnn import MTCNN

# MyCNN modeli tanımı
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),  # Beklenen giriş: 128 kanal * 16x16
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme
model_path = "deepfake_cnn_model.pth"
model = MyCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Yüz algılayıcılar: önce Haar Cascade, sonra MTCNN filtreleme
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn = MTCNN()

# Görüntü ön işlemleri (128x128) -- model eğitim boyutuna uygun
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Kaydedilecek dosyalar için
processed_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_model_inputs"

def analyze_video(video_path, sample_rate=5):
    """
    1) Videodan her sample_rate kareyi alır.
    2) Haar Cascade ile yüz algılar, bulduğu yüze crop ve 128x128 boyut verip kaydeder.
    3) Kaydedilen tüm görüntülerde MTCNN ile yüz doğrulaması yapar, yüz yoksa siler.
    4) Kalan görüntüleri modele besleyip tahmin yapar, ortalama deepfake olasılığı döner.
    """
    # 1. Adım: Face extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(processed_folder, video_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x,y,w,h = faces[0]
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (128,128))
                fname = f"face_{saved:04d}.jpg"
                cv2.imwrite(os.path.join(save_dir, fname), face_resized)
                saved += 1
        frame_num += 1
    cap.release()
    print(f"Extracted and saved {saved} face images to {save_dir}")

    # 2. Adım: MTCNN ile silme
    for img_name in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_name)
        img = cv2.imread(img_path)
        if img is None or len(mtcnn.detect_faces(img)) == 0:
            os.remove(img_path)
    remaining = len(os.listdir(save_dir))
    print(f"Remaining after MTCNN filter: {remaining} images")

    # 3. Adım: Model tahminleri
    preds = []
    for img_name in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            p = torch.softmax(out, dim=1)[0,1].item()
            preds.append(p)

    if preds:
        mean_p = sum(preds)/len(preds)
        print(f"Mean Deepfake Probability: {mean_p:.4f}")
        print("Video classified as", "Deepfake" if mean_p>0.5 else "Real")
    else:
        print("No valid face inputs for model.")


if __name__ == "__main__":
    video_file = r"C:\Users\admin\Desktop\yapa zeka\data\Deepfakes\108_052.mp4"
    analyze_video(video_file)