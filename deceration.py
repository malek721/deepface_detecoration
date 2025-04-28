import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
from deepface import DeepFace
from torch import nn

# ================== إعدادات ==================
video_path = r"C:\Users\admin\Desktop\yapa zeka\data\Deepfakes\109_109.mp4"
output_dir = "output_faces"
frame_interval = 30
model_path = "deepfake_cnn_new_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== تعريف نموذج MyCNN (ResNet-like) ==================
class MyCNN(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_rate=0.3):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.drop = nn.Dropout(drop_rate)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    def __init__(self, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, drop_rate=drop_rate)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_rate):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [MyCNN.ResidualBlock(in_channels, out_channels, stride, downsample, drop_rate)]
        for _ in range(1, blocks):
            layers.append(MyCNN.ResidualBlock(out_channels, out_channels, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# تحميل النموذج
model = MyCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# تجهيز مجلد الحفظ
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# استخراج الوجوه من الفيديو
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # تحويل من BGR إلى RGB
            result = DeepFace.extract_faces(img_path=rgb_frame, detector_backend='opencv', enforce_detection=False)
            for idx, face in enumerate(result):
                face_img = face['face']
                face_path = os.path.join(output_dir, f"face_{saved_count}.jpg")
                Image.fromarray(face_img).save(face_path)
                saved_count += 1
        except Exception as e:
            print(f"خطأ في استخراج الوجه من الفريم {frame_count}: {e}")

    frame_count += 1

cap.release()
print(f"تم حفظ {saved_count} وجهًا في {output_dir}")

# تحليل الصور بالنموذج
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

total_fake_prob = 0
image_count = 0

for img_file in os.listdir(output_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(output_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()  # احتمال "Fake" (الفئة رقم 1)
            total_fake_prob += prob
            image_count += 1

average_fake = total_fake_prob / image_count if image_count > 0 else 0
print(f"\n📊 متوسط احتمالية التزييف (Fake): {average_fake:.4f}")
