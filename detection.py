import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms
from PIL import Image


# ------------------------------
# تعريف نموذج CNN (كما دربته سابقاً)
# ------------------------------
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
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # نفترض: 0 = Real، و 1 = Deepfake
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ------------------------------
# إعداد النموذج وتحميل الأوزان المُحفوظة
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device)
model_path = "deepfake_cnn_new_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # وضع الاختبار

# ------------------------------
# إعداد تحويل الصورة (Preprocessing)
# ------------------------------
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------------------
# إنشاء كاشف الوجوه باستخدام MTCNN
# ------------------------------
detector = MTCNN()

# ------------------------------
# إنشاء مجلد لحفظ الصور النهائية
# ------------------------------
save_folder = "processed_faces"
os.makedirs(save_folder, exist_ok=True)


# ------------------------------
# دالة لمعالجة الفيديو مع حفظ الوجوه المُستخرجة
# ------------------------------
def process_video(video_path, threshold=0.5, frame_step=30):
    """
    يعالج الفيديو:
      - يقرأ الإطارات باستخدام OpenCV.
      - يأخذ إطاراً واحداً فقط كل frame_step (مثلاً 30) إطار.
      - يكتشف الوجوه باستخدام MTCNN.
      - يحفظ صورة الوجه المستخرجة في مجلد.
      - يمرر الوجه عبر النموذج للحصول على احتمالية Deepfake.
      - يحسب متوسط الاحتمالية ثم يصدر القرار النهائي بناءً على العتبة.
    """
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []  # لتخزين الاحتمالات لكل إطار تمت معالجته
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # نهاية الفيديو

        frame_count += 1
        # معالجة إطار واحد فقط كل frame_step إطار
        if frame_count % frame_step != 0:
            continue

        # اكتشاف الوجوه في الإطار باستخدام MTCNN
        faces = detector.detect_faces(frame)
        if not faces:
            print(f"Frame {frame_count}: لم يتم اكتشاف وجه")
            continue

        # استخدام أول وجه تم اكتشافه (يمكن تعديل الكود لمعالجة جميع الوجوه)
        face_box = faces[0]['box']  # [x, y, width, height]
        x, y, w, h = face_box
        x, y = max(0, x), max(0, y)
        face = frame[y:y + h, x:x + w]

        if face.size == 0:
            continue

        # حفظ صورة الوجه المستخرجة
        face_save_path = os.path.join(save_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(face_save_path, face)

        # تحويل الوجه من BGR (OpenCV) إلى RGB (PIL) ومن ثم تطبيق Preprocessing
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        # تمرير الوجه عبر النموذج للحصول على التنبؤ
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            deepfake_prob = probabilities[0, 1].item()  # نفترض أن الفهرس 1 يمثل Deepfake

            print(f"Frame {frame_count}: احتمالية Deepfake = {deepfake_prob:.4f}")
            frame_predictions.append(deepfake_prob)

    cap.release()

    if not frame_predictions:
        print("لم يتم اكتشاف أي وجوه في الفيديو.")
        return None

    # حساب متوسط الاحتمالية لجميع الإطارات التي تمت معالجتها
    avg_deepfake_prob = np.mean(frame_predictions)
    print(f"\nمتوسط احتمالية Deepfake لجميع الوجوه: {avg_deepfake_prob:.4f}")

    decision = "Deepfake" if avg_deepfake_prob > threshold else "Real"
    print(f"القرار النهائي: الفيديو مصنف كـ {decision}")

    return avg_deepfake_prob, decision


# ------------------------------
# استخدام الدالة لمعالجة فيديو معين
# ------------------------------

video_path = r"C:\Users\admin\Desktop\yapa zeka\data\Deepfakes\630_623.mp4"  # عدل مسار الفيديو حسب الحاجة
process_video(video_path, threshold=0.5, frame_step=30)
