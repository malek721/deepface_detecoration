import os
import shutil
from sklearn.model_selection import train_test_split

# تحديد المسارات للمجلدات الأصلية
real_input_folder = r"C:\Users\admin\Desktop\yapa zeka\original"
fake_input_folder = r"C:\Users\admin\Desktop\yapa zeka\fakeData"

# تحديد المسارات للمجلدات الناتجة (التقسيم)
base_dir = r"C:\Users\admin\Desktop\yapa zeka\dataset_split"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# إنشاء المجلدات إذا لم تكن موجودة
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# إنشاء المجلدات الفرعية داخل train, val, test
for folder in ['real', 'fake']:
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, folder), exist_ok=True)


def split_data(input_folder, label):
    # جمع جميع الصور
    all_images = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # تعديل الامتدادات حسب الحاجة
                all_images.append(os.path.join(root, file))

    # تحقق إذا كانت القائمة فارغة
    if not all_images:
        print(f"No images found in {input_folder}. Skipping...")
        return  # الخروج إذا لم توجد صور

    # تقسيم الصور إلى مجموعات تدريب و تحقق واختبار
    train_val_images, test_images = train_test_split(all_images, test_size=0.05, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=0.10, random_state=42)

    # نقل الملفات إلى المسارات المناسبة
    for image in train_images:
        shutil.copy(image, os.path.join(train_dir, label, os.path.basename(image)))

    for image in val_images:
        shutil.copy(image, os.path.join(val_dir, label, os.path.basename(image)))

    for image in test_images:
        shutil.copy(image, os.path.join(test_dir, label, os.path.basename(image)))


# تقسيم بيانات الصور الحقيقية والمزيفة
split_data(real_input_folder, 'real')
split_data(fake_input_folder, 'fake')

print("Data splitting completed.")

