# import os
# import shutil
# import random
# import cv2
# import numpy as np
# from imgaug import augmenters as iaa
#
# # مسارات المجلدات الأصلية
# real_input_folder = r"C:\Users\admin\Desktop\yapa zeka\croped_original"
# fake_input_folder = r"C:\Users\admin\Desktop\yapa zeka\croped_fakeData"
#
# # مسار حفظ البيانات الجديدة
# output_dataset = r"C:\Users\admin\Desktop\yapa zeka\dataset"
#
# # تقسيم البيانات بالنسبة المئوية
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15
#
# # إنشاء مجلدات التقسيم
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(output_dataset, split, "real"), exist_ok=True)
#     os.makedirs(os.path.join(output_dataset, split, "fake"), exist_ok=True)
#
#
# # دالة لاستخراج جميع الصور من المجلدات الفرعية
# def collect_images_from_folder(root_folder):
#     all_images = []
#     for root, _, files in os.walk(root_folder):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 all_images.append(os.path.join(root, file))
#     return all_images
#
#
# # جمع الصور الأصلية والمزيفة
# real_images = collect_images_from_folder(real_input_folder)
# fake_images = collect_images_from_folder(fake_input_folder)
#
# # خلط البيانات
# random.shuffle(real_images)
# random.shuffle(fake_images)
#
#
# # دالة لحفظ الصور بعد التحسين
# def save_augmented_image(image, save_path, prefix, count):
#     new_filename = f"{prefix}_{count:04d}.jpg"
#     cv2.imwrite(os.path.join(save_path, new_filename), image)
#
#
# # دالة لتكثير البيانات الحقيقية
# def augment_images(image_paths, target_count):
#     augmented_images = []
#     augmenter = iaa.Sequential([
#         iaa.Fliplr(0.5),  # انعكاس أفقي بنسبة 50%
#         iaa.Affine(rotate=(-15, 15)),  # تدوير بين -15 و +15 درجة
#         iaa.GammaContrast((0.8, 1.2)),  # تغيير التباين
#         iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))  # إضافة ضوضاء خفيفة
#     ])
#
#     while len(augmented_images) < target_count:
#         for img_path in image_paths:
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue
#
#             # تطبيق تحويلات عشوائية
#             augmented_img = augmenter.augment_image(img)
#             augmented_images.append(augmented_img)
#
#             if len(augmented_images) >= target_count:
#                 break
#
#     return augmented_images
#
#
# # دالة لتقسيم البيانات
# def split_and_move(images, category, augment=False):
#     total = len(images)
#     train_count = int(total * train_ratio)
#     val_count = int(total * val_ratio)
#     test_count = total - (train_count + val_count)
#
#     train_images = images[:train_count]
#     val_images = images[train_count:train_count + val_count]
#     test_images = images[train_count + val_count:]
#
#     # إذا كانت البيانات حقيقية ونحتاج إلى مضاعفتها
#     if augment and len(images) < len(fake_images):
#         required_count = len(fake_images) - len(images)
#         extra_images = augment_images(train_images, required_count)
#
#         train_images.extend(extra_images)  # إضافة الصور الجديدة إلى بيانات التدريب
#
#     # حفظ الصور في المجلدات المناسبة
#     def move_images(images_list, split_name):
#         for idx, img in enumerate(images_list):
#             if isinstance(img, str):  # إذا كانت صورة عادية
#                 img_path = img
#                 dst = os.path.join(output_dataset, split_name, category, os.path.basename(img_path))
#                 shutil.copy(img_path, dst)
#             else:  # إذا كانت صورة مكثرة
#                 save_augmented_image(img, os.path.join(output_dataset, split_name, category), category, idx)
#
#     move_images(train_images, "train")
#     move_images(val_images, "val")
#     move_images(test_images, "test")
#
#
# # تطبيق التقسيم وتكثير الصور الأصلية
# split_and_move(real_images, "real", augment=True)
# split_and_move(fake_images, "fake")
#
# print("✅ تم تجهيز البيانات وتكثير الصور الحقيقية بنجاح!")

import os
import shutil
import random
import cv2
#
# # مسارات المجلدات الأصلية
# real_input_folder = r"C:\Users\admin\Desktop\yapa zeka\original"
# fake_input_folder = r"C:\Users\admin\Desktop\yapa zeka\fakeData"
#
# # مسار حفظ البيانات الجديدة
# output_dataset = r"C:\Users\admin\Desktop\yapa zeka\dataset"
#
# # تقسيم البيانات بالنسبة المئوية
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15
#
# # إنشاء مجلدات التقسيم
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(output_dataset, split, "real"), exist_ok=True)
#     os.makedirs(os.path.join(output_dataset, split, "fake"), exist_ok=True)
#
#
# # دالة لاستخراج جميع الصور من المجلدات الفرعية
# def collect_images_from_folder(root_folder):
#     all_images = []
#     for root, _, files in os.walk(root_folder):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 all_images.append(os.path.join(root, file))
#     return all_images
#
#
# # جمع الصور الأصلية والمزيفة
# real_images = collect_images_from_folder(real_input_folder)
# fake_images = collect_images_from_folder(fake_input_folder)

# خلط البيانات
# random.shuffle(real_images)
# random.shuffle(fake_images)
#
#
# # دالة لتقسيم البيانات فقط دون تكثيرها
# def split_and_move(images, category):
#     total = len(images)
#     train_count = int(total * train_ratio)
#     val_count = int(total * val_ratio)
#     test_count = total - (train_count + val_count)
#
#     train_images = images[:train_count]
#     val_images = images[train_count:train_count + val_count]
#     test_images = images[train_count + val_count:]
#
#     # حفظ الصور في المجلدات المناسبة
#     def move_images(images_list, split_name):
#         for img_path in images_list:
#             dst = os.path.join(output_dataset, split_name, category, os.path.basename(img_path))
#             shutil.copy(img_path, dst)
#
#     move_images(train_images, "train")
#     move_images(val_images, "val")
#     move_images(test_images, "test")
#
#
# # تطبيق التقسيم دون تكثير البيانات
# split_and_move(real_images, "real")
# split_and_move(fake_images, "fake")
#
# print("✅ تم تجهيز البيانات بدون تكثير بنجاح!")


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

