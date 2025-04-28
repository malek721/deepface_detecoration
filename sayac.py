import os

#real  84378
#fake  283647
# مسار المجلد الأساسي
root_folder = r"C:\Users\admin\Desktop\yapa zeka\fake"

# الامتدادات التي تعتبر صوراً
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

# عداد الصور
image_count = 0

# المشي داخل كل المجلدات الفرعية والملفات
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        # التأكد من أن الملف صورة حسب الامتداد
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_count += 1

print(f"عدد الصور هو: {image_count}")

