import os
import cv2

# المجلد الرئيسي للبيانات الأصلية
input_root = r"C:\Users\admin\Desktop\yapa zeka\fake\processed_FaceSwap\012_026"

# المجلد الذي سيتم حفظ الصور المحسنة فيه
output_root = r"C:\Users\admin\Desktop\yapa zeka\fake_enhanced"
os.makedirs(output_root, exist_ok=True)

# أنواع الصور المقبولة
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# المرور على كل المجلدات الفرعية
for subdir, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(image_extensions):
            input_path = os.path.join(subdir, file)

            # تحديد المسار النسبي (لنحافظ على نفس الهيكل)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)

            # إنشاء المجلد إن لم يكن موجودًا
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # تحميل الصورة
            img = cv2.imread(input_path)
            if img is None:
                continue

            # --- تحسين الصورة ---
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            # تحسين التباين باستخدام LAB
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            lab_eq = cv2.merge((l_eq, a, b))
            enhanced_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

            # حفظ الصورة الجديدة
            cv2.imwrite(output_path, enhanced_img)

        print("تم تحسين جميع الصور بنجاح ✅")