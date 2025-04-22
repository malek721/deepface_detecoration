import os
import cv2
from mtcnn import MTCNN

# تحديد المسار الرئيسي الذي يحتوي على المجلدات الفرعية
main_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_original"  # قم بتعديل هذا إلى مسار المجلد لديك

# إنشاء كاشف الوجوه
detector = MTCNN()

# التصفح داخل كل المجلدات الفرعية والصور بداخلها
for root, dirs, files in os.walk(main_folder):
    for filename in files:
        # إنشاء المسار الكامل للصورة
        image_path = os.path.join(root, filename)

        try:
            # تحميل الصورة
            image = cv2.imread(image_path)

            # التأكد من أن الصورة تم تحميلها بنجاح
            if image is None:
                print(f"⚠️ فشل في تحميل {filename}، قد تكون الصورة تالفة.")
                os.remove(image_path)  # حذف الصورة التالفة
                continue

            # اكتشاف الوجوه في الصورة
            faces = detector.detect_faces(image)

            # إذا لم يتم العثور على وجه، احذف الصورة
            if not faces:
                print(f"❌ حذف {filename} - لا يحتوي على وجه.")
                os.remove(image_path)
            else:
                print(f"✅ تم الاحتفاظ بـ {filename} - يحتوي على وجه.")

        except Exception as e:
            print(f"⚠️ خطأ أثناء معالجة {filename}: {e}")
