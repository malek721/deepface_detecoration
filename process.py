# import cv2
# import os
# import numpy as np
#
# # تحميل نموذج Haar Cascade لتحديد الوجوه
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#
# def detect_and_crop_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # الكشف عن الوجوه، يمكن تعديل المعاملات حسب الحاجة
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     if len(faces) == 0:
#         # إذا لم يتم الكشف عن وجه، يمكن إرجاع الفريم الأصلي أو التعامل معه بطريقة أخرى
#         return frame
#
#     # اختيار أول وجه (يمكن اختيار الأكبر أو بناءً على معايير أخرى)
#     (x, y, w, h) = faces[0]
#     cropped_face = frame[y:y + h, x:x + w]
#     return cropped_face
#
#
# def resize_with_padding(image, target_size):
#     h, w = image.shape[:2]
#     if h > w:
#         new_h = target_size
#         new_w = int(w * target_size / h)
#     else:
#         new_w = target_size
#         new_h = int(h * target_size / w)
#     resized_img = cv2.resize(image, (new_w, new_h))
#     top = (target_size - new_h) // 2
#     bottom = target_size - new_h - top
#     left = (target_size - new_w) // 2
#     right = target_size - new_w - left
#     final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#     return final_img
#
#
# def video_to_frames_and_process(video_path, output_folder, frame_rate=1, target_size=256):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     video_output_folder = os.path.join(output_folder, video_name)
#     if not os.path.exists(video_output_folder):
#         os.makedirs(video_output_folder)
#     frame_count = 0
#     saved_frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # استخراج الفريمات بمعدل معين
#         if frame_count % frame_rate == 0:
#             # تطبيق تحديد الوجه واقتصاصه
#             processed_frame = detect_and_crop_face(frame)
#             # تعديل حجم الصورة مع إضافة حواف سوداء
#             resized_frame = resize_with_padding(processed_frame, target_size)
#             frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.jpg")
#             cv2.imwrite(frame_filename, resized_frame)
#             saved_frame_count += 1
#         frame_count += 1
#         if frame_count % 1000 == 0:
#             print(f"Processed {frame_count}/{total_frames} frames of video {video_name}")
#     cap.release()
#     print(f"Frames extracted, processed, and saved to {video_output_folder}")
#
#
# def process_multiple_videos(videos_folder, output_folder, frame_rate=1, target_size=256):
#     video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
#     for video_file in video_files:
#         video_path = os.path.join(videos_folder, video_file)
#         video_to_frames_and_process(video_path, output_folder, frame_rate, target_size)
#         print(f"Finished processing {video_file}")
#
#
# # تعديل المسارات والمعاملات حسب الحاجة
# videos_folder = r"C:\Users\admin\Desktop\yapa zeka\data\fake"
# output_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_fake"
# frame_rate = 5  # استخراج فريم واحد كل 5 فريمات
# target_size = 256  # حجم الصورة النهائية
#
# process_multiple_videos(videos_folder, output_folder, frame_rate, target_size)


import cv2
import os
import numpy as np


def crop_center(img, target_size=256):
    """ يقتص أكبر مربع من الصورة ثم يغير حجمه إلى الحجم المطلوب """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    return resized_img


def video_to_frames(video_path, output_folder, frame_rate=5, target_size=256):
    """ استخراج الفريمات من الفيديو مع الاقتصاص من المنتصف """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # استخراج صورة كل 5 فريمات
        if frame_count % frame_rate == 0:
            processed_frame = crop_center(frame, target_size)
            frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, processed_frame)
            saved_frame_count += 1

        frame_count += 1

        # طباعة تقدم المعالجة
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count}/{total_frames} frames of video {video_name}")

    cap.release()
    print(f"Frames extracted and saved to {video_output_folder}")


def process_multiple_videos(videos_folder, output_folder, frame_rate=5, target_size=256):
    """ معالجة عدة فيديوهات داخل مجلد """
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_to_frames(video_path, output_folder, frame_rate, target_size)
        print(f"Finished processing {video_file}")


# تعديل المسارات والمعاملات حسب الحاجة
videos_folder = r"C:\Users\admin\Desktop\yapa zeka\data\original"
output_folder = r"C:\Users\admin\Desktop\yapa zeka\croped_original"
frame_rate = 1 # استخراج صورة كل 5 فريمات
target_size = 256  # حجم الصورة النهائي

process_multiple_videos(videos_folder, output_folder, frame_rate, target_size)
