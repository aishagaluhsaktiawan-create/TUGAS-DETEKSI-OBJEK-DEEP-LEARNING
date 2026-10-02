import os, shutil
from ultralytics import YOLO
import cv2

MODEL_PATH = "yolov8n.pt"
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"



# DETEKSI GAMBAR

def detect_objects():
    model = YOLO(MODEL_PATH)

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    for image_name in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, image_name)
        output_path = os.path.join(OUTPUT_FOLDER, image_name)

        if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        results = model.predict(input_path, conf=0.5, save=False)

        for result in results:
            anotated_image = result.plot()
            cv2.imwrite(output_path, anotated_image)
            print(f"hasil deteksi di simpan di simpan ke: {output_path}")



# DETEKSI KAMERA

def detect_camera():
    # Memanggil model YOLO
    model = YOLO(MODEL_PATH)

    # Membuka kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    print("Kamera aktif.")
    print("Arahkan objek ke kamera.")
    print("klik X untuk keluar.")

    while True:

        # Membaca gambar dari kamera
        ret, frame = cap.read()

        if not ret:
            print("Frame tidak terbaca.")
            break

        # Deteksi objek menggunakan YOLO
        results = model.predict(
            frame,
            conf=0.25,
            verbose=False
        )

        # Membuat bounding box dan nama objek
        annotated_frame = results[0].plot()

        # Menampilkan hasil deteksi
        cv2.imshow(
            "YOLO Object Detection",
            annotated_frame
        )

        # untuk keluar
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Kamera ditutup.")
            break

        # X untuk keluar
        if cv2.getWindowProperty(
            "YOLO Object Detection",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            print("Kamera ditutup.")
            break

    cap.release()
    cv2.destroyAllWindows()



# PROGRAM UTAMA

if __name__ == "__main__":
    detect_objects()
    detect_camera()