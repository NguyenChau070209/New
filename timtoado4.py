import argparse
import sys
import time
import threading

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize
from picamera2 import Picamera2

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

detection_results = []  # Danh sách chứa kết quả phát hiện
lock = threading.Lock()  # Khóa để bảo vệ truy cập đến danh sách

def save_results_to_file():
    """Hàm để ghi kết quả vào file."""
    while True:
        time.sleep(2)  # Ghi file mỗi 2 giây (hoặc khoảng thời gian bạn muốn)
        with lock:
            file_path = "/home/xuanv/myenv/tflite-custom-object-bookworm-main/toado.txt"
            with open(file_path, "w") as file:
                if not detection_results:
                    file.write("0\n")  # Không có phát hiện
                else:
                    file.write("1\n")  # Có phát hiện
                    for result in detection_results:
                        for detection in result.detections:
                            bbox = detection.bounding_box
                            x = bbox.origin_x
                            y = bbox.origin_y
                            w = bbox.width
                            h = bbox.height
                            file.write(f"{x}, {y}, {w}, {h}\n")  # Ghi tọa độ

def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera."""

    global detection_results  # Để truy cập danh sách từ hàm save_results_to_file

    # Các tham số trực quan hóa
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Tính FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        with lock:
            detection_results.append(result)  # Thêm kết quả vào danh sách
        COUNTER += 1

    # Khởi tạo mô hình phát hiện đối tượng
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results, score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    # Bắt đầu luồng ghi file
    threading.Thread(target=save_results_to_file, daemon=True).start()

    # Liên tục chụp hình từ camera và chạy phát hiện
    while True:
        im = picam2.capture_array()
        image = cv2.resize(im, (640, 480))
        image = cv2.flip(image, -1)

        # Chuyển đổi hình ảnh từ BGR sang RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Chạy phát hiện đối tượng
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Hiển thị FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_results:
            current_frame = visualize(current_frame, detection_results[-1])  # Sử dụng kết quả mới nhất
            detection_frame = current_frame

        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)

        # Dừng chương trình nếu phím ESC được nhấn.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='best.tflite')
    parser.add_argument(
        '--maxResults',
        help='Max number of detection results.',
        required=False,
        default=5)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of detection results.',
        required=False,
        type=float,
        default=0.7)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
