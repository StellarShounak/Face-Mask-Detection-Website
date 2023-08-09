from keras.models import load_model
import cv2
import time
import mediapipe as mp
import logging

logging.basicConfig(filename='face_mask_detection.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Loading the face mask detection model")
model = load_model('Face_mask_model.h5')

logging.info("Starting video capture")
cap = cv2.VideoCapture(0)

mpFaceDetect = mp.solutions.face_detection
Draw = mp.solutions.drawing_utils
facedetection = mpFaceDetect.FaceDetection()


def detect_mask(img_arr):
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    y_prediction = model.predict(img_arr)

    return y_prediction[0][0]


def draw_label(image, text, pos, bg_color, text_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, cv2.FILLED)

    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(image, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 1, cv2.LINE_AA)


def generate_frames():
    prev_time = 0
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            logging.error("Failed to capture frame from camera.")
            break
        else:
            results = facedetection.process(frame)
            print(results)

            # img = frame

            if results.detections:
                for id, detection in enumerate(results.detections):
                    # print(id, detection)
                    # Draw.draw_detection(frame, detection)
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ch = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                    img = frame[int(bboxC.ymin * ih): int(bboxC.ymin * ih) + int(bboxC.height * ih),
                          int(bboxC.xmin * iw): int(bboxC.xmin * iw) + int(bboxC.width * iw), :]

            img = cv2.resize(img, (100, 100))

            y_pred = detect_mask(img)

            if y_pred >= 0.5:
                draw_label(frame, "MASK", (30, 30), (0, 255, 0), (0, 255, 0))
            else:
                draw_label(frame, "NO MASK", (30, 30), (0, 0, 255), (0, 0, 255))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame, f'FPS:{int(fps)}', (500, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


if __name__ == "__main__":
    # Start the video capture
    logging.info("Starting video capture")
    cap = cv2.VideoCapture(0)

    # Generate frames and perform mask detection
    logging.info("Face mask detection started")
    for frame in generate_frames():
        # Do nothing here, as the frames are being generated and processed in the generate_frames() function.
        pass
