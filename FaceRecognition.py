import face_recognition as face
import numpy as np
import cv2
import logging
import yaml
import time
import requests
import os
import signal
from collections import defaultdict

class LineNotify:
    def __init__(self, token):
        self.token = token
        self.url = "https://notify-api.line.me/api/notify"

    def send_message(self, message):
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {"message": message}
        response = requests.post(self.url, headers=headers, data=data)
        return response.status_code

    def send_image(self, message, image_path):
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {"message": message}
        files = {"imageFile": open(image_path, "rb")}
        response = requests.post(self.url, headers=headers, data=data, files=files)
        return response.status_code

class FaceRecognition:
    def __init__(self, config):
        self.known_face_names = config.get("known_face_names", [])
        self.known_face_encodings = self.load_reference_images(config.get("image_paths", {}))
        self.process_this_frame = 0
        self.frame_skip = config.get("frame_skip", 10)
        self.model = config.get("model", "hog")
        self.prev_results = ([], [], [])
        self.notification_sent = {}
        self.line_notify = LineNotify(config.get('line_notify_token', ''))
        self.confidence_threshold = config.get('confidence_threshold', 0.6)

    def load_reference_images(self, image_paths):
        encodings = defaultdict(list)
        for name, paths in image_paths.items():
            for path in paths:
                try:
                    image = face.load_image_file(path)
                    encoding = face.face_encodings(image)[0]
                    encodings[name].append(encoding)
                    logging.info(f"Loaded face encoding for {name} from {path}")
                except IndexError as e:
                    logging.error(f"Could not find a face in the image: {path} - {e}")
                except Exception as e:
                    logging.error(f"Error processing image {path}: {e}")
        return encodings

    def resize_frame(self, frame, max_width=640, max_height=480):
        h, w = frame.shape[:2]
        if h > max_height or w > max_width:
            scale = min(max_height/h, max_width/w)
            return cv2.resize(frame, (int(w*scale), int(h*scale))), 1/scale
        return frame, 1

    def process_frame(self, frame):
        small_frame, scale_factor = self.resize_frame(frame)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations, face_names, face_percent = [], [], []

        if self.process_this_frame == 0:
            face_locations = face.face_locations(rgb_small_frame, model=self.model, number_of_times_to_upsample=0)
            face_encodings = face.face_encodings(rgb_small_frame, face_locations, num_jitters=1)

            for face_encoding in face_encodings:
                name = "UNKNOWN"
                percent_confidence = 0

                for known_name, known_encodings in self.known_face_encodings.items():
                    face_distances = face.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]

                    current_percent_confidence = (1.0 - best_distance) * 100

                    if current_percent_confidence > percent_confidence and current_percent_confidence >= self.confidence_threshold * 100:
                        name = known_name
                        percent_confidence = current_percent_confidence

                face_names.append(name)
                face_percent.append(round(percent_confidence, 2))

            if face_locations:
                self.prev_results = (face_locations, face_names, face_percent)
        else:
            face_locations, face_names, face_percent = self.prev_results

        self.process_this_frame = (self.process_this_frame + 1) % self.frame_skip

        return face_locations, face_names, face_percent, scale_factor

    def draw_results(self, frame, face_locations, face_names, face_percent, scale_factor=1):
        font = cv2.FONT_HERSHEY_DUPLEX
        for (top, right, bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top = int(top * scale_factor)
            right = int(right * scale_factor)
            bottom = int(bottom * scale_factor)
            left = int(left * scale_factor)

            color = [46, 2, 209] if name == "UNKNOWN" else [255, 102, 51]

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left - 1, top - 30), (right + 1, top), color, cv2.FILLED)
            cv2.rectangle(frame, (left - 1, bottom), (right + 1, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, top - 6), font, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"MATCH: {percent}%", (left + 6, bottom + 23), font, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Model: {self.model}", (10, 20), font, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Frame Skip: {self.frame_skip}", (10, 40), font, 0.6, (0, 255, 0), 1)

    def save_frame(self, frame, name):
        directory = "captures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(directory, f"{name}_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        logging.info(f"Saved frame for {name} at {filepath}")
        return filepath

    def send_notification(self, name, frame):
        current_time = time.strftime("%d/%m/%y %H:%M:%S")
        message = f"รูปภาพจากการสแกน\nสวัสดีครับคุณ {name}\nสแกนใบหน้า เวลา: {current_time}"
        image_path = self.save_frame(frame, name)
        status_code = self.line_notify.send_image(message, image_path)
        if status_code == 200:
            self.notification_sent[name] = time.time()
            logging.info(f"Sent LINE notification for {name}")
        else:
            logging.error(f"Failed to send LINE Notify message for {name}. Status code: {status_code}")

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Convert image paths to dictionary
        image_paths = defaultdict(list)
        for name, path in zip(config['known_face_names'], config['image_paths']):
            image_paths[name].append(path)
        config['image_paths'] = dict(image_paths)
        
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        config = {}
    return config

def signal_handler(sig, frame):
    logging.info('You pressed Ctrl+C!')
    global running
    running = False

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- แก้ไขตรงนี้: เปลี่ยนเป็น Relative Path ---
    config = load_config('config.yaml')

    if not config:
        logging.error("Configuration not found. Exiting...")
        return

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Cannot open camera. Exiting...")
        return

    face_recognition = FaceRecognition(config)

    signal.signal(signal.SIGINT, signal_handler)
    global running
    running = True

    try:
        while running:
            start_time = time.time()

            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from camera")
                break

            face_locations, face_names, face_percent, scale_factor = face_recognition.process_frame(frame)
            face_recognition.draw_results(frame, face_locations, face_names, face_percent, scale_factor)

            for name, percent in zip(face_names, face_percent):
                if name != "UNKNOWN" and percent >= face_recognition.confidence_threshold * 100:
                    if name not in face_recognition.notification_sent or time.time() - face_recognition.notification_sent[name] > 300:
                        face_recognition.send_notification(name, frame)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Frame processing time: {processing_time:.4f} seconds")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()