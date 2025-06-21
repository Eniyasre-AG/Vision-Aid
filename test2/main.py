import cv2
import numpy as np
import easyocr
from spellchecker import SpellChecker
from wordsegment import load, segment
import pyttsx3
import face_recognition
import os
import sys
import math
import requests  # Import requests to interact with AI API
import google.generativeai as genai

# Load the word segment model
load()

# Initialize the pyttsx3 engine for TTS
engine = pyttsx3.init()
# Gemini API key
API_KEY = 'AIzaSyD_YHXNDHd7UkPvNa_lEHGrkYwZNc-iRXw'  # Replace with your actual API key
genai.configure(api_key=API_KEY)


# Initialize the EasyOCR reader (using English for this example)
reader = easyocr.Reader(['en'])

def preprocess_image(image, method='otsu'):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a mild Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'adaptive':
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        # Apply Otsu's thresholding for dynamic binarization
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Apply a global threshold with a lower value to preserve text
        _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # Optionally invert the image (depends on the original image)
    inverted = cv2.bitwise_not(binary)

    return inverted

def correct_text(text):
    spell = SpellChecker()
    
    # Tokenize the text into words
    words = text.split()
    
    # Correct misspelled words
    corrected_text = ' '.join([spell.correction(word) if spell.correction(word) is not None else word for word in words])
    
    return corrected_text

def segment_text(text):
    # Segment the text into proper words using the wordsegment library
    segmented_words = segment(text)
    
    # Join the segmented words with spaces
    segmented_text = ' '.join(segmented_words)
    
    return segmented_text

def read_text(text):
    # Use the pyttsx3 engine to read the text
    engine.say(text)
    engine.runAndWait()

# Function to get AI response (dummy function)
def get_ai_response(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(text)  # Generate content based on the text input
    if response:
        return response.text  # Return the generated content
    else:
        return "No AI-generated content found."


# Helper function to calculate face confidence
def face_confidence(face_distance, face_match_threshold=0.4):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# Function to check if an image is blurry
def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # A threshold for blur detection

# Preprocessing function to improve face image quality
def preprocess_face_image(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True

        # Initialize by encoding known faces
        self.encode_faces()

    # Encode faces from the "faces" directory
    def encode_faces(self):
        if not os.path.exists('faces'):
            sys.exit("Faces directory not found!")

        for person_name in os.listdir('faces'):
            person_folder = os.path.join('faces', person_name)
            if not os.path.isdir(person_folder):
                continue  # Skip if it's not a directory

            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                print(f"Processing image: {image_path}")
                face_image = face_recognition.load_image_file(image_path)
                
                # Preprocess image for better recognition
                face_image = preprocess_face_image(face_image)
                
                # Check if the image is blurry and skip it if true
                if detect_blur(face_image):
                    print(f"Image {image_name} is blurry, skipping...")
                    continue

                face_encodings = face_recognition.face_encodings(face_image)

                # Check if a face was found in the image
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person_name)
                else:
                    print(f"No face found in {image_name}, skipping...")

        if len(self.known_face_encodings) == 0:
            sys.exit("No faces found in the directory. Please add known faces.")

        print(f"Known faces: {self.known_face_names}")

    def run_face_recognition(self):
        # Use the ESP32-CAM stream URL instead of the local camera
        #stream_url = "http://192.168.30.35/stream"
        stream_url = "http://192.168.30.35:81/stream"

        video_capture = cv2.VideoCapture(stream_url)

        if not video_capture.isOpened():
            sys.exit("Error: Unable to access ESP32-CAM stream...")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame from ESP32-CAM stream.")
                break

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                self.face_colors = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'
                    box_color = (0, 0, 255)

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        confidence = face_confidence(face_distances[best_match_index])
                        if float(confidence.strip('%')) >= 50:
                            name = self.known_face_names[best_match_index]
                            box_color = (0, 255, 0)

                    self.face_names.append(f'{name} ({confidence})' if name != "Unknown" else "Unknown")
                    self.face_colors.append(box_color)

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name, color in zip(self.face_locations, self.face_names, self.face_colors):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Text Recognition (OCR) Mode
def run_text_recognition():
    cap = cv2.VideoCapture(0)  # Ensure correct camera index

    # Set the resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Camera Feed - Press "c" to Capture', frame)

        # Press 'c' to capture the current frame for OCR
        if cv2.waitKey(1) & 0xFF == ord('c'):
            original_image_path = 'captured_image.png'
            cv2.imwrite(original_image_path, frame)
            preprocessed_image = preprocess_image(frame)
            processed_image_path = 'processed_image.png'
            cv2.imwrite(processed_image_path, preprocessed_image) # Preprocess the image for better OCR
            result = reader.readtext(preprocessed_image)


            if result:
                # Extract and join the detected text
                extracted_text = ' '.join([res[1] for res in result])
                print("Extracted Text:", extracted_text)

                # Optional: Correct and segment text
                corrected_text = correct_text(extracted_text)
                #segmented_text = segment_text(corrected_text)

                print("Corrected Text:", corrected_text)
                #print("Segmented Text:", segmented_text)

                # Read the text using TTS
                read_text(corrected_text)

                # Ask user for additional support
                choice = input("Press '1' for additional support or '2' for no further support: ")
                
                if choice == '1':
                    ai_response = get_ai_response("give more information on following cover every detail which you think is important "+extracted_text)
                    print(ai_response)
                    read_text(ai_response)
                else:
                    print("No additional support requested.")

            else:
                print("No text detected. Please try again.")

        # Press 'q' to quit
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    read_text("Press 1 for Face Detection Mode. Press 2 for Text Recognition Mode.")

    mode = input("Enter 1 for Face Detection or 2 for Text Recognition: ")

    if mode == '1':
        fr = FaceRecognition()
        fr.run_face_recognition()
    elif mode == '2':
        run_text_recognition()
    else:
        read_text("Invalid input. Please restart and try again.")
