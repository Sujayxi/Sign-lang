from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# Load the trained models
model_right = load_model('models/asl_model_right.h5')
model_left = load_model('models/asl_model_left.h5')

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the video capture
cap = cv2.VideoCapture(0)

# Global variable to store detected words and current letter
detected_word = ""
last_predicted_letter = ""

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Function to generate video frames
def generate_frames():
    global detected_word, last_predicted_letter
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        last_prediction_time = time.time()  # Track the last prediction time

        while True:
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Define section boxes for hand detection
            image_height, image_width, _ = image.shape
            left_hand_box = (int(image_width * 0.1), int(image_height * 0.1), int(image_width * 0.4), int(image_height * 0.7))
            right_hand_box = (int(image_width * 0.6), int(image_height * 0.1), int(image_width * 0.9), int(image_height * 0.7))

            # Process the image to find hand landmarks
            results = hands.process(image_rgb)

            # Create a white background for the hand landmarks
            white_bg = np.ones(image.shape, dtype=np.uint8) * 255

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine if the hand is left or right
                    hand_label = handedness.classification[0].label
                    model = model_right if hand_label == 'Right' else model_left

                    # Get hand bounding box coordinates
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    min_x, max_x = int(min(x_coords) * image.shape[1]), int(max(x_coords) * image.shape[1])
                    min_y, max_y = int(min(y_coords) * image.shape[0]), int(max(y_coords) * image.shape[0])

                    # Check if the hand is within the respective section box
                    if (hand_label == 'Right' and right_hand_box[0] <= min_x <= right_hand_box[2] and right_hand_box[1] <= min_y <= right_hand_box[3]) or \
                       (hand_label == 'Left' and left_hand_box[0] <= min_x <= left_hand_box[2] and left_hand_box[1] <= min_y <= left_hand_box[3]):
                        # Draw landmarks on the original image and white background
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(
                            white_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                        # Check if enough time has passed since the last prediction
                        current_time = time.time()
                        if current_time - last_prediction_time > 3:  # Adjust delay as needed
                            last_prediction_time = current_time  # Update the last prediction time

                            # Add a margin to the bounding box for better hand coverage
                            margin = 20
                            min_x = max(0, min_x - margin)
                            max_x = min(image.shape[1], max_x + margin)
                            min_y = max(0, min_y - margin)
                            max_y = min(image.shape[0], max_y + margin)

                            # Crop the hand region from the white background
                            hand_region = white_bg[min_y:max_y, min_x:max_x]

                            # Ensure the cropped hand region is valid
                            if hand_region.size > 0:
                                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                                resized = cv2.resize(gray, (64, 64))
                                reshaped = np.reshape(resized, (1, 64, 64, 1)) / 255.0  # Normalize and reshape

                                # Predict the ASL sign using the respective model
                                prediction = model.predict(reshaped)
                                class_index = np.argmax(prediction)

                                # Map class index to the corresponding alphabet
                                if hand_label == 'Right':
                                    last_predicted_letter = chr(65 + class_index)
                                else:
                                    last_predicted_letter = chr(78 + class_index)

                                # Update the detected word
                                detected_word += last_predicted_letter

            # Combine the original image and the white background with landmarks side by side
            combined_image = np.hstack((image, white_bg))

            ret, buffer = cv2.imencode('.jpg', combined_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_word', methods=['POST'])
def update_word():
    global detected_word
    action = request.json.get("action")
    if action == "space":
        detected_word += " "
    elif action == "delete":
        detected_word = detected_word[:-1] if detected_word else detected_word
    return jsonify({'updated_word': detected_word})

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    """Return the current detected word and last predicted letter."""
    global detected_word, last_predicted_letter
    return jsonify({
        'current_letter': last_predicted_letter,
        'prediction': detected_word
    })

if __name__ == '__main__':
    app.run(debug=True)
