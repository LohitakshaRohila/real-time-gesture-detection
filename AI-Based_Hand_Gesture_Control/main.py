import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('gesture_classifier.h5')

# Function to preprocess the frame from webcam
def prepare_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image.reshape(1, 28, 28, 1)  # Reshape to model input
    image = image.astype('float32') / 255  # Normalize the image
    return image

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the captured frame
    prepared_image = prepare_image(frame)

    # Predict the digit
    prediction = model.predict(prepared_image)
    predicted_digit = np.argmax(prediction)

    # Display the prediction on the frame
    cv2.putText(frame, f"Predicted: {predicted_digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Webcam Handwritten Digit Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
