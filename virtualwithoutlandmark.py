import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing model for visualization
mp_drawing = mp.solutions.drawing_utils

# Function to overlay PNG image on another image
def overlayPNG(imgBack, imgFront, pos):
    hf, wf, _ = imgFront.shape
    hb, wb, _ = imgBack.shape
    *_, alpha_channel = cv2.split(imgFront)
    alpha_mask = alpha_channel / 255.0

    # Ensure the front image is within the bounds of the back image
    if pos[1] + hf > hb:
        hf = hb - pos[1]
        imgFront = imgFront[:hf, :, :]
        alpha_mask = alpha_mask[:hf, :]
    if pos[0] + wf > wb:
        wf = wb - pos[0]
        imgFront = imgFront[:, :wf, :]
        alpha_mask = alpha_mask[:, :wf]

    for c in range(3):
        imgBack[pos[1]:pos[1]+hf, pos[0]:pos[0]+wf, c] = \
            imgBack[pos[1]:pos[1]+hf, pos[0]:pos[0]+wf, c] * (1 - alpha_mask) + \
            imgFront[:, :, c] * alpha_mask

    return imgBack

# Define the fixed ratio for resizing
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440

# Load dress image
dress_path = r"C:\virtualtryon\RED.png"  # Path to the PNG image with a transparent background
dress_img = cv2.imread(dress_path, cv2.IMREAD_UNCHANGED)

# Check if the dress image was loaded successfully
if dress_img is None:
    raise FileNotFoundError(f"Image file not found at {dress_path}")

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set up the output window
cv2.namedWindow("Virtual Try-On", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Try-On", 800, 1000)  # Set the window size

# Define padding for the T-shirt
padding = 20  # Add some padding to the dimensions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        left_shoulder_x = int(left_shoulder.x * frame.shape[1])
        left_shoulder_y = int(left_shoulder.y * frame.shape[0])
        right_shoulder_x = int(right_shoulder.x * frame.shape[1])
        right_shoulder_y = int(right_shoulder.y * frame.shape[0])
        left_hip_x = int(left_hip.x * frame.shape[1])
        left_hip_y = int(left_hip.y * frame.shape[0])
        right_hip_x = int(right_hip.x * frame.shape[1])
        right_hip_y = int(right_hip.y * frame.shape[0])

        # Calculate the width and height of the T-shirt with increased width
        torso_width = int((left_shoulder_x - right_shoulder_x) * 1.4)  # Increase width multiplier
        torso_height = int((left_hip_y - left_shoulder_y) * 1.2)
        width_of_shirt = int(torso_width * fixedRatio)
        height_of_shirt = int(torso_height * shirtRatioHeightWidth)
        
        if width_of_shirt > 0 and height_of_shirt > 0:
            dress_resized = cv2.resize(dress_img, (width_of_shirt, height_of_shirt))

            # Calculate the current scale and offset
            current_scale = (left_shoulder_x - right_shoulder_x) / 190
            offset_x = int(44 * current_scale)
            offset_y = int(48 * current_scale)

            # Calculate the position to place the shirt image on the frame with padding and left shift
            dress_x = max(right_shoulder_x - offset_x - padding // 2 - 20, 0)  # Move left by 20 pixels
            dress_y = max(right_shoulder_y - offset_y - padding // 2, 0)

            # Define points for perspective transformation
            src_points = np.array([
                [0, 0],
                [dress_img.shape[1], 0],
                [dress_img.shape[1], dress_img.shape[0]],
                [0, dress_img.shape[0]]
            ], dtype="float32")

            dst_points = np.array([
                [dress_x, dress_y],
                [dress_x + width_of_shirt, dress_y],
                [dress_x + width_of_shirt, dress_y + height_of_shirt],
                [dress_x, dress_y + height_of_shirt]
            ], dtype="float32")

            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            transformed_dress = cv2.warpPerspective(dress_img, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

            # Overlay the transformed dress image on the frame
            frame = overlayPNG(frame, transformed_dress, (0, 0))

    cv2.imshow("Virtual Try-On", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
