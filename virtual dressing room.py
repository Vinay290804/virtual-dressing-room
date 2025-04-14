import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import mediapipe as mp
import matplotlib.pyplot as plt
import os

class VirtualDressingRoom:
    def __init__(self):
        # Initialize body pose estimation using MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load pre-trained model for body segmentation
        self.segmentation_model = self._load_segmentation_model()

        # Initialize garment database
        self.garments = self._initialize_garment_database()

    def _load_segmentation_model(self):
        """Load and return a pre-trained model for body segmentation"""
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=output)
        # In a real implementation, you would load specific weights for body segmentation
        return model

    def _initialize_garment_database(self):
        """Initialize a database of 3D garment models"""
        # In a real implementation, this would load actual 3D models
        return {
            'tshirt_red': {
                'id': 1,
                'name': 'Red T-Shirt',
                'type': 'top',
                'color': 'red',
                'file_path': 'models/tshirt_red.obj',
                'texture_path': 'textures/tshirt_red.jpg'
            },
            'jeans_blue': {
                'id': 2,
                'name': 'Blue Jeans',
                'type': 'bottom',
                'color': 'blue',
                'file_path': 'models/jeans_blue.obj',
                'texture_path': 'textures/jeans_blue.jpg'
            },
            'dress_black': {
                'id': 3,
                'name': 'Black Dress',
                'type': 'full',
                'color': 'black',
                'file_path': 'models/dress_black.obj',
                'texture_path': 'textures/dress_black.jpg'
            }
        }

    def detect_body_pose(self, image):
        """Detect body pose landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks

    def segment_body(self, image):
        """Segment the body from the background"""
        # Resize image to match model input size
        resized = cv2.resize(image, (224, 224))
        normalized = resized / 255.0
        # Predict segmentation mask
        # In a real implementation, this would use a proper segmentation model
        prediction = self.segmentation_model.predict(np.expand_dims(normalized, axis=0))
        # Generate binary mask
        mask = (prediction > 0.5).astype(np.uint8)
        mask = cv2.resize(mask[0], (image.shape[1], image.shape[0]))
        return mask

    def fit_garment(self, image, pose_landmarks, garment_id):
        """Fit a 3D garment model to the detected body"""
        if garment_id not in self.garments:
            raise ValueError(f"Garment ID {garment_id} not found in database")

        garment = self.garments[garment_id]

        # In a real implementation, this would:
        # 1. Load the 3D garment model
        # 2. Scale and position it based on pose landmarks
        # 3. Apply physics-based simulation for realistic draping
        # 4. Render the garment onto the image

        # For this example, we'll just create a simple overlay
        overlay = image.copy()

        # Draw a simple placeholder for the garment
        if garment['type'] == 'top':
            # Draw a rectangle for the top
            if pose_landmarks:
                left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

                h, w, _ = image.shape
                top_left = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                top_right = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                bottom_left = (int(left_hip.x * w), int(left_hip.y * h))
                bottom_right = (int(right_hip.x * w), int(right_hip.y * h))

                pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                pts = pts.reshape((-1, 1, 2))

                # Convert garment color to BGR
                color = (0, 0, 255)  # Red in BGR
                if garment['color'] == 'blue':
                    color = (255, 0, 0)
                elif garment['color'] == 'green':
                    color = (0, 255, 0)
                elif garment['color'] == 'black':
                    color = (0, 0, 0)

                cv2.fillPoly(overlay, [pts], color)

        # Apply the overlay with transparency
        alpha = 0.6
        output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return output

    def process_frame(self, frame, selected_garment):
        """Process a single frame for virtual try-on"""
        # Detect body pose
        pose_landmarks = self.detect_body_pose(frame)

        # Segment the body
        body_mask = self.segment_body(frame)

        # Fit the selected garment
        output_frame = self.fit_garment(frame, pose_landmarks, selected_garment)

        # Draw pose landmarks for debugging
        if pose_landmarks:
            frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            self.mp_drawing.draw_landmarks(
                frame_rgb, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        return output_frame

    def run_realtime(self, camera_index=0):
        """Run the virtual dressing room in real-time using webcam"""
        cap = cv2.VideoCapture(camera_index)
        selected_garment = 'tshirt_red'  # Default garment
         # Create a Matplotlib figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 10))  
        plt.ion()  # Turn on interactive mode for live updates
        plt.show()

        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            output_frame = self.process_frame(frame, selected_garment)

            # Display output
            cv2.imshow('AI Virtual Dressing Room', output_frame)
             # Display output using matplotlib
            # Convert BGR to RGB for matplotlib
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)  
            ax.imshow(output_frame_rgb)  
            ax.set_title('AI Virtual Dressing Room')
            fig.canvas.draw()  # Update the plot
            fig.canvas.flush_events()  # Ensure the plot is displayed
           

            # Press 'q' to quit, '1', '2', '3' to change garments
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                selected_garment = 'tshirt_red'
            elif key == ord('2'):
                selected_garment = 'jeans_blue'
            elif key == ord('3'):
                selected_garment = 'dress_black'

        cap.release()
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)  # Close the Matplotlib figure


    def process_image(self, image_path, selected_garment):
        """Process a single image for virtual try-on"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        elif key == ord('2'):
             selected_garment = 'jeans_blue'
        elif key == ord('3'):
             selected_garment = 'dress_black'

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path, selected_garment):
        """Process a single image for virtual try-on"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Process the image
        output_image = self.process_frame(image, selected_garment)

        return output_image

# Example usage
if __name__ == "__main__":
    vdr = VirtualDressingRoom()

    # Option 1: Run in real-time with webcam
    vdr.run_realtime()

    # Option 2: Process a single image
    # output = vdr.process_image('path/to/image.jpg', 'tshirt_red')
    # cv2.imwrite('output.jpg', output)
