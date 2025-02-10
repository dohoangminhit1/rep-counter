import cv2
import PIL
import numpy as np
from sklearn.svm import LinearSVC
import os

class Model:
    def __init__(self):
        self.model = LinearSVC()
        self.image_size = (150, 150)
        self.flattened_size = self.image_size[0] * self.image_size[1]

    def train_model(self, counters):
        global img_path
        img_list = []
        class_list = []

        # Process class 1 images
        for i in range(1, counters[0]):
            try:
                img_path = f"1/frame{i}.jpg"
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.image_size)
                img_list.append(img.reshape(self.flattened_size))
                class_list.append(1)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        # Process class 2 images
        for i in range(1, counters[1]):
            try:
                img_path = f"2/frame{i}.jpg"
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.image_size)
                img_list.append(img.reshape(self.flattened_size))
                class_list.append(2)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        if not img_list:
            raise ValueError("No valid images found for training")

        X = np.array(img_list)
        y = np.array(class_list)
        self.model.fit(X, y)
        print(f"Model trained on {len(img_list)} images")

    def predict(self, frame):
        try:
            frame = frame[1]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, self.image_size)
            flattened = resized.reshape(self.flattened_size)
            prediction = self.model.predict([flattened])
            return prediction[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None