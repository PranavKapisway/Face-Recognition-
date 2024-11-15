import cv2
import numpy as np
from PIL import Image
import os

sample_path = 'samples'

# Create the trainer directory if it doesn't exist
trainer_dir = 'trainer'
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def Images_And_Labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("Training faces. It will take a few seconds. Wait ...")

faces, ids = Images_And_Labels(sample_path)
recognizer.train(faces, np.array(ids))

# Save the trainer.yml file in the trainer directory
trainer_file = os.path.join(trainer_dir, 'trainer.yml')
recognizer.write(trainer_file)

print("Model trained. Now we can recognize your face.")
