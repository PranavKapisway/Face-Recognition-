import cv2
import os

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the samples directory if it doesn't exist
if not os.path.exists('samples'):
    os.makedirs('samples')

# Ask for user ID and name
face_id = input("Enter a Numeric user ID here: ")
face_name = input("Enter the user name here: ")

# Save user info
with open('usernames.csv', 'a') as f:
    f.write(f"{face_id},{face_name}\n")

print("Taking samples, look at the camera...")
count = 0

while True:
    ret, img = cam.read()
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the samples directory
        cv2.imwrite(f"samples/face.{face_id}.{count}.jpg", converted_image[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100:
        break

print("Samples taken, now closing the program...")
cam.release()
cv2.destroyAllWindows()
