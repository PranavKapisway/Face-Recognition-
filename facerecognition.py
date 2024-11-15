import cv2
import csv

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the face cascade
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define the font for text on the image
font = cv2.FONT_HERSHEY_SIMPLEX

# Load user names from file
names = {}
with open('usernames.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        names[int(row[0])] = row[1]

# Open the webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # Set video frame width
cam.set(4, 480)  # Set video frame height

# Define minimum window size to be recognized as a face
minH = 0.1 * cam.get(4)
minW = 0.1 * cam.get(3)

while True:
    # Read the frame
    ret, img = cam.read()

    # Convert the image to grayscale
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        converted_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict the ID and accuracy of the detected face
        id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])

        # Adjust the threshold for recognized faces
        if accuracy < 50:
            id_name = names.get(id, "Unknown")
            accuracy_text = "  {0}%".format(round(100 - accuracy))
        else:
            id_name = "Unknown"
            accuracy_text = "  {0}%".format(round(100 - accuracy))

        # Display the ID and accuracy on the image
        cv2.putText(img, str(id_name), (x + 5, y - 5), font, 1, (255, 25, 255), 2)
        cv2.putText(img, str(accuracy_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Show the image
    cv2.imshow('camera', img)

    # Break the loop if 'ESC' key is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Release the webcam and close the window
print("Thanks for using this program, have a good day.")
cam.release()
cv2.destroyAllWindows()

    
