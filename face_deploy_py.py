import streamlit as st
import numpy as np
import pickle
import cv2
from keras_facenet import FaceNet

# Load model
model = pickle.load(open("svm_face_model.pkl", 'rb'))
facenet = FaceNet()

# Face extraction function
def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    return face

# Predict function
def predict_face(img):
    face = extract_face(img)
    if face is None:
        return "No face detected"

    face = np.expand_dims(face, axis=0)
    embedding = facenet.embeddings(face)[0]
    prediction = model.predict([embedding])
    return prediction[0]

# Streamlit UI Title
st.title("Face Recognition App")

# List of celebrities
celebrities = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman",
    "Jennifer Lawrence", "Johnny Depp", "Megan Fox", "Tom Cruise",
    "Tom Hanks", "Will Smith"
]

# Info message
st.info(
    f"Please upload a clear image of one of the following celebrities for recognition:\n\n"
    + "\n".join(f"- {name}" for name in celebrities)
)

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success("Image uploaded. Running recognition...")

    prediction = predict_face(img)
    st.write(f"Predicted Person: {prediction}")
