import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import mysql.connector
# Set the page configuration
st.set_page_config(
    page_title="MELODYMATE",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


from PIL import Image
# Open the image file
background = Image.open("background.png")
# Calculate the middle of the image
width, height = background.size
middle = height // 2
# Split the image into two parts
top_half = background.crop((0, 0, width, middle))
bottom_half = background.crop((0, middle, width, height))
    # Rest of your code...

    # Display the top half of the image
st.image(top_half, use_column_width=True)

    # Display the heading
    #st.header("MELODYMATE")
st.markdown("""
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .header {
        color: black;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="header">𝕄𝔼𝕃𝕆𝔻𝕐𝕄𝔸𝕋𝔼</h1>', unsafe_allow_html=True)
    # Display the bottom half of the image
st.image(bottom_half, use_column_width=True)


model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1,-1)

        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

        np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


lang = st.text_input("Language")


if lang and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"

st.subheader("Feedback")
st.write("Please rate your experience (out of 5 stars):")
stars = st.slider("Stars", min_value=1, max_value=5, step=1)

suggestion = st.text_area("Suggestions (optional)")

if st.button("Submit"):
    # Process the feedback (e.g., save it to a file or database)
    if stars:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="feedback"
        )

        # Create a cursor object to execute SQL queries
        cursor = db.cursor()

        # Insert the feedback into the database
        insert_query = "INSERT INTO feedback_table (stars, suggestion) VALUES (%s, %s)"
        feedback_data = (stars, suggestion)
        cursor.execute(insert_query, feedback_data)
        db.commit()

        # Close the database connection
        cursor.close()
        db.close()
        with open("feedback.txt", "a") as f:
            f.write(f"Stars: {stars}\n")
            if suggestion:
                f.write(f"Suggestion: {suggestion}\n")
        st.success("Thank you for your feedback!")
        st.experimental_rerun()  # Rerun the app after feedback submission
    else:
        st.warning("Please provide a star rating before submitting.")