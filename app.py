import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Hand Sign Recognizer", page_icon="🤟")
st.title("🤟 AI Hand Sign Recognizer")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Add or rename signs here. Keep labels simple—no spaces.
GESTURES = [
    "HELLO",
    "FIST",
    "PEACE",
    "THUMBS_UP",
    "OK",
    "ROCK",
    "CALL_ME",
    "STOP",
]


def landmarks_to_features(hand_landmarks):
    """Convert 21 hand landmarks into normalized x/y/z features."""
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32,
    )

    # Make coordinates relative to the wrist, then normalize for hand size.
    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 0:
        points /= scale

    return points.flatten()


def collect_samples():
    st.subheader("Step 1: Collect training examples")
    st.write(
        "For each sign, show it clearly to the camera and collect examples "
        "from slightly different hand positions."
    )

    selected_gesture = st.selectbox("Choose the sign you are showing", GESTURES)
    sample_count = st.slider("Examples to collect", 20, 100, 40, step=10)

    start = st.checkbox("Start collecting")
    video_area = st.empty()
    status = st.empty()

    if start:
        camera = cv2.VideoCapture(0)
        samples = []

        if not camera.isOpened():
            st.error("Could not access the camera.")
            return None, None

        try:
            with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:
                while len(samples) < sample_count:
                    ok, frame = camera.read()
                    if not ok:
                        status.error("Could not read a camera frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    result = hands.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )

                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS
                        )
                        samples.append(landmarks_to_features(hand))

                    video_area.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                    )
                    status.write(
                        f"Collected {len(samples)} / {sample_count} examples"
                    )
        finally:
            camera.release()

        if samples:
            labels = [selected_gesture] * len(samples)
            return samples, labels

    return None, None


def recognize_live(model):
    st.subheader("Step 3: Recognize signs")
    run = st.checkbox("Start recognition")
    video_area = st.empty()
    result_area = st.empty()

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Could not access the camera.")
            return

        try:
            with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:
                while run:
                    ok, frame = camera.read()
                    if not ok:
                        st.error("Could not read a camera frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    result = hands.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )

                    message = "Show a hand sign"
                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS
                        )
                        features = landmarks_to_features(hand).reshape(1, -1)
                        message = model.predict(features)[0]

                    video_area.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                    )
                    result_area.subheader(f"Recognized sign: {message}")
        finally:
            camera.release()


if "training_data" not in st.session_state:
    st.session_state.training_data = []
    st.session_state.training_labels = []
    st.session_state.model = None

new_samples, new_labels = collect_samples()

if new_samples:
    st.session_state.training_data.extend(new_samples)
    st.session_state.training_labels.extend(new_labels)
    st.success(f"Saved {len(new_samples)} examples for {new_labels[0]}.")

if st.button("Train / update recognizer"):
    if len(set(st.session_state.training_labels)) < 2:
        st.warning("Collect examples for at least two different signs first.")
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(
            np.array(st.session_state.training_data),
            st.session_state.training_labels,
        )
        st.session_state.model = model
        st.success("Recognizer trained!")

if st.session_state.model is not None:
    recognize_live(st.session_state.model)
else:
    st.info(
        "Collect examples for at least two signs, then click "
        "'Train / update recognizer'."
    )