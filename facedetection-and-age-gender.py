#------- This file integrate InsightFace and ONNX Emotion pipline to detect facial age/gender/emotion -------"
#------- Integrated advanced features for leverage efficiency of HCI ------------------#

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque


# ---------- Config ----------
#----- Reference Research Paper: https://arxiv.org/abs/1608.01041 -----#
#----- The model below is a Deep Convolution Neural Network for Emotion Recognition in Faces -----#
#----- Download the model: curl -L -o emotion-ferplus-8.onnx https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

MODEL_PATH = "emotion-ferplus-8.onnx"     # FER+ Emotion Recoginition Pre-Trained ONNX Model: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus

EMOTIONS = ["Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt"]


# ---------- Load ONNX model ----------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]  # macOS CPU
)

input_name = session.get_inputs()[0].name  # usually "Input3" for FER+

# ---------- InsightFace detector ----------
app = FaceAnalysis(name="buffalo_s")    # Model Pack Name = buffalo_l, Detection Model: RetinaFace-10GF and Recognition Model: ResNet50@WebFace600K	
app.prepare(ctx_id=0, det_size=(640, 640))  # CPU

# Gender Mapping
gender_map = {0: "Female", 1: "Male"}

# Global dictionary: ID -> embedding for collecting face_id
known_faces = {}    

#----- Age Locking Machanism Config ------------------
age_locks = {}  # Stores final locked ages per face_id, e.g., {"A123": 32},  Final confirmed (locked) ages.
age_buffers = {}   # Stores recent age predictions per face_id, e.g., {"A123": [31, 33, 32, ...]}, Rolling window of predicted ages to check stability over time.


# Age Locking Machanism Function
def update_age_lock(face_id, new_age, threshold=0.8, buffer_size=30, min_frames=30):
    """
    Lock age only after:
    1. At least `min_frames` observed (warm-up period)
    2. Stability (low variance) is high enough
    Returns locked_age or current smoothed estimate.
    """

    # Initialize Buffers
    if face_id not in age_buffers:
        age_buffers[face_id] = []
    if face_id not in age_locks:
        age_locks[face_id] = None

    # If already locked, return locked value
    if age_locks[face_id] is not None:
        return age_locks[face_id]

    # Add new age to buffer
    age_buffers[face_id].append(new_age)
    if len(age_buffers[face_id]) > buffer_size:
        age_buffers[face_id].pop(0)

    # Check stability after enough frames
    if len(age_buffers[face_id]) >= min_frames:
        std = np.std(age_buffers[face_id])      # Standard deviation of predictions
        mean = np.mean(age_buffers[face_id])    # Average predicted age
        stability = 1 - (std / (mean + 1e-6))   # Stability score: closer to 1 = stable

        if stability > threshold:
            age_locks[face_id] = int(round(mean))


    return age_locks[face_id] if age_locks[face_id] is not None else int(round(np.mean(age_buffers[face_id])))


def cleanup_locks(active_face_ids):
    """
    Remove locks and face_id for faces that have disappeared from the frame.
    """
    for fid in list(age_buffers.keys()):
        if fid not in active_face_ids:
            del age_buffers[fid]
            age_locks.pop(fid, None)
            known_faces.pop(fid, None)  # Clenup the individual face_id


# --------------------------------------------------------------------------------------------------
# Function: Generate Individual Face ID and lock ID for individual face to prevent ID vary changing.
# --------------------------------------------------------------------------------------------------

def generate_random_id(length=8):
    """Generate random alphanumeric ID"""
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=length))

def get_face_id(new_embedding, threshold=0.45):
    # If no faces yet
    if not known_faces:
        new_id = generate_random_id()
        known_faces[new_id] = new_embedding
      
        return new_id

    # Compare with known embeddings
    sims = {fid: cosine_similarity([new_embedding], [emb])[0][0]
            for fid, emb in known_faces.items()}


    best_id, best_sim = max(sims.items(), key=lambda x: x[1])

    if best_sim > threshold:
        # Match existing ID
        known_faces[best_id] = new_embedding
        
        return best_id
    else:
        # New person → assign fresh random ID
        new_id = generate_random_id()
        known_faces[new_id] = new_embedding
       
        return new_id


# ------------------------------------------------------------------------------------------
# ---------- Utilities for Emotion Estimation with FER+ Emotion Recognition Model ----------
# ------------------------------------------------------------------------------------------

def preprocess_for_ferplus(bgr_face: np.ndarray) -> np.ndarray:
    """ This prepares each detected face for the FER+ ONNX emotion model. """
    """Convert cropped BGR face to FER+ input: (1,1,64,64) float32."""

    # Convert to grayscale
    gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    # Optional: improve contrast for robustness
    gray = cv2.equalizeHist(gray)
    # Resize to 64x64
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    # (1,1,64,64) float32
    x = gray.astype(np.float32)[None, None, :, :]
    return x

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)



def main():
    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        raise SystemExit

    print("Webcam started. Press 'q' to quit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces on the full frame
        faces = app.get(frame)

        active_face_ids = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop face & run FER+
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            # # Running the ONNX FER+ model to get emotion probabilities
            inp = preprocess_for_ferplus(face_crop)
            probs = softmax(session.run(None, {input_name: inp})[0].squeeze())

            # # Selecting the top emotion
            top_idx = int(np.argmax(probs))
            emotion = EMOTIONS[top_idx]
            conf = float(probs[top_idx]) # Confident score of emotion

            # print(face)

            # Age + Gender from InsightFace
            gender = gender_map.get(face.gender, "Unknown")
            current_age = face.age

            # Facial Get Embedding  
            embedding = face.normed_embedding
            face_id = get_face_id(embedding)
            active_face_ids.append(face_id)

            # Update locking logic
            display_age = update_age_lock(face_id, current_age)
            
            # Change box color if locked
            color_box_label = (0, 255, 0) if age_locks.get(face_id) is not None else (255, 0, 0)


            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box_label, 2)

            # Emotion, Age, Gender Display Information
            label = f"{emotion} {conf:.2f}, Age: {display_age}, Gender: {gender}"

            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box_label, 2)
            
            # Face ID at bottom-right of box
            id_label = f"ID: {face_id}"
            cv2.putText(frame, id_label, (x1, y2 + 20),   # shift left so it fits inside box area
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box_label, 2)

        # Real-time Emotion (FER+ ONNX + InsightFace)
        cv2.imshow("Real-time Facial Attribute Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Clean up the face_id and Age Locking Machanism when the individual face is left from the frame.
        cleanup_locks(active_face_ids)


    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()

