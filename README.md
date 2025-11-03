# Face-Detection-Age-Gender-Estimation

Real-time Facial Attribute Detector (Age, Gender, Emotion)

This project implements a real-time system for detecting and tracking facial attributes including Age, Gender, and Emotion from a live webcam feed. It combines the power of InsightFace for robust face detection, identification, and attribute estimation, with an ONNX-optimized model (FER+) for high-speed emotion recognition.

It integrates advanced features to leverage the efficiency of **Human-Computer Interaction (HCI)** applications, such as stable face ID assignment and an Age Locking Mechanism to prevent attribute flicker.



Features

- Real-time Attribute Detection: Simultaneous detection of Age, Gender, and Emotion.
- Persistent Face Tracking: Uses face embeddings and cosine similarity to assign a persistent Face ID to individuals across frames, even if they briefly leave the view.
- Age Locking Mechanism: Implements a stability buffer and thresholding to "lock" the predicted age for a tracked individual, significantly reducing age flicker and improving perceived accuracy.
- High-Speed Emotion Recognition: Utilizes the FER+ (Emotion Recognition in Faces) ONNX model for fast and accurate emotion classification (8 categories).
- Attribute Cleanup: Automatically cleans up internal tracking data (`face_id`, age locks) for individuals who have left the frame.
