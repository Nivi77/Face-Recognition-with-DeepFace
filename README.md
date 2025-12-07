---cam_fallback.py & with_fallback.py---

Dual-Model Face Recognition (CPU Version)

This repository contains two Python scripts that implement a robust fallback-based face recognition system using the DeepFace library.

Why Dual-Model?
Because no single model works perfectly in all real-world conditions:

- Facenet512 → High accuracy on clear still images  
- SFace → More robust to tilt, lighting issues, partial occlusion, hair covering face, webcam noise

---fast_recog_SFace.py---

This project is a real-time face-recognition system using OpenCV and DeepFace with the SFace ONNX model. It detects faces from a webcam, computes embeddings, compares them with saved known faces, and displays red/green tracking boxes with the person’s name, confidence, and match distance. The script also prints whether GPU or CPU is being used.

--------------------

Project Structure
your-repo/
│
├── with_fallback.py
├── cam_fallback.py
├── fast_recog.py
├── knownfaces/
│   ├── person1.jpg
│   ├── person2.png
│   └── ...
│
└── test_images/
    └── test1.png


Place all known faces inside knownfaces/

-----------------------

Install dependencies:

pip install deepface opencv-python pandas numpy

------------------------

