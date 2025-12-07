Dual-Model Face Recognition (CPU Version)

This repository contains two Python scripts that implement a robust fallback-based face recognition system using the DeepFace library.

Why Dual-Model?
Because no single model works perfectly in all real-world conditions:

- Facenet512 → High accuracy on clear still images  
- SFace → More robust to tilt, lighting issues, partial occlusion, hair covering face, webcam noise  

--------------------

Project Structure
your-repo/
│
├── with_fallback.py
├── cam_fallback.py
│
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

⚠️ No GPU / CUDA / onnxruntime-gpu required.
This project runs fully on CPU, because Facenet512 does not support GPU acceleration in DeepFace.

