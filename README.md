# Face-Recognition-App
An AI-powered face detection and recognition web application built using **Streamlit** and **DeepFace**. The app detects human faces from uploaded images, extracts facial features using deep learning, and identifies known individuals based on facial similarity.

## Features

- Upload an image containing one or more human faces
- Automatic face detection using **RetinaFace**
- Face recognition using **ArcFace** embeddings
- Bounding box visualization with identity labels
- Handles RGB and non-RGB images gracefully
- Lightweight and easy-to-deploy Streamlit UI

---

## Tech Stack

- **Python**
- **Streamlit** – Web UI
- **DeepFace** – Face recognition framework
- **ArcFace** – Face embedding model
- **RetinaFace** – Face detection model
- **NumPy** – Numerical computations
- **Pillow (PIL)** – Image processing

---

## System Architecture

Image Upload
↓
Image Preprocessing (RGB Conversion)
↓
Face Detection (RetinaFace)
↓
Face Embedding Extraction (ArcFace)
↓
Cosine Similarity Matching
↓
Face Labeling & Visualization


---

## How It Works

1. User uploads an image through the Streamlit interface.
2. Image is converted to RGB format if required.
3. RetinaFace detects all faces and returns bounding boxes.
4. ArcFace generates embeddings for each detected face.
5. Each embedding is compared with known face embeddings using cosine distance.
6. Faces are labeled if a match is found, otherwise marked as **Unknown**.

---

## Matching Logic

- Similarity Metric: **Cosine Distance**
- Recognition Threshold: **0.4**
- Best match below threshold is considered a valid identity

---

## Project Structure

face-recognition-app/
│
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Documentation


---

## Running the Application

### Install Dependencies
```bash
pip install -r requirements.txt

### Run Streamlit App
streamlit run app.py
