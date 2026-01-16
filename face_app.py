import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageDraw
from deepface import DeepFace
from numpy.linalg import norm
import io

st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("🧠 Face Detection & Recognition")

# ---------- Utility Functions ----------

def convert_to_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file)
    image = convert_to_rgb(image)
    return np.array(image)

def load_image_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image = convert_to_rgb(image)
        return np.array(image)
    except Exception:
        return None

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

# ---------- Load Known Faces ----------
@st.cache_resource
def load_known_faces():
    known_embeddings = []
    known_names = []

    known_people = {
        "Chris Evans": "https://media.suara.com/pictures/970x544/2020/05/02/69243-chris-evans-shutterstock.jpg",
        "Chris Hemsworth": "https://asset.kompas.com/crops/m8WyOBGlOq_zcGPMWdnZ3EWykd8=/0x0:680x453/750x500/data/photo/2022/06/25/62b69da1f0c8b.jpg",
        "Mark Ruffalo": "https://i.pinimg.com/564x/3c/fa/6e/3cfa6e41d376e56d3a401739d3a4faf1--mark-ruffalo-beautiful-men.jpg",
        "Robert Downey Jr.": "https://www.themoviedb.org/t/p/w300_and_h450_bestv2/5qHNjhtjMD4YWH3UP0rm4tKwxCL.jpg"
    }

    for name, url in known_people.items():
        img = load_image_from_url(url)
        if img is None:
            continue

        embedding = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            enforce_detection=False
        )[0]["embedding"]

        known_embeddings.append(np.array(embedding))
        known_names.append(name)

    return known_embeddings, known_names

known_face_embeddings, known_face_names = load_known_faces()

# ---------- UI ----------
st.subheader("📸 Upload an Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

image = None
if uploaded_file:
    image = load_image_from_upload(uploaded_file)

# ---------- Face Detection & Recognition ----------
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    detections = DeepFace.extract_faces(
        img_path=image,
        detector_backend="retinaface",
        enforce_detection=False
    )

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for face in detections:
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        face_img = image[y:y+h, x:x+w]

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name="ArcFace",
            enforce_detection=False
        )[0]["embedding"]

        embedding = np.array(embedding)

        name = "Unknown"
        distances = [cosine_distance(embedding, k) for k in known_face_embeddings]

        if distances:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

        draw.rectangle(
            ((x, y), (x + w, y + h)),
            outline=(255, 255, 0),
            width=4
        )
        draw.text((x, y - 10), name, fill=(255, 255, 0))

    st.subheader("🧍 Detected Faces")
    st.image(pil_image, use_container_width=True)
    st.success(f"Detected {len(detections)} face(s)")
