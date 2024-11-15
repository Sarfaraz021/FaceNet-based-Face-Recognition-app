import streamlit as st
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np


# Initialize face matching components
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def face_match(img_path, data_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))

    face, prob = mtcnn(img, return_prob=True)

    # Check if a face is detected
    if face is None:
        return ('No Face Detected', 0, 'No Match')

    emb = resnet(face.unsqueeze(0)).detach()

    saved_data = torch.load(data_path)
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    dist_list = []

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    threshold = 0.6
    if min(dist_list) < threshold:
        return (name_list[idx_min], min(dist_list), 'Match')
    else:
        return ('Unknown', min(dist_list), 'No Match')


def main():
    st.title("AI Face Recognition")
    st.markdown(
        """
        <style>
        .main {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_image = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the image temporarily for processing
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Perform face matching
        if st.button("Match Face"):
            try:
                # Replace 'data.pt' with the path to your data file
                result = face_match(image_path, "data.pt")
                st.success(f"Face matched with: {result[0]}")
                st.info(f"Distance: {result[1]} | Status: {result[2]}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
