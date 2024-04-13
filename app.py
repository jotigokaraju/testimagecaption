import streamlit as st
from transformers import pipeline
from PIL import Image

caption=pipeline('image-to-text', model="ydshieh/vit-gpt2-coco-en")
upload = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if upload is not None:
    image = Image.open(upload)
    st.image(image)

if st.button("Caption"):
    captions = caption(image)
    st.write(captions[0]['generated_text'])
