import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the pipeline outside Streamlit script
caption = None

@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline('image-to-text', model="ydshieh/vit-gpt2-coco-en")

# Streamlit script
def main():
    global caption
    if caption is None:
        caption = load_model()

    upload = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if upload is not None:
        image = Image.open(upload)
        st.image(image)

    if st.button("Caption") and image is not None:
        captions = caption(image)
        st.write(captions[0]['generated_text'])

if __name__ == "__main__":
    main()
