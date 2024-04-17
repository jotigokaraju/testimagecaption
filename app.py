import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the processor and model
@st.cache_resource
def load_processor_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
    return processor, model

# Streamlit script
def main():
    # Load processor and model
    processor, model = load_processor_model()

    # Capture image
    photo = st.camera_input("Take a picture")

    # Process the captured image
    if photo is not None:
        image = Image.open(photo).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption when button is clicked
        if st.button("Generate Caption"):
            # Process image with the processor
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            # Generate caption with the model
            generated_ids = model.vision_encoder(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Display the generated caption
            st.write("Generated Caption:", generated_text)

if __name__ == "__main__":
    main()
