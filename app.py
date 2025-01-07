import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Decorator for wait
def wait_for_model(func):
    def wrapper(*args, **kwargs):
        with st.spinner('Loading the model...'):
            time.sleep(2)  # Simulate waiting time for model loading
            return func(*args, **kwargs)
    return wrapper

# Main app
@wait_for_model
def process_image(query, img):
    raw_image = Image.open(img).convert('RGB')
    # Conditional image captioning
    inputs = processor(raw_image, query, return_tensors="pt")
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    return result

# Streamlit UI
st.title('Image Captioning & Query')
st.write("Upload an image and ask any query related to it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
query = st.text_input("Ask a query related to the image:", "")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

# Submit button for processing
if st.button('Submit') and uploaded_file is not None and query != "":
    result = process_image(query, uploaded_file)
    st.write("Query Result: ", result)
