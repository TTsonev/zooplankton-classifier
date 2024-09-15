import streamlit as st
from PIL import Image
import requests
import io

def main():
    st.title("ZOOPLANKTON CLASSIFIER")
    img_file = st.file_uploader("Upload Image: ")
    uploaded_image_button(img_file)

def uploaded_image_button(img_file):
    if st.button("Classify"):
        if img_file == None: 
            st.write("No image provided") 
        else:
            print_image(img_file)
            classify(img_file)

def print_image(img_file):
    bytes_data = img_file.read()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, width=500)

def classify(img_file):
    img_data = {"file": img_file.getvalue()}
    result = requests.post(f"http://127.0.0.1:8000/classifyUpload", files=img_data).json()
    st.header(result.get("prediction"))
    st.text("Confidence: " + str(round(result.get("confidence")*100,2)) + "%")

if __name__=='__main__':
    main()
