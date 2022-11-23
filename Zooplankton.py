import streamlit as st
from PIL import Image
import requests
import io

def main():
    st.title("ZOOPLANKTON CLASSIFIER")
    img_file = st.file_uploader("Upload Image: ")
    
    if st.button("Classify"):
        if img_file == None: 
            st.write("No image provided") 
        else:
            bytes_data = img_file.read()
            image = Image.open(io.BytesIO(bytes_data))
            st.image(image, width=500)
            img_data = {"file": img_file.getvalue()}
            res = requests.post(f"http://127.0.0.1:8000/classifyFile", files=img_data)
            res_data = res.json()
            st.header(res_data.get("prediction"))
            st.text("Confidence: " + str(round(res_data.get("confidence")*100,2)) + "%")

if __name__=='__main__':
    main()