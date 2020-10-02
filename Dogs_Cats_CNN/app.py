import streamlit as st
import tensorflow
from classify import predict
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache
# model = load_model

def main():
    st.title("Dogs/Cats Classification")
    html_temp = """
    <div style="background-color:tomato;padding:15px;"
    <h1> With Convoultion Neural Networks </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image...",type=["jpg","png"])

    if uploaded_file is not None:
        file = uploaded_file.read()
        file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
        img_cv2 = cv2.imdecode(np.fromstring(file, np.uint8), 1)
        img = Image.open(uploaded_file)
        st.image(img,caption='Uploaded Image.',use_column_width=True)
        pred = predict(img_cv2)
        if pred == 1:
            st.success("Bow Bow, it's a DOG")
        else:
            st.success("Meow, it's a CAT")
        st.balloons()

if __name__ == '__main__':
    main()