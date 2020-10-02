import streamlit as st
import tensorflow
from classify import predict
from PIL import Image
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
    #file = uploaded_file.read()
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img,caption='Uploaded Image.',use_column_width=True)
        pred = predict(uploaded_file)
        if pred == 1:
            st.success("Bow Bow, it's a DOG")
        else:
            st.success("Meow, it's a CAT")
        st.balloons()

if __name__ == '__main__':
    main()