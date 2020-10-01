import streamlit as st
import joblib
from PIL import Image

vectorizer = joblib.load('vectorizer.joblib')
model_NB = joblib.load('model_NB.joblib')

@st.cache
def predict_gender(data):
    vect = vectorizer.transform(data)
    result = model_NB.predict(vect)
    return result

def load_images(image_name):
    img = Image.open(image_name)
    return st.image(img,width=300)

def main():

    st.title("Gender Classification ML App")

    html_temp = """
    <div style="background-color:tomato;padding:15px;">
    <h2> With Multinomial NaiveBayes Model </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    name = st.text_input("Enter Name")
    if st.button("Classify"):
        result = predict_gender([name.lower()])
        if result[0] == 0:
            prediction = 'Female'
            c_image = 'female.png'
        else:
            prediction = 'Male'
            c_image = 'male.png'
        
        st.success("{}, was classified as {}".format(name.title(),prediction))
        load_images(c_image)

if __name__ == '__main__':
    main()