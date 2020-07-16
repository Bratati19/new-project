import numpy as np
import pickle
import pandas as pd
import streamlit as st
#app=Flask(__name__)
#Swagger(app) 

from PIL import Image
pickle_in=open("model.pkl",'rb')
model = pickle.load(pickle_in)
def welcome():
    return "Wellcome All"

def predict_note_authentication(radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se ,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst):
    input = np.array([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se ,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst ,fractal_dimension_worst]]).astype(np.float32)
    
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][1], 1)
    return float(pred)
    
    



def main():
    st.title("Breast Cancer Prediction")
    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:white;text-align:center;"> Breast Cancer Predictor  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    radius_mean= st.text_input("radius_mean","Type Here")
    texture_mean= st.text_input("texture_mean","Type Here")
    perimeter_mean= st.text_input("perimeter_mean","Type Here")
    area_mean = st.text_input("area_mean","Type Here")
    smoothness_mean = st.text_input("smoothness_mean","Type Here")
    compactness_mean = st.text_input("compactness_mean","Type Here")
    concavity_mean = st.text_input("concavity_mean","Type Here")
    concave_points_mean = st.text_input("concave points_mean","Type Here")
    symmetry_mean = st.text_input("symmetry_mean","Type Here")
    fractal_dimension_mean = st.text_input("fractal_dimension_mean","Type Here")
    radius_se= st.text_input("radius_se","Type Here")
    texture_se = st.text_input("texture_se","Type Here")
    perimeter_se = st.text_input("perimeter_se","Type Here")
    area_se = st.text_input("area_se","Type Here")
    smoothness_se = st.text_input("smoothness_se","Type Here")
    compactness_se = st.text_input("compactness_se","Type Here")
    concavity_se = st.text_input("concavity_se","Type Here")
    concave_points_se = st.text_input("concave points_se","Type Here")
    symmetry_se = st.text_input("symmetry_se","Type Here")
    fractal_dimension_se = st.text_input("fractal_dimension_se","Type Here")
    radius_worst = st.text_input("radius_worst","Type Here")
    texture_worst = st.text_input("texture_worst","Type Here")
    perimeter_worst = st.text_input("perimeter_worst","Type Here")
    area_worst = st.text_input("area_worst","Type Here")
    smoothness_worst = st.text_input("smoothness_worst","Type Here")
    compactness_worst = st.text_input("compactness_worst","Type Here")
    concavity_worst = st.text_input("concavity_worst","Type Here")
    concave_points_worst = st.text_input("concave point_worst","Type Here")
    symmetry_worst= st.text_input("symmetry_worst","Type Here")
    fractal_dimension_worst = st.text_input("fractional_dimension_worst","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst ,fractal_dimension_worst)
    
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> You are  safe,congratulation</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> You are in danger,please consult a doctor</h2>
       </div>
    """
    st.success('The probability of breast cancer is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
    if (float(result) > 0.5):
            st.markdown(danger_html,unsafe_allow_html=True)
    else:
            st.markdown(safe_html,unsafe_allow_html=True)


if __name__=='__main__':
    main()