import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model_path = 'svc_model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

st.title('👩🏻‍🎓 Performance Analysis')
# Create a sidebar for user input
st.sidebar.header("Input")

# Defining input fields for each feature
gender = st.sidebar.selectbox('Gender', options=[0, 1])
ethnicity = st.sidebar.selectbox('Ethnicity', options=[0, 1, 2])
parental_education = st.sidebar.selectbox('Parental Education', options=[0, 1, 2, 3])
study_time_weekly = st.sidebar.slider('Study Time Weekly', min_value=0, max_value=40, value=10)
absences = st.sidebar.slider('Absences', min_value=0, max_value=100, value=0)
tutoring = st.sidebar.selectbox('Tutoring', options=[0, 1])
parental_support = st.sidebar.selectbox('Parental Support', options=[0, 1])
extracurricular = st.sidebar.selectbox('Extracurricular', options=[0, 1])
sports = st.sidebar.selectbox('Sports', options=[0, 1])
music = st.sidebar.selectbox('Music', options=[0, 1])
volunteering = st.sidebar.selectbox('Volunteering', options=[0, 1])

# Create DataFrame from inputs
input_data = {
    'Gender': [gender],
    'Ethnicity': [ethnicity],
    'ParentalEducation': [parental_education],
    'StudyTimeWeekly': [study_time_weekly],
    'Absences': [absences],
    'Tutoring': [tutoring],
    'ParentalSupport': [parental_support],
    'Extracurricular': [extracurricular],
    'Sports': [sports],
    'Music': [music],
    'Volunteering': [volunteering]
}

input_df = pd.DataFrame(input_data)

# Prediction button
if st.sidebar.button('Predict'):
    # Note: Below encoders and scalers should ideally be loaded from trained objects
    label_encoder = joblib.load("C:\\Users\\Vaibhav\\Downloads\\Performance_Analysis\\label_encoder.pkl")
    scaler = joblib.load("C:\\Users\\Vaibhav\\Downloads\\Performance_Analysis\\scaler.pkl")

    # Encoding and scaling
    for column in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']:
        input_df[column] = label_encoder.transform(input_df[column])  # Replace with loaded encoder

    numeric_columns = ['StudyTimeWeekly', 'Absences']
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])  # Replace with loaded scaler

    # Prediction
    prediction = loaded_model.predict(input_df)

    # Display the prediction
    #st.write(f"The given student has been classified as Class: {prediction[0]}")
    #st.markdown(f"<h3 style='font-weight:bold;'>The given student has been classified as Class: {prediction[0]}</h3>", unsafe_allow_html=True)
    # Display the prediction in bold, center-aligned, moved down slightly, and in green color
    st.markdown("<br>" * 4, unsafe_allow_html=True)  # Adds space before the output
    st.markdown(
        f"<div style='text-align: center;'><h3 style='font-weight:bold; color: green;'>The given student has been classified as Class: {prediction[0]}</h3></div>",
        unsafe_allow_html=True)
else:
    st.write("Enter the input values and press predict.")
