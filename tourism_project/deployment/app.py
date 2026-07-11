import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model

# Define the Hugging Face dataset repository ID
repo_id = "gayathriL/wellness_tourism_package_prediction"

# replace with your repoid
model_path = hf_hub_download(repo_id=repo_id, filename="best_wellness_tourism_model_v1.joblib")

model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package. 
Please enter the customer details below to get a purchase prediction.
""")

# User input
Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 0, 10, 1)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=30000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    # Changed result messages
    result = "Will Purchase Wellness Tourism Package" if prediction == 1 else "Will Not Purchase Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
