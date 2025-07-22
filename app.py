import streamlit as st 
import pandas as pd 
import numpy as np 
import time
import joblib 

XGB_model = joblib.load("models/XGBRegressor.sav")
DT_model = joblib.load("models/DecisionTreeRegressor.sav")
RF_model = joblib.load("models/RandomForestRegressor.sav")
ELN_model =joblib.load("models/ElasticNet.sav")


with open ("data/Addrese.txt") as f : 
    Address = f.read().split("\n")


st.title("Tehran House _Prediction_")
st.write("### Specific information is needed for get better results ")

area = st.slider("### Area of House", 30, 200, 70) # area start with min 30 and max 200 and defualt 70
room = st.selectbox("### Room of House",[0,1,2,3,4,5])
parking = int(st.checkbox("Have Parking "))
warehouse = int(st.checkbox("Have warehouse "))
elevator = int(st.checkbox("Have elevator "))
address = st.selectbox("Address of House",Address)
dt_model = st.checkbox("Decision Tree")
rf_model = st.checkbox("Random Forest")
xgb_model = st.checkbox("XGBoost")
eln_model = st.checkbox("ElasticNet")

columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']

ok = st.button("Calculate Price")

if ok : 
    X_new = np.array([area,room,parking,warehouse,elevator,address])
    X_new_df = pd.DataFrame([X_new], columns = columns)
    if dt_model:
        price = DT_model.predict(X_new_df)
    elif rf_model:
        price = RF_model.predict(X_new_df)
    elif xgb_model:
        price = XGB_model.predict(X_new_df)
    elif eln_model:
        price = ELN_model.predict(X_new_df)
    else:
        price = list(1)
    
    time.sleep(0.5)
    st.success(f"The estimated price is {price[0]:,.0f} Toman")

    st.info("I Try to add more Algorithms :)")

    

    
    
