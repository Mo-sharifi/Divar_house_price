import streamlit as st 
import pandas as pd 
import numpy as np 
import time
import joblib 

XGB_model = joblib.load("Divar_house_prediction/models/XGBRegressor.sav")
DT_model = joblib.load("Divar_house_prediction/models/DecisionTreeRegressor.sav")
RF_model = joblib.load("Divar_house_prediction/models/RandomForestRegressor.sav")
ELN_model =joblib.load("Divar_house_prediction/models/XGBRegressor.sav")

st.title("Tehran House _Prediction_")
st.write("### Specific information is needed for get better results ")

Address = (
 'Shahran',
 'Pardis',
 'Shahrake Qods',
 'Shahrake Gharb',
 'North Program Organization',
 'Andisheh',
 'West Ferdows Boulevard',
 'Narmak',
 'Zafar',
 'Islamshahr',
 'Pirouzi',
 'Shahrake Shahid Bagheri',
 'Moniriyeh',
 'Saadat Abad',
 'Amirieh',
 'Southern Janatabad',
 'Salsabil',
 'Zargandeh',
 'Feiz Garden',
 'Water Organization',
 'ShahrAra',
 'Gisha',
 'Ray',
 'Abbasabad',
 'Ostad Moein',
 'Farmanieh',
 'Parand',
 'Punak',
 'Qasr-od-Dasht',
 'Aqdasieh',
 'Railway',
 'Central Janatabad',
 'East Ferdows Boulevard',
 'Pakdasht KhatunAbad',
 'Sattarkhan',
 'Shahryar',
 'Northern Janatabad',
 'Daryan No',
 'Southern Program Organization',
 'Rudhen',
 'West Pars',
 'Afsarieh',
 'Marzdaran',
 'Sadeghieh',
 'Chahardangeh',
 'Pakdasht',
 'Baqershahr',
 'Jeyhoon',
 'Lavizan',
 'Shams Abad',
 'Fatemi',
 'Keshavarz Boulevard',
 'Baghestan',
 'Kahrizak',
 'Qarchak',
 'Northren Jamalzadeh',
 'Azarbaijan',
 'Bahar',
 'Persian Gulf Martyrs Lake',
 'Beryanak',
 'Heshmatieh',
 'Elm-o-Sanat',
 'Golestan',
 'Shahr-e-Ziba',
 'Pasdaran',
 'Chardivari',
 'Gholhak',
 'Heravi',
 'Hashemi',
 'Dehkade Olampic',
 'Republic',
 'Zaferanieh',
 'Gheitarieh',
 'Qazvin Imamzadeh Hassan',
 'Niavaran',
 'Valiasr',
 'Amir Bahador',
 'Ekhtiarieh',
 'Ekbatan',
 'Haft Tir',
 'Mahallati',
 'Ozgol',
 'Tajrish',
 'Abazar',
 'Koohsar',
 'Hekmat',
 'Parastar',
 'Majidieh',
 'Southern Chitgar',
 'Karimkhan',
 'Si Metri Ji',
 'Karoon',
 'Northern Chitgar',
 'East Pars',
 'Kook',
 'Air force',
 'Sohanak',
 'Velenjak',
 'Kamranieh',
 'Komeil',
 'Azadshahr',
 'Zibadasht',
 'Amirabad',
 'Dorous',
 'Mirdamad',
 'Razi',
 'Qalandari',
 'Jordan',
 'Yaftabad',
 'Mehran',
 'Nasim Shahr',
 'Tenant',
 'Chardangeh',
 'Fallah',
 'Eskandari',
 'Shahrakeh Naft',
 'Ajudaniye',
 'Tehransar',
 'Nawab',
 'Yousef Abad',
 'Northern Suhrawardi',
 'Villa',
 'Hakimiyeh',
 'Nezamabad',
 'Garden of Saba',
 'Tarasht',
 'Azari',
 'Shahrake Apadana',
 'Araj',
 'Vahidieh',
 'Malard',
 'Shahrake Azadi',
 'Darband',
 'Tehran Now',
 'Dezashib'
)

area = st.slider("### Area of House", 30, 200, 70) # area start with min 30 and max 200 and defualt 70
room = st.selectbox("### Room of House",[0,1,2,3,4,5])
parking = int(st.checkbox("Have Parking "))
warehouse = int(st.checkbox("Have warehouse "))
elevator = int(st.checkbox("Have elevator "))
address = st.selectbox("Address of House",Address)
dt_model = st.checkbox("DecisionTree")
rf_model = st.checkbox("RandomForest")
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

    

    
    
