import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle as pickle
import base64
import os

def set_background(image_path):
    if not os.path.exists(image_path):
        st.error("Image not found. Please check the file path.")
        return

    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode()

    background_css = f"""
    <style>
    .stApp {{
        background-image:url("data:image/jpg;base64,{base64_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        filter:brightness(1.1);
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Relative Path Example
image_path = "Ratings_background.jpg"
set_background(image_path)

st.title("**Product Recommendation Engine**")

st.write("### **Data Preview**")
# Safe CSV Reading with Error Handling
try:
    df = pd.read_csv("rating_short.csv")
    st.write(df.head(5))
except FileNotFoundError:
    st.error("Ratings CSV file not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

df["date"] = pd.to_datetime(df.date, unit="s")
df["rating"] = df.rating.astype("int8")

rat = df[["date","rating"]]
latest_date = rat["date"].max()
current_date = pd.Timestamp.today()
date_shift = current_date - latest_date
rat["date"] = rat["date"] + date_shift
rat.set_index(rat.columns[0], inplace=True)
rat = rat.resample("D").mean().fillna(rat.rating.mean())

# Model Loading with Error Handling
def load_model(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model {file_name} not found. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model {file_name}: {e}")
        st.stop()

ari = load_model('ari.pkl')
RF2 = load_model('RF2.pkl')
LG2 = load_model('LG2.pkl')
LE_pid = load_model('LE_pid.pkl')
LE_y = load_model('LE_y.pkl')

LE_uid = LabelEncoder()
LE_uid.fit_transform(df.userid)

st.write("### **Ratings Timeline**")
if st.button("#### **Timeline**"):
    yforecast = ari.forecast(steps=500)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(rat, color='black')
    ax.plot(yforecast, color='green')   
    st.pyplot(fig)

st.write("### **Ratings Forecast**")
s = st.slider("**Select the number of days to be forecasted:**", 2,11,1)
if st.button("**Forecast**"):
    yforecast = ari.forecast(steps=s)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(yforecast, color='green', marker='*', markersize=20, markeredgecolor="red")
    st.write("### **Rating Forecast**")
    st.pyplot(fig)

st.write("### **Customized recommendations**")
u = st.selectbox("**Select Customer**", df.userid.unique())

st.write("**ing for individual prodcuct**")
p = st.selectbox("Select Product", df.productid.unique())
if st.button("**Ratin**g"):    
    ypredict = RF2.predict(np.array([LE_uid.transform([u]), LE_pid.transform([p])]).reshape(1, 2))
    y = LE_y.inverse_transform(ypredict.reshape(-1, 1))[0]
    st.write(f"#### *Predicted rating for product {p} by Customer {u} is {y}*")

st.write("##### **Top n Recommendations**")
n = st.slider("**Select the number of top products**", 1, 11, 1)
if st.button("**Top Product Recommended**"):   
    col2 = df.productid.unique()
    col1 = [u] * len(col2)
    dfr = pd.DataFrame({'user': col1, 'product': col2})
    dfr['product'] = LE_pid.transform(dfr['product'])
    dfr['user'] = LE_uid.transform(dfr['user'])
    dfr['rating'] = LG2.predict(dfr)
    dfr['rating'] = LE_y.inverse_transform(dfr['rating'])
    dfr.sort_values(by='rating', ascending=False, inplace=True)
    dfr = dfr.head(n)
    recommendations = LE_pid.inverse_transform(dfr['product'])
    st.write(f"**Top recommended products for customer {u} are:**", recommendations)




