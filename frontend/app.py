# filename: app.py

import streamlit as st
import requests

st.title("Dashboard")

if st.button("Ping Server"):
    try:
        response = requests.get("http://localhost:8000/ping")
        if response.status_code == 200:
            data = response.json()
            st.success(f"Response: {data['message']}")
        else:
            st.error("Server responded with an error.")
    except Exception as e:
        st.error(f"Error: {e}")
