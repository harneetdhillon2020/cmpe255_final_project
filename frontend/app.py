import streamlit as st
import pandas as pd
import requests

# Fetch data from CNRA API
url = "https://data.cnra.ca.gov/api/3/action/datastore_search_sql?sql=SELECT%20*%20from%20%22af157380-fb42-4abf-b72a-6f9f98868077%22%20WHERE%20%22title%22%20LIKE%20%27jones%27"
response = requests.get(url)
data = response.json()

# Print the structure to debug
st.write("API Response Structure:", data.keys())

# If 'result' exists, check its structure
if 'result' in data:
    st.write("Result Structure:", data['result'].keys())
    
    # Check if 'records' exists in result
    if 'records' in data['result']:
        # Convert to DataFrame
        df = pd.DataFrame(data['result']['records'])
        
        # Check if the dataframe has latitude and longitude columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Ensure coordinates are numeric
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Display the map
            st.map(df)
        else:
            st.error("The dataset doesn't contain latitude and longitude columns")
            # Display available columns
            st.write("Available columns:", df.columns.tolist())
    else:
        st.error("No 'records' found in the API response")
else:
    st.error("No 'result' found in the API response")
