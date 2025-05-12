import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import dask_bigquery

# Set page configuration
st.set_page_config(
    page_title="California Water Usage Dashboard",
    page_icon="💧",
    layout="wide"
)

# Load model and preprocessor (if available)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('water_usage_model.pkl')
        preprocessor = joblib.load('water_usage_preprocessor.pkl')
        return model, preprocessor
    except:
        st.warning("Model files not found. Prediction functionality disabled.")
        return None, None

model, preprocessor = load_model()

# Load data from BigQuery using dask_bigquery
@st.cache_data
def load_data():
    # Connect to BigQuery and load data
    ddf = dask_bigquery.read_gbq(
        project_id='total-method-443918-j6',
        dataset_id='california_water_data',
        table_id='water_levels'
    )
    
    # Convert to pandas DataFrame for Streamlit
    df = ddf.compute()
    
    # Convert month_year to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['month_year']):
        df['month_year'] = pd.to_datetime(df['month_year'])
    
    # Add policy periods for conservation analysis
    policy_dates = {
        'Pre-Policy': pd.Timestamp('2014-01-01'),
        'Emergency Drought Declaration': pd.Timestamp('2014-01-17'),
        'Mandatory Restrictions': pd.Timestamp('2015-04-01'),
        'Extended Emergency Regulations': pd.Timestamp('2016-02-02'),
        'Drought Emergency Ended': pd.Timestamp('2017-04-07'),
        'Water Conservation Legislation': pd.Timestamp('2018-05-31'),
        'Drought Emergency Renewed': pd.Timestamp('2021-10-19')
    }
    
    df['policy_period'] = 'Pre-Policy'
    for policy, date in policy_dates.items():
        mask = df['month_year'] >= date
        df.loc[mask, 'policy_period'] = policy
    
    return df

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data from BigQuery: {e}")
    st.info("Using sample data instead.")
    # Create sample data if BigQuery connection fails
    dates = pd.date_range(start='2014-01-01', end='2023-12-31', freq='M')
    df = pd.DataFrame({
        'month_year': dates,
        'water_usage_per_capita': np.random.normal(150, 25, len(dates)),
        'reported_final_total_potable_water_production': np.random.normal(1000000, 200000, len(dates)),
        'percent_non_revenue': np.random.normal(12, 3, len(dates)),
        'reservoir_storage': np.random.normal(75, 15, len(dates)),
        'county_under_drought_declaration': np.random.choice([True, False], len(dates)),
        'hydrologic_region': np.random.choice(['North Coast', 'Sacramento River', 'San Joaquin River', 
                                             'Central Coast', 'Tulare Lake', 'South Coast', 'Colorado River'], len(dates))
    })

# Sidebar filters
st.sidebar.title("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['month_year'].min().date(), df['month_year'].max().date()),
    min_value=df['month_year'].min().date(),
    max_value=df['month_year'].max().date()
)

# Region filter
regions = ['All'] + sorted(df['hydrologic_region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Hydrologic Region", regions)

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df['month_year'].dt.date >= date_range[0]) & 
                          (filtered_df['month_year'].dt.date <= date_range[1])]

if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['hydrologic_region'] == selected_region]

# Main dashboard
st.title("California Water Usage Dashboard")
st.write(f"Last updated: {datetime.now().strftime('%B %d, %Y, %I:%M %p PDT')}")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_usage = filtered_df['water_usage_per_capita'].mean()
    st.metric("Average Water Usage", f"{avg_usage:.1f} gal/capita")

with col2:
    try:
        conservation_rate = ((filtered_df[filtered_df['policy_period'] == 'Pre-Policy']['water_usage_per_capita'].mean() - 
                             filtered_df[filtered_df['policy_period'] != 'Pre-Policy']['water_usage_per_capita'].mean()) / 
                             filtered_df[filtered_df['policy_period'] == 'Pre-Policy']['water_usage_per_capita'].mean() * 100)
        st.metric("Conservation Rate", f"{conservation_rate:.1f}%")
    except:
        st.metric("Conservation Rate", "N/A")

with col3:
    drought_pct = filtered_df['county_under_drought_declaration'].mean() * 100
    st.metric("Drought Declaration", f"{drought_pct:.1f}% of time")

with col4:
    non_revenue = filtered_df['percent_non_revenue'].mean()
    st.metric("Non-Revenue Water", f"{non_revenue:.1f}%")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Water Usage Trends", "Regional Analysis", 
                                        "Conservation Impact", "Prediction Tool", "Data Explorer"])

with tab1:
    st.header("Water Usage Trends")
    
    # Time series chart
    monthly_usage = filtered_df.groupby(pd.Grouper(key='month_year', freq='M'))['water_usage_per_capita'].mean().reset_index()
    
    fig = px.line(monthly_usage, x='month_year', y='water_usage_per_capita',
                 title="Average Water Usage per Capita Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Usage per Capita (gallons)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Water production over time
    monthly_production = filtered_df.groupby(pd.Grouper(key='month_year', freq='M'))['reported_final_total_potable_water_production'].mean().reset_index()
    
    fig = px.line(monthly_production, x='month_year', y='reported_final_total_potable_water_production',
                 title="Average Potable Water Production Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Production")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Regional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional comparison
        regional_usage = filtered_df.groupby('hydrologic_region')['water_usage_per_capita'].mean().reset_index()
        regional_usage = regional_usage.sort_values('water_usage_per_capita', ascending=False)
        
        fig = px.bar(regional_usage, x='hydrologic_region', y='water_usage_per_capita',
                    title="Average Water Usage per Capita by Region")
        fig.update_layout(xaxis_title="Hydrologic Region", yaxis_title="Water Usage per Capita (gallons)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drought impact
        drought_impact = filtered_df.groupby('county_under_drought_declaration')['water_usage_per_capita'].mean().reset_index()
        
        fig = px.bar(drought_impact, x='county_under_drought_declaration', y='water_usage_per_capita',
                    title="Water Usage by Drought Declaration Status",
                    labels={'county_under_drought_declaration': 'Under Drought Declaration'})
        fig.update_layout(xaxis_title="Under Drought Declaration", yaxis_title="Water Usage per Capita (gallons)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Non-revenue water by region
    regional_nonrevenue = filtered_df.groupby('hydrologic_region')['percent_non_revenue'].mean().reset_index()
    regional_nonrevenue = regional_nonrevenue.sort_values('percent_non_revenue', ascending=False)
    
    fig = px.bar(regional_nonrevenue, x='hydrologic_region', y='percent_non_revenue',
                title="Average Non-Revenue Water Percentage by Region")
    fig.update_layout(xaxis_title="Hydrologic Region", yaxis_title="Non-Revenue Water Percentage")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Conservation Impact Analysis")
    
    # Water usage by policy period
    policy_impact = filtered_df.groupby('policy_period')['water_usage_per_capita'].mean().reset_index()
    
    fig = px.bar(policy_impact, x='policy_period', y='water_usage_per_capita',
                title="Average Water Usage by Conservation Policy Period")
    fig.update_layout(xaxis_title="Policy Period", yaxis_title="Water Usage per Capita (gallons)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series with policy markers - FIXED VERSION
    fig = px.line(monthly_usage, x='month_year', y='water_usage_per_capita',
                 title="Water Usage with Conservation Policy Implementation Dates")
    
    # Add vertical lines for policy dates - FIXED by separating lines and annotations
    policy_dates = {
        'Emergency Drought Declaration': '2014-01-17',
        'Mandatory Restrictions': '2015-04-01',
        'Extended Emergency Regulations': '2016-02-02',
        'Drought Emergency Ended': '2017-04-07',
        'Water Conservation Legislation': '2018-05-31',
        'Drought Emergency Renewed': '2021-10-19'
    }
    
    # First add the lines without annotations
    for policy, date in policy_dates.items():
        fig.add_vline(x=date, line_dash="dash", line_color="red")
        
        # Then add separate annotations
        fig.add_annotation(
            x=date,
            y=1.05,
            text=policy,
            showarrow=False,
            yref="paper",
            textangle=-90,
            font=dict(size=10)
        )
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Usage per Capita (gallons)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional conservation impact
    if 'Pre-Policy' in filtered_df['policy_period'].unique():
        # Calculate regional conservation rates
        pre_policy_regional = filtered_df[filtered_df['policy_period'] == 'Pre-Policy'].groupby('hydrologic_region')['water_usage_per_capita'].mean()
        post_policy_regional = filtered_df[filtered_df['policy_period'] != 'Pre-Policy'].groupby('hydrologic_region')['water_usage_per_capita'].mean()
        
        conservation_by_region = pd.DataFrame({
            'hydrologic_region': pre_policy_regional.index,
            'pre_policy': pre_policy_regional.values,
            'post_policy': post_policy_regional.values
        })
        
        conservation_by_region['reduction_percent'] = (conservation_by_region['pre_policy'] - conservation_by_region['post_policy']) / conservation_by_region['pre_policy'] * 100
        conservation_by_region = conservation_by_region.sort_values('reduction_percent', ascending=False)
        
        fig = px.bar(conservation_by_region, x='hydrologic_region', y='reduction_percent',
                    title="Water Usage Reduction Percentage by Region")
        fig.update_layout(xaxis_title="Hydrologic Region", yaxis_title="Reduction Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Water Usage Prediction Tool")
    
    if model is not None and preprocessor is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            
            population = st.slider("Population Served", 1000, 1000000, 100000, step=1000)
            
            drought_status = st.radio("Drought Declaration Status", ["No Drought", "Drought"])
            drought_value = 1 if drought_status == "Drought" else 0
            
            climate_zone = st.selectbox("Climate Zone", [1, 2, 3, 4])
            
            month = st.selectbox("Month", list(range(1, 13)), 
                                format_func=lambda x: datetime(2023, x, 1).strftime('%B'))
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'total_population_served': [population],
                'county_under_drought_declaration': [drought_value],
                'climate_zone': [climate_zone],
                'month': [month],
                'year': [2023],
                'water_shortage_level_indicator': [0],
                'reservoir_storage': [75],
                'reported_recycled_water': [0],
                'percent_non_revenue': [12],
                'hydrologic_region': ['South Coast']
            })
            
            if st.button("Predict Water Usage"):
                # Preprocess the input data
                input_processed = preprocessor.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_processed)[0]
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Display prediction
                    st.markdown(f"### Predicted Water Usage: **{prediction:.2f}** gallons per capita")
                    
                    # Compare to average
                    avg_usage = df['water_usage_per_capita'].mean()
                    
                    if prediction < avg_usage * 0.8:
                        efficiency = "highly efficient"
                        color = "green"
                    elif prediction < avg_usage * 1.2:
                        efficiency = "average"
                        color = "blue"
                    else:
                        efficiency = "inefficient"
                        color = "red"
                    
                    st.markdown(f"This prediction indicates <span style='color:{color}'>{efficiency}</span> water usage compared to the state average of {avg_usage:.2f} gallons per capita.", unsafe_allow_html=True)
                    
                    # Display factors
                    st.subheader("Key Factors")
                    
                    factors = []
                    
                    if population > 500000:
                        factors.append("Large population size typically increases per capita usage")
                    elif population < 50000:
                        factors.append("Small population size typically decreases per capita usage")
                    
                    if drought_value == 1:
                        factors.append("Drought conditions typically reduce usage by 10-20%")
                    
                    if month >= 6 and month <= 9:
                        factors.append(f"Summer months ({datetime(2023, month, 1).strftime('%B')}) typically show higher usage")
                    else:
                        factors.append(f"Non-summer months ({datetime(2023, month, 1).strftime('%B')}) typically show lower usage")
                    
                    for factor in factors:
                        st.markdown(f"- {factor}")
    else:
        st.info("Prediction model not available. Please ensure model files are properly loaded.")

with tab5:
    st.header("Data Explorer")
    
    # Show the raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)
    
    # Download the data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="california_water_data.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown("California Water Usage Dashboard | Data Mining Class Project | May 2025")