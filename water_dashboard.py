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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    
    # Extract month and year for easier filtering
    df['month'] = df['month_year'].dt.month
    df['year'] = df['month_year'].dt.year
    
    # Add season column
    month_to_season = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df['season'] = df['month'].map(month_to_season)
    
    # Create drought severity categories based on palmer_hydro
    if 'palmer_hydro' in df.columns:
        df['drought_severity'] = pd.cut(
            df['palmer_hydro'], 
            bins=[-float('inf'), -4, -3, -2, -1, 0, float('inf')],
            labels=['Extreme', 'Severe', 'Moderate', 'Mild', 'Normal', 'Wet']
        )
    
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
                                             'Central Coast', 'Tulare Lake', 'South Coast', 'Colorado River'], len(dates)),
        'precip': np.random.normal(5, 2, len(dates)),
        'max_temp': np.random.normal(25, 5, len(dates)),
        'palmer_hydro': np.random.normal(-1, 2, len(dates)),
        'pH': np.random.normal(7.5, 0.5, len(dates)),
        'DissolvedOxygen': np.random.normal(9, 2, len(dates))
    })
    # Extract month and year
    df['month'] = df['month_year'].dt.month
    df['year'] = df['month_year'].dt.year
    # Add season column
    month_to_season = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df['season'] = df['month'].map(month_to_season)
    # Create drought severity categories
    df['drought_severity'] = pd.cut(
        df['palmer_hydro'], 
        bins=[-float('inf'), -4, -3, -2, -1, 0, float('inf')],
        labels=['Extreme', 'Severe', 'Moderate', 'Mild', 'Normal', 'Wet']
    )

# Create a form for filters to avoid constant rerunning
with st.sidebar:
    st.title("Filters")
    
    with st.form("filter_form"):
        st.header("Time Period")
        # Date range filter
        date_range = st.date_input(
            "Select Date Range",
            value=(df['month_year'].min().date(), df['month_year'].max().date()),
            min_value=df['month_year'].min().date(),
            max_value=df['month_year'].max().date()
        )
        
        # Season filter
        seasons = ['All'] + sorted(df['season'].unique().tolist())
        selected_season = st.selectbox("Select Season", seasons)
        
        st.header("Location")
        # Region filter
        regions = ['All'] + sorted(df['hydrologic_region'].unique().tolist())
        selected_region = st.selectbox("Select Hydrologic Region", regions)
        
        # County filter (dynamically populated based on region)
        if 'county' in df.columns:
            if selected_region != 'All':
                available_counties = ['All'] + sorted(df[df['hydrologic_region'] == selected_region]['county'].unique().tolist())
            else:
                available_counties = ['All'] + sorted(df['county'].unique().tolist())
            selected_county = st.selectbox("Select County", available_counties)
        
        st.header("Environmental Conditions")
        # Drought severity filter
        if 'drought_severity' in df.columns:
            drought_severities = ['All'] + sorted(df['drought_severity'].dropna().unique().tolist())
            selected_drought = st.selectbox("Drought Severity", drought_severities)
            
            # Set drought_value based on selected_drought
            if selected_drought in ['Wet', 'Normal']:
                drought_value = 0  # No drought
            elif selected_drought != 'All':
                drought_value = 1  # Drought condition
            else:
                # If 'All' is selected, you might want to use a default value
                # or leave it unset to not filter by drought_value
                drought_value = 0
        
        # Temperature range - single slider that sets both min and max
        if 'max_temp' in df.columns and 'min_temp' in df.columns:
            temp_absolute_min = float(df['min_temp'].min())
            temp_absolute_max = float(df['max_temp'].max())
            
            temp_range = st.slider(
                "Temperature Range (°F)",
                min_value=temp_absolute_min,
                max_value=temp_absolute_max,
                value=(temp_absolute_min, temp_absolute_max)
            )
            
            # Calculate average temperature from the range
            avg_temp = (temp_range[0] + temp_range[1]) / 2
        
        # Submit button for filters
        filter_submitted = st.form_submit_button("Apply Filters")

# Apply filters
filtered_df = df.copy()

# Date range filter
filtered_df = filtered_df[(filtered_df['month_year'].dt.date >= date_range[0]) & 
                          (filtered_df['month_year'].dt.date <= date_range[1])]

# Season filter
if selected_season != 'All':
    filtered_df = filtered_df[filtered_df['season'] == selected_season]

hydrologic_region = 'Central Coast'  # Default region if 'All' is selected
# Region filter
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['hydrologic_region'] == selected_region]
    hydrologic_region = selected_region

# County filter
if 'county' in df.columns and 'selected_county' in locals() and selected_county != 'All':
    filtered_df = filtered_df[filtered_df['county'] == selected_county]

# Drought severity filter
if 'drought_severity' in df.columns and 'selected_drought' in locals() and selected_drought != 'All':
    filtered_df = filtered_df[filtered_df['drought_severity'] == selected_drought]

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

# Create tabs with additional tabs for climate and water quality
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Water Usage Trends", 
    "Regional Analysis", 
    "Conservation Impact", 
    "Climate Analysis",
    "Water Quality",
    "Prediction Tool", 
    "Data Explorer"
])

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
    
    # Seasonal patterns
    if 'season' in filtered_df.columns:
        st.subheader("Seasonal Water Usage Patterns")
        
        seasonal_usage = filtered_df.groupby('season')['water_usage_per_capita'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        
        fig = px.bar(
            x=seasonal_usage.index,
            y=seasonal_usage.values,
            title="Average Water Usage by Season",
            labels={'x': 'Season', 'y': 'Water Usage per Capita (gallons)'}
        )
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
    
    # Drought severity by region if available
    if 'drought_severity' in filtered_df.columns:
        # Calculate percentage of each severity level by region
        drought_region = filtered_df.groupby(['hydrologic_region', 'drought_severity']).size().reset_index()
        drought_region.columns = ['hydrologic_region', 'drought_severity', 'count']
        total_by_region = drought_region.groupby('hydrologic_region')['count'].sum().reset_index()
        total_by_region.columns = ['hydrologic_region', 'total']
        drought_region = drought_region.merge(total_by_region, on='hydrologic_region')
        drought_region['percentage'] = drought_region['count'] / drought_region['total'] * 100
        
        fig = px.bar(
            drought_region,
            x='hydrologic_region',
            y='percentage',
            color='drought_severity',
            title="Drought Severity Distribution by Region",
            labels={'percentage': 'Percentage', 'hydrologic_region': 'Hydrologic Region'}
        )
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
    st.header("Climate Analysis")
    
    # Check if climate columns exist
    climate_cols = ['precip', 'max_temp', 'min_temp', 'avg_temp', 'palmer_hydro', 'palmer_z_index']
    available_climate_cols = [col for col in climate_cols if col in filtered_df.columns]
    
    if available_climate_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature vs Water Usage
            if 'avg_temp' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df, 
                    x='avg_temp', 
                    y='water_usage_per_capita',
                    color='hydrologic_region',
                    title="Temperature vs Water Usage",
                    labels={'avg_temp': 'Average Temperature (°C)', 'water_usage_per_capita': 'Water Usage per Capita (gallons)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precipitation vs Water Usage
            if 'precip' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df, 
                    x='precip', 
                    y='water_usage_per_capita',
                    color='hydrologic_region',
                    title="Precipitation vs Water Usage",
                    labels={'precip': 'Precipitation (inches)', 'water_usage_per_capita': 'Water Usage per Capita (gallons)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Drought index impact
        if 'palmer_hydro' in filtered_df.columns and 'drought_severity' in filtered_df.columns:
            drought_usage = filtered_df.groupby('drought_severity')['water_usage_per_capita'].mean().reset_index()
            
            fig = px.bar(
                drought_usage,
                x='drought_severity',
                y='water_usage_per_capita',
                title="Water Usage by Drought Severity",
                labels={'drought_severity': 'Drought Severity', 'water_usage_per_capita': 'Water Usage per Capita (gallons)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Climate heatmap
        st.subheader("Climate Correlation Analysis")
        climate_corr_cols = available_climate_cols + ['water_usage_per_capita']
        corr_matrix = filtered_df[climate_corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Between Climate Variables and Water Usage"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal climate patterns
        if 'season' in filtered_df.columns:
            seasonal_climate = filtered_df.groupby('season')[available_climate_cols].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
            
            # Normalize for visualization
            seasonal_climate_norm = (seasonal_climate - seasonal_climate.min()) / (seasonal_climate.max() - seasonal_climate.min())
            seasonal_climate_norm = seasonal_climate_norm.reset_index()
            
            # Melt for plotting
            seasonal_climate_melt = pd.melt(
                seasonal_climate_norm, 
                id_vars=['season'], 
                value_vars=available_climate_cols,
                var_name='Climate Variable', 
                value_name='Normalized Value'
            )
            
            fig = px.line(
                seasonal_climate_melt,
                x='season',
                y='Normalized Value',
                color='Climate Variable',
                title="Seasonal Climate Patterns (Normalized)",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No climate data available in the current dataset.")

with tab5:
    st.header("Water Quality Analysis")
    
    # Check if water quality columns exist
    water_quality_cols = ['DissolvedOxygen', 'pH', 'ElectricalConductance']
    available_wq_cols = [col for col in water_quality_cols if col in filtered_df.columns]
    
    if available_wq_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # pH vs Water Usage
            if 'pH' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df, 
                    x='pH', 
                    y='water_usage_per_capita',
                    color='hydrologic_region',
                    title="pH vs Water Usage",
                    labels={'pH': 'pH', 'water_usage_per_capita': 'Water Usage per Capita (gallons)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dissolved Oxygen vs Water Usage
            if 'DissolvedOxygen' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df, 
                    x='DissolvedOxygen', 
                    y='water_usage_per_capita',
                    color='hydrologic_region',
                    title="Dissolved Oxygen vs Water Usage",
                    labels={'DissolvedOxygen': 'Dissolved Oxygen (mg/L)', 'water_usage_per_capita': 'Water Usage per Capita (gallons)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Water quality by region
        regional_wq = filtered_df.groupby('hydrologic_region')[available_wq_cols].mean().reset_index()
        
        # Melt for plotting
        regional_wq_melt = pd.melt(
            regional_wq, 
            id_vars=['hydrologic_region'], 
            value_vars=available_wq_cols,
            var_name='Water Quality Metric', 
            value_name='Value'
        )
        
        fig = px.bar(
            regional_wq_melt,
            x='hydrologic_region',
            y='Value',
            color='Water Quality Metric',
            barmode='group',
            title="Water Quality Metrics by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Water quality correlation with usage
        st.subheader("Water Quality Impact on Usage")
        wq_corr_cols = available_wq_cols + ['water_usage_per_capita', 'percent_non_revenue']
        corr_matrix = filtered_df[wq_corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Between Water Quality and Usage Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No water quality data available in the current dataset.")

with tab6:
    st.header("Water Usage Prediction Tool")
    
    if model is not None and preprocessor is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            
            population = st.slider("Population Served", 1000, 1000000, 100000, step=1000)
            
            # Use the drought_value and hydrological region from the sidebar
            water_shortage_level = drought_value            
            climate_zone = st.selectbox("Climate Zone", [1, 2, 3, 4])
            
            month = st.selectbox("Month", list(range(1, 13)), 
                                format_func=lambda x: datetime(2023, x, 1).strftime('%B'))
            
            current_year = 2023
            year = st.selectbox(
                "Year", 
                options=list(range(2014, 2025)),
                index=list(range(2014, 2025)).index(current_year)
            )

            # Add reservoir storage slider in gallons
            reservoir_storage = st.slider(
                "Reservoir Storage (gallons)", 
                min_value=0, 
                max_value=10000000,  # 10 million gallons
                value=5000000,       # 5 million gallons default
                step=100000,         # 100,000 gallon steps
                format="%d",         # Format as integer
                help="Volume of water currently stored in reservoir (in gallons)"
            )

            st.subheader("Climate Variables")

            # Palmer Hydrological Index
            palmer_hydro = st.slider("Palmer Hydrological Index", 
                                   min_value=-6.0, 
                                   max_value=6.0, 
                                   value=0.0,
                                   help="Drought severity index: negative values indicate drought conditions, positive values indicate wet conditions")
            
            # Precipitation
            precip = st.slider("Precipitation (inches)", 
                             min_value=0.0, 
                             max_value=20.0, 
                             value=2.0,
                             help="Monthly precipitation amount in inches")
            
            # ETo_mm (Evapotranspiration)
            eto_mm = st.slider("Evapotranspiration (mm)", 
                              min_value=0.0, 
                              max_value=300.0, 
                              value=150.0,
                              help="Reference evapotranspiration - water loss from soil evaporation and plant transpiration")
            
            # Palmer Z-index
            palmer_z_index = st.slider("Palmer Z-Index", 
                                     min_value=-5.0, 
                                     max_value=5.0, 
                                     value=0.0,
                                     help="Short-term drought index that responds quickly to current conditions")
            
            st.subheader("Water Quality Variables")
            # Electrical Conductance
            electrical_conductance = st.slider("Electrical Conductance (μS/cm)", 
                                             min_value=0, 
                                             max_value=2000, 
                                             value=500,
                                             help="Measure of water's ability to conduct electricity, indicates dissolved solids")
            
            # Dissolved Oxygen
            dissolved_oxygen = st.slider("Dissolved Oxygen (mg/L)", 
                                       min_value=0.0, 
                                       max_value=15.0, 
                                       value=8.0,
                                       help="Amount of oxygen dissolved in water, important for aquatic life")
            
            # pH Level
            ph = st.slider("pH Level", 
                         min_value=6.0, 
                         max_value=9.0, 
                         value=7.0,
                         help="Water pH level - 7 is neutral, below 7 is acidic, above 7 is alkaline")
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'total_population_served': [population],
                'county_under_drought_declaration': [drought_value],
                'climate_zone': [climate_zone],
                'month': [month],
                'year': [year],
                'water_shortage_level_indicator': [water_shortage_level],
                'reservoir_storage': [reservoir_storage],
                'reported_recycled_water': [0], # Placeholder for recycled water
                'percent_non_revenue': [12], # Placeholder for non-revenue water
                'hydrologic_region': [hydrologic_region],
                'ETo_mm': [eto_mm],
                'palmer_z_index': [palmer_z_index],
                'ElectricalConductance': [electrical_conductance],
                'DissolvedOxygen': [dissolved_oxygen],
                'palmer_hydro': [palmer_hydro],
                'pH': [ph],
                'precip': [precip]
            })

            # Add temperature values from sidebar if available
            if 'temp_range' in locals() and 'avg_temp' in locals():
                # Add all three temperature metrics to input data
                input_data['min_temp'] = [temp_range[0]]
                input_data['max_temp'] = [temp_range[1]]
                input_data['avg_temp'] = [avg_temp]
            
            if st.button("Predict Water Usage"):
                # Check if model can handle all input columns
                model_features = []
                try:
                    with open('model_features.txt', 'r') as f:
                        model_features = f.read().splitlines()
                except:
                    pass
                
                # Filter input_data to only include columns the model knows about
                if model_features:
                    input_data = input_data[[col for col in input_data.columns if col in model_features]]
                
                # Preprocess the input data
                try:
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

                        # Add climate factors if they were used in prediction
                        if 'palmer_hydro' in input_data.columns:
                            palmer_value = input_data['palmer_hydro'].iloc[0]
                            if palmer_value < -2:
                                factors.append(f"Severe drought conditions (Palmer Index: {palmer_value:.1f}) typically increase conservation efforts")
                            elif palmer_value > 2:
                                factors.append(f"Wet conditions (Palmer Index: {palmer_value:.1f}) may reduce conservation pressure")

                        if 'precip' in input_data.columns:
                            precip_value = input_data['precip'].iloc[0]
                            if precip_value < 1:
                                factors.append(f"Low precipitation ({precip_value:.1f} inches) typically increases irrigation needs")
                            elif precip_value > 5:
                                factors.append(f"High precipitation ({precip_value:.1f} inches) typically reduces outdoor water usage")

                        # Add water quality factors if they were used in prediction
                        if 'pH' in input_data.columns:
                            ph_value = input_data['pH'].iloc[0]
                            if ph_value < 6.5 or ph_value > 8.5:
                                factors.append(f"pH level ({ph_value:.1f}) outside optimal range may affect water quality perception")

                        if 'DissolvedOxygen' in input_data.columns:
                            do_value = input_data['DissolvedOxygen'].iloc[0]
                            if do_value < 5:
                                factors.append(f"Low dissolved oxygen ({do_value:.1f} mg/L) may indicate water quality concerns")

                        if 'ElectricalConductance' in input_data.columns:
                            ec_value = input_data['ElectricalConductance'].iloc[0]
                            if ec_value > 1000:
                                factors.append(f"High electrical conductance ({ec_value:.0f} μS/cm) indicates high mineral content")

                        if 'ETo_mm' in input_data.columns:
                            eto_value = input_data['ETo_mm'].iloc[0]
                            if eto_value > 150:
                                factors.append(f"High evapotranspiration ({eto_value:.1f} mm) increases outdoor water demand")

                        for factor in factors:
                            st.markdown(f"- {factor}")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("The model may not support all the input features. Try using the original model without climate variables.")
    else:
        st.info("Prediction model not available. Please ensure model files are properly loaded.")

with tab7:
    st.header("Data Explorer")
    
    # Add clustering option
    if st.checkbox("Perform Cluster Analysis"):
        # Select features for clustering
        cluster_options = ['water_usage_per_capita', 'total_population_served', 'percent_non_revenue', 'final_percent_residential_use']
        
        # Add climate and water quality options if available
        if 'precip' in filtered_df.columns:
            cluster_options.append('precip')
        if 'max_temp' in filtered_df.columns:
            cluster_options.append('max_temp')
        if 'palmer_hydro' in filtered_df.columns:
            cluster_options.append('palmer_hydro')
        if 'pH' in filtered_df.columns:
            cluster_options.append('pH')
        if 'DissolvedOxygen' in filtered_df.columns:
            cluster_options.append('DissolvedOxygen')
        
        selected_features = st.multiselect(
            "Select features for clustering",
            options=cluster_options,
            default=['water_usage_per_capita', 'total_population_served']
        )
        
        if selected_features:
            n_clusters = st.slider("Number of clusters", 2, 6, 4)
            
            # Prepare data for clustering
            cluster_data = filtered_df[selected_features].dropna()
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to the data
            cluster_data['cluster'] = cluster_labels
            
            # Display cluster visualization
            if len(selected_features) >= 2:
                fig = px.scatter(
                    cluster_data,
                    x=selected_features[0],
                    y=selected_features[1],
                    color='cluster',
                    title=f"Clusters by {selected_features[0]} and {selected_features[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster profiles
            st.subheader("Cluster Profiles")
            cluster_profiles = cluster_data.groupby('cluster')[selected_features].mean()
            st.dataframe(cluster_profiles)
            
            # Visualize cluster characteristics
            st.subheader("Cluster Characteristics")
            
            # Create a radar chart for cluster profiles
            fig = go.Figure()
            
            # Normalize the data for radar chart
            normalized_profiles = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
            
            for cluster_id in range(n_clusters):
                fig.add_trace(go.Scatterpolar(
                    r=normalized_profiles.loc[cluster_id].values,
                    theta=normalized_profiles.columns,
                    fill='toself',
                    name=f'Cluster {cluster_id}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Normalized Cluster Profiles"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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