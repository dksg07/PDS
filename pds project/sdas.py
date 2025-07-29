import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# --- 1. Simulate Data ---
# Let's create hypothetical data for infrastructure, traffic, and weather.

def generate_infrastructure_data(num_assets=100):
    """
    Generates realistic infrastructure data based on Indian infrastructure patterns.
    
    Creates a mix of bridges, road segments, metro stations, and tunnels with
    realistic naming conventions, geographic distribution, and characteristics
    reflecting actual Indian infrastructure.
    """
    # Real-world asset naming patterns
    highway_prefixes = ['NH48', 'NH44', 'NH19', 'NH16', 'NH66', 'GQ', 'NH24']
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    bridge_types = ['Yamuna_Bridge', 'Narmada_Bridge', 'Sealink', 'Howrah_Bridge', 'Flyover']
    metro_lines = ['Blue_Line', 'Yellow_Line', 'Red_Line', 'Green_Line', 'Purple_Line']
    
    asset_ids = []
    asset_types = []
    
    # Generate realistic asset IDs and types
    for i in range(num_assets):
        asset_type = np.random.choice(['Bridge', 'Road Segment', 'Metro Station', 'Tunnel'], 
                                    p=[0.3, 0.5, 0.15, 0.05])
        
        if asset_type == 'Bridge':
            city = np.random.choice(cities)
            bridge_type = np.random.choice(bridge_types)
            asset_id = f"{city}_{bridge_type}_{i:03d}"
        elif asset_type == 'Road Segment':
            highway = np.random.choice(highway_prefixes)
            km = np.random.randint(10, 500)
            asset_id = f"{highway}_Km_{km}_{i:03d}"
        elif asset_type == 'Metro Station':
            city = np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
            line = np.random.choice(metro_lines)
            asset_id = f"{city}_Metro_{line}_{i:03d}"
        else:  # Tunnel
            city = np.random.choice(cities)
            asset_id = f"{city}_Tunnel_{i:03d}"
        
        asset_ids.append(asset_id)
        asset_types.append(asset_type)
    
    data = {
        'asset_id': asset_ids,
        'type': asset_types,
        'age_years': np.random.randint(5, 70, num_assets),
        'construction_material': np.random.choice(['Concrete', 'Steel', 'Asphalt', 'Composite'], num_assets),
        'last_inspection_date': [
            (datetime.now() - timedelta(days=random.randint(30, 1000))).strftime('%Y-%m-%d')
            for _ in range(num_assets)
        ],
        'condition_rating_initial': np.random.randint(70, 100, num_assets), # Initial condition (e.g., 1-100 scale, 100 is perfect)
        'latitude': np.random.uniform(8.0, 37.0, num_assets), # India's latitude range
        'longitude': np.random.uniform(68.0, 97.0, num_assets) # India's longitude range
    }
    df = pd.DataFrame(data)

    # Simulate deterioration over time
    df['condition_rating_current'] = df.apply(
        lambda row: max(10, row['condition_rating_initial'] - (row['age_years'] * np.random.uniform(0.5, 1.5))),
        axis=1
    )
    # Introduce some random noise/variability
    df['condition_rating_current'] = df['condition_rating_current'].apply(
        lambda x: max(10, min(100, x + np.random.normal(0, 5)))
    ).astype(int)
    return df

def generate_traffic_data(infrastructure_df):
    """
    Generates realistic traffic data based on asset type and location.
    
    Traffic volumes are based on actual Indian highway and urban traffic patterns:
    - National Highways: 20,000-100,000 vehicles/day
    - Urban Roads: 50,000-150,000 vehicles/day  
    - Metro Stations: 50,000-400,000 passengers/day
    - Bridges: Varies by location and importance
    """
    traffic_volumes = []
    
    for _, row in infrastructure_df.iterrows():
        asset_type = row['type']
        asset_id = row['asset_id']
        
        if asset_type == 'Road Segment':
            # Highway segments have high traffic
            if any(highway in asset_id for highway in ['NH48', 'NH44', 'NH19', 'GQ']):
                volume = random.randint(30000, 100000)  # National highways
            else:
                volume = random.randint(15000, 60000)   # State highways
        elif asset_type == 'Bridge':
            # Major bridges have higher traffic
            if any(city in asset_id for city in ['Mumbai', 'Delhi', 'Kolkata']):
                volume = random.randint(40000, 120000)  # Major city bridges
            else:
                volume = random.randint(15000, 50000)   # Smaller bridges
        elif asset_type == 'Metro Station':
            # Metro stations measured in passengers
            if 'Delhi' in asset_id or 'Mumbai' in asset_id:
                volume = random.randint(100000, 400000)  # Major metro systems
            else:
                volume = random.randint(25000, 150000)   # Newer metro systems
        else:  # Tunnel
            volume = random.randint(20000, 80000)
        
        traffic_volumes.append(volume)
    
    data = {
        'asset_id': infrastructure_df['asset_id'].tolist(),
        'traffic_volume_daily': traffic_volumes
    }
    return pd.DataFrame(data)

def generate_weather_data(infrastructure_df):
    """
    Generates realistic weather exposure data based on Indian climate patterns.
    
    Weather factors vary significantly across India:
    - North India: High temperature variation, dust storms, moderate rain
    - West Coast: High humidity, heavy monsoons, salt corrosion
    - South India: Moderate climate, seasonal rains
    - East India: High humidity, cyclones, heavy rains
    - Mountain regions: Freeze-thaw cycles, landslides
    """
    weather_data = []
    
    for _, row in infrastructure_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        asset_id = row['asset_id']
        
        # Determine climate zone based on latitude and asset location
        if lat > 30:  # North India
            freeze_thaw = np.random.randint(15, 35)
            heavy_rain_days = np.random.randint(25, 45)
        elif lat < 15:  # South India
            freeze_thaw = np.random.randint(0, 5)
            heavy_rain_days = np.random.randint(40, 80)
        elif 'Mumbai' in asset_id or 'Goa' in asset_id:  # West Coast
            freeze_thaw = np.random.randint(0, 2)
            heavy_rain_days = np.random.randint(60, 120)
        elif 'Kolkata' in asset_id or lon > 85:  # East India
            freeze_thaw = np.random.randint(0, 8)
            heavy_rain_days = np.random.randint(55, 95)
        else:  # Central India
            freeze_thaw = np.random.randint(5, 20)
            heavy_rain_days = np.random.randint(30, 60)
        
        weather_data.append({
            'asset_id': row['asset_id'],
            'freeze_thaw_cycles_avg_annual': freeze_thaw,
            'heavy_rain_days_avg_annual': heavy_rain_days
        })
    
    return pd.DataFrame(weather_data)

# Generate realistic infrastructure datasets
print("=== SMART INFRASTRUCTURE ASSET MANAGEMENT SYSTEM ===")
print("Generating realistic Indian infrastructure data...")
print("Assets include: National Highways, City Bridges, Metro Networks, Urban Roads")
print()

infrastructure_df = generate_infrastructure_data()
traffic_df = generate_traffic_data(infrastructure_df)
weather_df = generate_weather_data(infrastructure_df)

print("--- Sample Infrastructure Assets ---")
print(infrastructure_df[['asset_id', 'type', 'age_years', 'construction_material', 'condition_rating_current']].head())
print("\n--- Traffic Volume Analysis ---")
traffic_summary = traffic_df.groupby('asset_id')['traffic_volume_daily'].first()
print(f"Average daily traffic: {traffic_summary.mean():.0f} vehicles/passengers")
print(f"Highest traffic asset: {traffic_summary.idxmax()} - {traffic_summary.max():,} per day")
print("\n--- Weather Exposure Analysis ---")
print(weather_df[['asset_id', 'freeze_thaw_cycles_avg_annual', 'heavy_rain_days_avg_annual']].head())

# --- 2. Data Cleaning & Standardization ---
# For this simulated data, cleaning is minimal, but we'll ensure correct types.
# Merge all dataframes into a single master dataframe.

# Convert last_inspection_date to datetime objects
infrastructure_df['last_inspection_date'] = pd.to_datetime(infrastructure_df['last_inspection_date'])

# Calculate days since last inspection
infrastructure_df['days_since_last_inspection'] = (datetime.now() - infrastructure_df['last_inspection_date']).dt.days

# Merge dataframes
df_master = infrastructure_df.merge(traffic_df, on='asset_id', how='left')
df_master = df_master.merge(weather_df, on='asset_id', how='left')

print("\n--- Merged and Cleaned Data (first 5 rows) ---")
print(df_master.head())
print(df_master.info())

# --- 3. Time-Series Analysis (Simplified) ---
# We don't have true time-series data for condition changes per asset,
# but we can analyze how condition *tends* to change with age.
# Let's group by age and material to see average deterioration.

print("\n--- Average Condition Rating by Age Group and Material ---")
df_master['age_group'] = pd.cut(df_master['age_years'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                                labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+'])
avg_condition_by_age_material = df_master.groupby(['age_group', 'construction_material'])['condition_rating_current'].mean().unstack()
print(avg_condition_by_age_material)

# --- 4. Regression Analysis (Basic) using NumPy ---
# Model the relationship between age, weather, traffic, and deterioration.
# We'll predict 'condition_rating_current' based on other features using basic linear regression (Normal Equation).

features = ['age_years', 'traffic_volume_daily', 'freeze_thaw_cycles_avg_annual', 'heavy_rain_days_avg_annual']
target = 'condition_rating_current'

# Prepare data for regression
X = df_master[features].values
y = df_master[target].values

# Add a bias (intercept) term to X
X_b = np.c_[np.ones((len(X), 1)), X] # Add x0 = 1 to each instance

# Calculate coefficients using the Normal Equation: beta = (X_T * X)^-1 * X_T * y
try:
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    intercept = theta[0]
    coefficients = theta[1:]

    # Make predictions (for the entire dataset for simplicity in this example)
    y_pred = X_b @ theta

    # Calculate Mean Squared Error manually
    mse = np.mean((y_pred - y)**2)

    print(f"\n--- Regression Analysis Results (NumPy) ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Model Intercept: {intercept:.2f}")
    print(f"Model Coefficients (for {features}): {coefficients}")

    # Interpretation of coefficients:
    # A negative coefficient for 'age_years' would indicate that as age increases, condition decreases (as expected).
    # Similarly for traffic and weather factors.
except np.linalg.LinAlgError:
    print("\n--- Regression Analysis Results (NumPy) ---")
    print("Could not perform linear regression: Singular matrix encountered. This might happen with highly correlated features or insufficient data.")


# --- 5. Prioritization Algorithm ---
# Prioritize repairs based on a multi-criteria scoring system.
# Factors: Condition (lower is worse), Age (higher is worse), Traffic Volume (higher means more critical),
# Days Since Last Inspection (higher means more urgent).

# Normalize factors to a 0-1 scale (or similar) to combine them
df_master['normalized_condition'] = 1 - (df_master['condition_rating_current'] / 100) # 0=perfect, 1=worst
df_master['normalized_age'] = df_master['age_years'] / df_master['age_years'].max()
df_master['normalized_traffic'] = df_master['traffic_volume_daily'] / df_master['traffic_volume_daily'].max()
df_master['normalized_days_since_inspection'] = df_master['days_since_last_inspection'] / df_master['days_since_last_inspection'].max()

# Define weights for each factor (these can be adjusted based on policy)
# Example: Condition is most important, then traffic, then age, then inspection urgency.
weights = {
    'condition': 0.4,
    'age': 0.2,
    'traffic': 0.3,
    'days_since_inspection': 0.1
}

df_master['priority_score'] = (
    df_master['normalized_condition'] * weights['condition'] +
    df_master['normalized_age'] * weights['age'] +
    df_master['normalized_traffic'] * weights['traffic'] +
    df_master['normalized_days_since_inspection'] * weights['days_since_inspection']
)

# Higher score means higher priority
df_master = df_master.sort_values(by='priority_score', ascending=False)

print("\n--- Top 10 Prioritized Infrastructure Assets ---")
print(df_master[['asset_id', 'type', 'age_years', 'condition_rating_current', 'traffic_volume_daily', 'priority_score']].head(10))

# --- 6. Matplotlib Visualizations ---

plt.style.use('seaborn-v0_8-darkgrid') # Modern looking plots

# Scatter Plot: Condition Rating vs. Age, colored by Material Type
plt.figure(figsize=(12, 7))
for material in df_master['construction_material'].unique():
    subset = df_master[df_master['construction_material'] == material]
    plt.scatter(subset['age_years'], subset['condition_rating_current'], label=material, alpha=0.7, s=50)
plt.title('Condition Rating vs. Age by Construction Material', fontsize=16)
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Current Condition Rating (1-100)', fontsize=12)
plt.legend(title='Material')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart: Compare average deterioration rates across different materials
# (Deterioration = Initial Condition - Current Condition)
df_master['deterioration'] = df_master['condition_rating_initial'] - df_master['condition_rating_current']
avg_deterioration_by_material = df_master.groupby('construction_material')['deterioration'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
avg_deterioration_by_material.plot(kind='bar', color='skyblue')
plt.title('Average Deterioration by Construction Material', fontsize=16)
plt.xlabel('Construction Material', fontsize=12)
plt.ylabel('Average Deterioration (Points)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Histogram: Show the distribution of condition ratings
plt.figure(figsize=(10, 6))
plt.hist(df_master['condition_rating_current'], bins=10, edgecolor='black', alpha=0.7, color='lightcoral')
plt.title('Distribution of Current Condition Ratings', fontsize=16)
plt.xlabel('Condition Rating (1-100)', fontsize=12)
plt.ylabel('Number of Assets', fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Bar Chart: Top 10 Assets by Priority Score
plt.figure(figsize=(12, 7))
top_10_priority = df_master.head(10).set_index('asset_id')
top_10_priority['priority_score'].plot(kind='barh', color='darkorange')
plt.title('Top 10 Infrastructure Assets by Priority Score', fontsize=16)
plt.xlabel('Priority Score (Higher = More Urgent)', fontsize=12)
plt.ylabel('Asset ID', fontsize=12)
plt.gca().invert_yaxis() # Highest priority at the top
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# --- 7. Actionable Insights ---
print("\n--- Actionable Insights ---")
print(f"Total assets analyzed: {len(df_master)}")
print(f"Average current condition rating: {df_master['condition_rating_current'].mean():.2f}")
print(f"Material with highest average deterioration: {avg_deterioration_by_material.index[0]} (Avg Deterioration: {avg_deterioration_by_material.iloc[0]:.2f} points)")

# Identify assets below a certain condition threshold
critical_condition_threshold = 50
critical_assets = df_master[df_master['condition_rating_current'] < critical_condition_threshold]
print(f"\nNumber of assets with condition below {critical_condition_threshold}: {len(critical_assets)}")
if not critical_assets.empty:
    print("Consider immediate inspection/repair for these assets:")
    print(critical_assets[['asset_id', 'type', 'condition_rating_current', 'priority_score']].head())

# Suggest maintenance based on last inspection date for high priority items
inspection_due_threshold_days = 365 * 2 # 2 years
high_priority_needing_inspection = df_master[
    (df_master['priority_score'] > df_master['priority_score'].quantile(0.8)) & # Top 20% priority
    (df_master['days_since_last_inspection'] > inspection_due_threshold_days)
]
print(f"\nHigh priority assets needing inspection (last inspected > {inspection_due_threshold_days} days ago): {len(high_priority_needing_inspection)}")
if not high_priority_needing_inspection.empty:
    print("Prioritize inspections for these assets:")
    print(high_priority_needing_inspection[['asset_id', 'type', 'condition_rating_current', 'days_since_last_inspection', 'priority_score']].head())