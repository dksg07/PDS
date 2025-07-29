# 🚀 Installation & Setup Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python Version**: 3.8 or higher
- **RAM**: 4 GB minimum (8 GB recommended for large datasets)
- **Storage**: 2 GB free space for installation and data
- **Internet Connection**: Required for package installation

### Recommended Requirements
- **RAM**: 16 GB for processing 10,000+ assets
- **CPU**: Multi-core processor for parallel processing
- **Storage**: SSD for faster data processing
- **Display**: 1920x1080 resolution for optimal chart viewing

## 📦 Installation Steps

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://python.org)
2. Run the installer and check "Add Python to PATH"
3. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Step 2: Create Project Directory

```bash
# Create project folder
mkdir infrastructure-analysis
cd infrastructure-analysis

# Download or clone the project files
# Copy sdas.py and documentation files to this directory
```

### Step 3: Install Required Packages

```bash
# Install core dependencies
pip install pandas numpy matplotlib seaborn

# Optional: Install additional packages for advanced features
pip install scikit-learn jupyter plotly dash

# For database connectivity (optional)
pip install sqlalchemy psycopg2-binary

# For geospatial analysis (optional)  
pip install geopandas folium
```

### Step 4: Verify Installation

Create a test file `test_installation.py`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("✅ All packages installed successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
```

Run the test:
```bash
python test_installation.py
```

## 🗂️ Project Structure Setup

### Recommended Folder Structure
```
infrastructure-analysis/
├── data/
│   ├── raw/
│   │   ├── infrastructure.csv
│   │   ├── traffic.csv
│   │   └── weather.csv
│   ├── processed/
│   └── exports/
├── scripts/
│   ├── sdas.py
│   ├── data_processor.py
│   └── visualizer.py
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── dashboards/
├── config/
│   └── config.json
├── docs/
│   ├── README.md
│   ├── USER_GUIDE.md
│   ├── TECHNICAL_DOCS.md
│   └── REAL_WORLD_DATA.md
└── logs/
    └── analysis.log
```

### Create Folder Structure

#### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path data\raw, data\processed, data\exports, scripts, outputs\charts, outputs\reports, outputs\dashboards, config, docs, logs
```

#### macOS/Linux
```bash
mkdir -p data/{raw,processed,exports} scripts outputs/{charts,reports,dashboards} config docs logs
```

## ⚙️ Configuration Setup

### Create Configuration File

Create `config/config.json`:

```json
{
    "priority_weights": {
        "condition": 0.40,
        "age": 0.20,
        "traffic": 0.30,
        "inspection_urgency": 0.10
    },
    "condition_thresholds": {
        "critical": 30,
        "poor": 50,
        "fair": 70,
        "good": 85
    },
    "analysis_parameters": {
        "max_age_years": 100,
        "max_traffic_daily": 200000,
        "max_inspection_days": 1095
    },
    "data_sources": {
        "infrastructure_file": "data/raw/infrastructure.csv",
        "traffic_file": "data/raw/traffic.csv",
        "weather_file": "data/raw/weather.csv"
    },
    "output_settings": {
        "chart_format": "png",
        "chart_dpi": 300,
        "report_format": "pdf"
    },
    "regional_settings": {
        "country": "India",
        "currency": "INR",
        "date_format": "%Y-%m-%d",
        "timezone": "Asia/Kolkata"
    }
}
```

## 📊 Data Setup

### Option 1: Use Demo Data (Recommended for Testing)

The system includes built-in data generation. Simply run:

```bash
python sdas.py
```

### Option 2: Use Your Own Data

#### Prepare Infrastructure Data (`data/raw/infrastructure.csv`)

```csv
asset_id,type,age_years,construction_material,condition_rating_initial,condition_rating_current,latitude,longitude,last_inspection_date
NH48_Bridge_001,Bridge,25,Concrete,90,72,28.6139,77.2090,2023-11-15
NH44_Road_Km156,Road Segment,18,Bituminous,88,65,12.8230,79.1568,2024-01-22
Delhi_Metro_Blue_01,Metro Station,20,RCC,92,75,28.6328,77.2197,2024-02-28
```

#### Prepare Traffic Data (`data/raw/traffic.csv`)

```csv
asset_id,traffic_volume_daily,heavy_vehicle_percentage,peak_hour_volume
NH48_Bridge_001,85000,15,12500
NH44_Road_Km156,55000,32,8250
Delhi_Metro_Blue_01,350000,0,65000
```

#### Prepare Weather Data (`data/raw/weather.csv`)

```csv
asset_id,freeze_thaw_cycles_avg_annual,heavy_rain_days_avg_annual,temperature_max_avg,humidity_avg
NH48_Bridge_001,25,35,42,65
NH44_Road_Km156,0,48,36,68
Delhi_Metro_Blue_01,22,40,42,63
```

## 🏃‍♂️ Running the System

### Basic Usage

```bash
# Run with demo data
python sdas.py

# Run with custom data files
python sdas.py --infrastructure data/raw/infrastructure.csv --traffic data/raw/traffic.csv --weather data/raw/weather.csv
```

### Advanced Usage

#### Custom Configuration
```bash
python sdas.py --config config/custom_config.json
```

#### Specific Analysis
```bash
# Analyze only bridges
python sdas.py --filter-type Bridge

# Analyze assets in specific region
python sdas.py --filter-region "Mumbai"

# Generate only priority analysis
python sdas.py --analysis priority
```

#### Output Control
```bash
# Save charts to specific directory
python sdas.py --output-dir outputs/monthly_analysis/

# Export results to Excel
python sdas.py --export-format excel

# Generate dashboard
python sdas.py --dashboard
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **Import Error: No module named 'pandas'**
```bash
# Solution: Install missing packages
pip install pandas numpy matplotlib seaborn
```

#### 2. **Memory Error with Large Datasets**
```python
# Add to your script for large datasets
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Process in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

#### 3. **Matplotlib Display Issues**

##### Windows
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

##### Linux (without GUI)
```python
import matplotlib
matplotlib.use('Agg')  # For headless systems
import matplotlib.pyplot as plt
```

#### 4. **Date Parsing Errors**
```python
# Fix date format issues
df['last_inspection_date'] = pd.to_datetime(
    df['last_inspection_date'], 
    format='%Y-%m-%d',
    errors='coerce'  # Convert invalid dates to NaT
)
```

#### 5. **Performance Issues**
```python
# Optimize for large datasets
import warnings
warnings.filterwarnings('ignore')

# Use efficient data types
df['asset_id'] = df['asset_id'].astype('category')
df['age_years'] = pd.to_numeric(df['age_years'], downcast='integer')
```

### System Performance Monitoring

Create `scripts/performance_monitor.py`:

```python
import time
import psutil
import pandas as pd

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        print(f"Function: {func.__name__}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        print("-" * 50)
        
        return result
    return wrapper

# Usage example
@monitor_performance
def analyze_infrastructure(df):
    # Your analysis code here
    return df.groupby('type')['condition_rating_current'].mean()
```

## 🔐 Security Considerations

### Data Protection

1. **Sensitive Data Handling**
   ```python
   # Anonymize asset IDs for sharing
   import hashlib
   
   def anonymize_data(df):
       df['asset_id_hash'] = df['asset_id'].apply(
           lambda x: hashlib.sha256(x.encode()).hexdigest()[:10]
       )
       return df.drop('asset_id', axis=1)
   ```

2. **Access Control**
   ```python
   # Log data access
   import logging
   
   logging.basicConfig(
       filename='logs/access.log',
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s'
   )
   
   def log_data_access(user, action, data_type):
       logging.info(f"User: {user}, Action: {action}, Data: {data_type}")
   ```

3. **Environment Variables**
   ```bash
   # Store sensitive configuration in environment variables
   export DB_PASSWORD="your_password"
   export API_KEY="your_api_key"
   ```

## 📱 Integration Options

### Database Integration

#### SQLite (Recommended for small to medium datasets)
```python
import sqlite3
import pandas as pd

# Create database connection
conn = sqlite3.connect('infrastructure.db')

# Save DataFrame to database
df.to_sql('infrastructure_assets', conn, if_exists='replace', index=False)

# Read from database
df = pd.read_sql_query("SELECT * FROM infrastructure_assets", conn)
```

#### PostgreSQL (For large enterprise datasets)
```python
from sqlalchemy import create_engine
import pandas as pd

# Create connection
engine = create_engine('postgresql://user:password@localhost:5432/infrastructure_db')

# Save and read data
df.to_sql('assets', engine, if_exists='replace', index=False)
df = pd.read_sql('SELECT * FROM assets WHERE condition_rating_current < 50', engine)
```

### API Integration

```python
import requests
import pandas as pd

def fetch_traffic_data(api_key, asset_ids):
    """Fetch real-time traffic data from API"""
    traffic_data = []
    
    for asset_id in asset_ids:
        response = requests.get(
            f"https://api.traffic-service.com/data/{asset_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        if response.status_code == 200:
            traffic_data.append(response.json())
    
    return pd.DataFrame(traffic_data)
```

### Web Dashboard Integration

#### Using Streamlit
```bash
pip install streamlit

# Create dashboard.py
streamlit run dashboard.py
```

#### Using Dash
```bash
pip install dash plotly

# Create interactive dashboard
python dashboard_app.py
```

## 📚 Next Steps

### 1. **Learn the System**
- Review the [User Guide](USER_GUIDE.md)
- Understand the [Technical Documentation](TECHNICAL_DOCS.md)
- Explore [Real-World Examples](REAL_WORLD_DATA.md)

### 2. **Customize for Your Needs**
- Modify priority weights in configuration
- Add custom asset types
- Integrate with your existing systems

### 3. **Scale Up**
- Set up database connections
- Implement automated data pipelines
- Deploy web dashboards

### 4. **Advanced Features**
- Machine learning integration
- IoT sensor data processing
- Predictive maintenance algorithms

## 📞 Support

### Getting Help
- **Documentation**: Check all `.md` files in the `docs/` folder
- **Code Issues**: Review error messages and logs
- **Performance**: Use the performance monitoring tools
- **Data Issues**: Validate your data format against examples

### Community Resources
- **GitHub Issues**: Report bugs and request features
- **Stack Overflow**: Tag questions with `infrastructure-analysis`
- **Email Support**: technical-support@your-domain.com

---

You're now ready to start using the Smart Infrastructure Asset Management System! Begin with the demo data to familiarize yourself with the system, then gradually integrate your own data sources.
