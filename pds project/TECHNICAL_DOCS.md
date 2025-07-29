# 🔧 Technical Documentation: Infrastructure Asset Management System

## 📋 System Overview

### Architecture Components
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Layer        │    │   Processing Layer  │    │   Presentation Layer│
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ • CSV/Excel Files   │    │ • Pandas DataFrames │    │ • Matplotlib Charts │
│ • Database Conn.    │    │ • NumPy Calculations│    │ • Console Reports   │
│ • API Endpoints     │    │ • ML Algorithms     │    │ • Export Functions  │
│ • IoT Sensors       │    │ • Priority Engine   │    │ • Dashboard Views   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Technology Stack
- **Core Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (optional)
- **Database**: SQLite/PostgreSQL (expandable)
- **Web Framework**: Flask/Django (future enhancement)

## 🗄️ Data Model

### Core Entity Relationships
```sql
-- Infrastructure Assets (Primary table)
CREATE TABLE infrastructure_assets (
    asset_id VARCHAR(50) PRIMARY KEY,
    asset_type VARCHAR(30) NOT NULL,
    age_years INTEGER,
    construction_material VARCHAR(30),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    condition_rating_initial INTEGER,
    condition_rating_current INTEGER,
    last_inspection_date DATE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Traffic Data (Related table)
CREATE TABLE traffic_data (
    asset_id VARCHAR(50) REFERENCES infrastructure_assets(asset_id),
    traffic_volume_daily INTEGER,
    heavy_vehicle_percentage DECIMAL(5,2),
    peak_hour_volume INTEGER,
    measurement_date DATE,
    data_source VARCHAR(50)
);

-- Weather Data (Related table)
CREATE TABLE weather_data (
    asset_id VARCHAR(50) REFERENCES infrastructure_assets(asset_id),
    freeze_thaw_cycles_annual INTEGER,
    heavy_rain_days_annual INTEGER,
    temperature_max_avg DECIMAL(5,2),
    temperature_min_avg DECIMAL(5,2),
    humidity_avg DECIMAL(5,2),
    year INTEGER
);

-- Maintenance History (Tracking table)
CREATE TABLE maintenance_history (
    maintenance_id SERIAL PRIMARY KEY,
    asset_id VARCHAR(50) REFERENCES infrastructure_assets(asset_id),
    maintenance_type VARCHAR(50),
    maintenance_date DATE,
    cost_amount DECIMAL(12,2),
    condition_before INTEGER,
    condition_after INTEGER,
    contractor VARCHAR(100),
    notes TEXT
);
```

### Data Validation Schema
```python
import pandas as pd
from typing import Dict, List, Optional

class AssetDataValidator:
    """Validates infrastructure asset data integrity"""
    
    REQUIRED_COLUMNS = {
        'infrastructure': [
            'asset_id', 'type', 'age_years', 'construction_material',
            'condition_rating_current', 'last_inspection_date'
        ],
        'traffic': ['asset_id', 'traffic_volume_daily'],
        'weather': ['asset_id', 'freeze_thaw_cycles_avg_annual', 'heavy_rain_days_avg_annual']
    }
    
    VALID_RANGES = {
        'condition_rating_current': (0, 100),
        'age_years': (0, 150),
        'traffic_volume_daily': (0, 500000),
        'freeze_thaw_cycles_avg_annual': (0, 365),
        'heavy_rain_days_avg_annual': (0, 365)
    }
    
    VALID_CATEGORIES = {
        'type': ['Bridge', 'Road Segment', 'Tunnel', 'Flyover', 'Underpass'],
        'construction_material': ['Concrete', 'Steel', 'Asphalt', 'Composite', 'Stone']
    }
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, data_type: str) -> Dict[str, List[str]]:
        """Validates DataFrame against schema rules"""
        errors = {'missing_columns': [], 'invalid_values': [], 'data_quality': []}
        
        # Check required columns
        required_cols = AssetDataValidator.REQUIRED_COLUMNS.get(data_type, [])
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors['missing_columns'] = list(missing_cols)
        
        # Check value ranges
        for column, (min_val, max_val) in AssetDataValidator.VALID_RANGES.items():
            if column in df.columns:
                invalid_mask = (df[column] < min_val) | (df[column] > max_val)
                if invalid_mask.any():
                    errors['invalid_values'].append(
                        f"{column}: {invalid_mask.sum()} values outside range ({min_val}, {max_val})"
                    )
        
        # Check categorical values
        for column, valid_values in AssetDataValidator.VALID_CATEGORIES.items():
            if column in df.columns:
                invalid_mask = ~df[column].isin(valid_values)
                if invalid_mask.any():
                    errors['invalid_values'].append(
                        f"{column}: {invalid_mask.sum()} invalid categories"
                    )
        
        # Check data quality issues
        if 'asset_id' in df.columns:
            duplicates = df['asset_id'].duplicated().sum()
            if duplicates > 0:
                errors['data_quality'].append(f"Duplicate asset_ids: {duplicates}")
        
        return errors
```

## 🧮 Algorithm Implementation

### Priority Scoring Engine
```python
import numpy as np
from typing import Dict, Optional

class PriorityScoreCalculator:
    """Calculates maintenance priority scores for infrastructure assets"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'condition': 0.40,
            'age': 0.20,
            'traffic': 0.30,
            'inspection_urgency': 0.10
        }
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensures weights sum to 1.0"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def calculate_condition_factor(self, condition_rating: float) -> float:
        """Converts condition rating to priority factor (0-1)"""
        # Inverse relationship: lower condition = higher priority
        return max(0, (100 - condition_rating) / 100)
    
    def calculate_age_factor(self, age_years: int, max_age: int = 100) -> float:
        """Converts age to priority factor (0-1)"""
        return min(age_years / max_age, 1.0)
    
    def calculate_traffic_factor(self, traffic_volume: int, max_traffic: int = 200000) -> float:
        """Converts traffic volume to priority factor (0-1)"""
        return min(traffic_volume / max_traffic, 1.0)
    
    def calculate_inspection_factor(self, days_since_inspection: int, max_days: int = 1095) -> float:
        """Converts inspection urgency to priority factor (0-1)"""
        return min(days_since_inspection / max_days, 1.0)
    
    def calculate_priority_score(self, asset_data: Dict) -> float:
        """
        Calculates overall priority score for an asset
        
        Args:
            asset_data: Dictionary containing asset information
                - condition_rating_current: int (0-100)
                - age_years: int
                - traffic_volume_daily: int
                - days_since_last_inspection: int
        
        Returns:
            float: Priority score (0-1, higher = more urgent)
        """
        condition_factor = self.calculate_condition_factor(
            asset_data['condition_rating_current']
        )
        age_factor = self.calculate_age_factor(asset_data['age_years'])
        traffic_factor = self.calculate_traffic_factor(
            asset_data['traffic_volume_daily']
        )
        inspection_factor = self.calculate_inspection_factor(
            asset_data['days_since_last_inspection']
        )
        
        priority_score = (
            condition_factor * self.weights['condition'] +
            age_factor * self.weights['age'] +
            traffic_factor * self.weights['traffic'] +
            inspection_factor * self.weights['inspection_urgency']
        )
        
        return round(priority_score, 4)
    
    def batch_calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate priority scores for entire DataFrame"""
        df = df.copy()
        
        # Vectorized calculations for better performance
        df['condition_factor'] = (100 - df['condition_rating_current']) / 100
        df['age_factor'] = np.minimum(df['age_years'] / 100, 1.0)
        df['traffic_factor'] = np.minimum(df['traffic_volume_daily'] / 200000, 1.0)
        df['inspection_factor'] = np.minimum(df['days_since_last_inspection'] / 1095, 1.0)
        
        df['priority_score'] = (
            df['condition_factor'] * self.weights['condition'] +
            df['age_factor'] * self.weights['age'] +
            df['traffic_factor'] * self.weights['traffic'] +
            df['inspection_factor'] * self.weights['inspection_urgency']
        )
        
        return df.round({'priority_score': 4})
```

### Deterioration Prediction Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class DeteriorationPredictor:
    """Predicts infrastructure asset deterioration over time"""
    
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        self.is_trained = False
        self.feature_names = [
            'age_years', 'traffic_volume_daily', 
            'freeze_thaw_cycles_avg_annual', 'heavy_rain_days_avg_annual'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepares feature matrix for model training/prediction"""
        return df[self.feature_names].values
    
    def train(self, df: pd.DataFrame, target_column: str = 'condition_rating_current'):
        """
        Trains the deterioration prediction model
        
        Args:
            df: Training data containing features and target
            target_column: Name of the target variable column
        """
        X = self.prepare_features(df)
        y = df[target_column].values
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store training metrics
        train_score = self.model.score(X, y)
        print(f"Model R² Score: {train_score:.4f}")
        
        # Feature importance (for linear regression)
        coefficients = self.model.named_steps['regressor'].coef_
        feature_importance = dict(zip(self.feature_names, coefficients))
        print("Feature Coefficients:")
        for feature, coef in feature_importance.items():
            print(f"  {feature}: {coef:.4f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts condition ratings for given assets"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def predict_future_condition(self, df: pd.DataFrame, years_ahead: int) -> np.ndarray:
        """
        Predicts asset condition after specified years
        
        Args:
            df: Current asset data
            years_ahead: Number of years to predict ahead
        
        Returns:
            Array of predicted condition ratings
        """
        df_future = df.copy()
        df_future['age_years'] = df_future['age_years'] + years_ahead
        
        return self.predict(df_future)
    
    def save_model(self, filepath: str):
        """Saves trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Loads trained model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
```

## 📊 Visualization Engine

### Chart Generation System
```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

class InfrastructureVisualizer:
    """Generates various visualizations for infrastructure analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.figure_size = (12, 8)
        self.color_palette = sns.color_palette("husl", 8)
    
    def plot_condition_vs_age(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Creates scatter plot of condition rating vs age, colored by material"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        materials = df['construction_material'].unique()
        
        for i, material in enumerate(materials):
            subset = df[df['construction_material'] == material]
            ax.scatter(
                subset['age_years'], 
                subset['condition_rating_current'],
                label=material,
                alpha=0.7,
                s=60,
                color=self.color_palette[i % len(self.color_palette)]
            )
        
        ax.set_title('Infrastructure Condition vs Age by Material Type', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Age (Years)', fontsize=12)
        ax.set_ylabel('Current Condition Rating (0-100)', fontsize=12)
        ax.legend(title='Construction Material', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_priority_rankings(self, df: pd.DataFrame, top_n: int = 15, 
                              save_path: Optional[str] = None):
        """Creates horizontal bar chart of top priority assets"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        top_assets = df.nlargest(top_n, 'priority_score')
        
        bars = ax.barh(
            range(len(top_assets)), 
            top_assets['priority_score'],
            color=plt.cm.Reds(top_assets['priority_score'])
        )
        
        ax.set_yticks(range(len(top_assets)))
        ax.set_yticklabels(top_assets['asset_id'])
        ax.set_xlabel('Priority Score (Higher = More Urgent)', fontsize=12)
        ax.set_title(f'Top {top_n} Infrastructure Assets by Priority Score', 
                    fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_deterioration_by_material(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Creates bar chart showing average deterioration by material type"""
        df['deterioration'] = df['condition_rating_initial'] - df['condition_rating_current']
        avg_deterioration = df.groupby('construction_material')['deterioration'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        bars = avg_deterioration.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Average Deterioration by Construction Material', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Construction Material', fontsize=12)
        ax.set_ylabel('Average Deterioration (Rating Points)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_condition_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Creates histogram showing distribution of condition ratings"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        n, bins, patches = ax.hist(
            df['condition_rating_current'], 
            bins=20, 
            edgecolor='black', 
            alpha=0.7,
            color='lightcoral'
        )
        
        # Color bars based on condition thresholds
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 30:
                patch.set_facecolor('red')
            elif bin_center < 50:
                patch.set_facecolor('orange')
            elif bin_center < 70:
                patch.set_facecolor('yellow')
            else:
                patch.set_facecolor('green')
        
        ax.set_title('Distribution of Current Condition Ratings', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Condition Rating (0-100)', fontsize=12)
        ax.set_ylabel('Number of Assets', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add condition threshold lines
        for threshold, label, color in [(30, 'Critical', 'red'), 
                                       (50, 'Poor', 'orange'), 
                                       (70, 'Fair', 'yellow')]:
            ax.axvline(threshold, color=color, linestyle='--', alpha=0.7, label=label)
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Creates a comprehensive dashboard with multiple visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        # Subplot 1: Condition vs Age
        ax1 = plt.subplot(2, 2, 1)
        materials = df['construction_material'].unique()
        for i, material in enumerate(materials):
            subset = df[df['construction_material'] == material]
            ax1.scatter(subset['age_years'], subset['condition_rating_current'],
                       label=material, alpha=0.7, s=50)
        ax1.set_title('Condition vs Age by Material')
        ax1.set_xlabel('Age (Years)')
        ax1.set_ylabel('Condition Rating')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Priority Rankings
        ax2 = plt.subplot(2, 2, 2)
        top_assets = df.nlargest(10, 'priority_score')
        ax2.barh(range(len(top_assets)), top_assets['priority_score'])
        ax2.set_yticks(range(len(top_assets)))
        ax2.set_yticklabels(top_assets['asset_id'], fontsize=8)
        ax2.set_title('Top 10 Priority Assets')
        ax2.set_xlabel('Priority Score')
        ax2.invert_yaxis()
        
        # Subplot 3: Material Deterioration
        ax3 = plt.subplot(2, 2, 3)
        df['deterioration'] = df['condition_rating_initial'] - df['condition_rating_current']
        avg_deterioration = df.groupby('construction_material')['deterioration'].mean()
        avg_deterioration.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Average Deterioration by Material')
        ax3.set_ylabel('Deterioration (Points)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Subplot 4: Condition Distribution
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(df['condition_rating_current'], bins=15, alpha=0.7, color='lightgreen')
        ax4.set_title('Condition Rating Distribution')
        ax4.set_xlabel('Condition Rating')
        ax4.set_ylabel('Number of Assets')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

## 🔄 Data Processing Pipeline

### ETL (Extract, Transform, Load) System
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class InfrastructureETL:
    """Handles data extraction, transformation, and loading for infrastructure analysis"""
    
    def __init__(self):
        self.validator = AssetDataValidator()
        self.processed_data = {}
    
    def extract_from_csv(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Extracts data from CSV files
        
        Args:
            file_paths: Dictionary mapping data types to file paths
                {'infrastructure': 'path/to/infrastructure.csv',
                 'traffic': 'path/to/traffic.csv',
                 'weather': 'path/to/weather.csv'}
        
        Returns:
            Dictionary of DataFrames
        """
        dataframes = {}
        
        for data_type, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} records from {file_path}")
                dataframes[data_type] = df
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return dataframes
    
    def transform_infrastructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms and cleans infrastructure data"""
        df = df.copy()
        
        # Convert date columns
        if 'last_inspection_date' in df.columns:
            df['last_inspection_date'] = pd.to_datetime(df['last_inspection_date'])
            df['days_since_last_inspection'] = (
                datetime.now() - df['last_inspection_date']
            ).dt.days
        
        # Handle missing values
        df['condition_rating_current'].fillna(df['condition_rating_current'].median(), inplace=True)
        df['age_years'].fillna(df['age_years'].median(), inplace=True)
        
        # Create derived features
        if 'condition_rating_initial' in df.columns:
            df['deterioration_rate'] = (
                df['condition_rating_initial'] - df['condition_rating_current']
            ) / df['age_years']
            df['deterioration_rate'].fillna(0, inplace=True)
        
        # Standardize categorical values
        df['construction_material'] = df['construction_material'].str.title()
        df['type'] = df['type'].str.title()
        
        return df
    
    def transform_traffic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms and cleans traffic data"""
        df = df.copy()
        
        # Handle outliers in traffic volume
        q1 = df['traffic_volume_daily'].quantile(0.25)
        q3 = df['traffic_volume_daily'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Cap extreme values
        df['traffic_volume_daily'] = np.clip(
            df['traffic_volume_daily'], 
            lower_bound, 
            upper_bound
        )
        
        return df
    
    def transform_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms and cleans weather data"""
        df = df.copy()
        
        # Create weather severity index
        df['weather_severity_index'] = (
            df['freeze_thaw_cycles_avg_annual'] * 0.6 +
            df['heavy_rain_days_avg_annual'] * 0.4
        )
        
        # Normalize weather factors
        df['weather_severity_normalized'] = (
            df['weather_severity_index'] / df['weather_severity_index'].max()
        )
        
        return df
    
    def merge_datasets(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merges all datasets into a master DataFrame"""
        if 'infrastructure' not in dataframes:
            raise ValueError("Infrastructure data is required as the base dataset")
        
        master_df = dataframes['infrastructure'].copy()
        
        # Merge traffic data
        if 'traffic' in dataframes:
            master_df = master_df.merge(
                dataframes['traffic'], 
                on='asset_id', 
                how='left'
            )
        
        # Merge weather data
        if 'weather' in dataframes:
            master_df = master_df.merge(
                dataframes['weather'], 
                on='asset_id', 
                how='left'
            )
        
        return master_df
    
    def validate_merged_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validates the merged dataset for quality issues"""
        issues = {
            'missing_data': [],
            'duplicate_records': [],
            'inconsistent_data': []
        }
        
        # Check for missing critical data
        critical_columns = [
            'asset_id', 'condition_rating_current', 'age_years'
        ]
        
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    issues['missing_data'].append(
                        f"{col}: {missing_count} missing values"
                    )
        
        # Check for duplicates
        duplicate_count = df['asset_id'].duplicated().sum()
        if duplicate_count > 0:
            issues['duplicate_records'].append(
                f"Duplicate asset_ids: {duplicate_count}"
            )
        
        # Check for data inconsistencies
        if 'condition_rating_initial' in df.columns and 'condition_rating_current' in df.columns:
            inconsistent = df[
                df['condition_rating_current'] > df['condition_rating_initial']
            ]
            if len(inconsistent) > 0:
                issues['inconsistent_data'].append(
                    f"Assets with current condition > initial condition: {len(inconsistent)}"
                )
        
        return issues
    
    def process_pipeline(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """Runs the complete ETL pipeline"""
        print("Starting ETL Pipeline...")
        
        # Extract
        print("1. Extracting data from sources...")
        raw_data = self.extract_from_csv(file_paths)
        
        # Transform
        print("2. Transforming data...")
        transformed_data = {}
        
        if 'infrastructure' in raw_data:
            transformed_data['infrastructure'] = self.transform_infrastructure_data(
                raw_data['infrastructure']
            )
        
        if 'traffic' in raw_data:
            transformed_data['traffic'] = self.transform_traffic_data(
                raw_data['traffic']
            )
        
        if 'weather' in raw_data:
            transformed_data['weather'] = self.transform_weather_data(
                raw_data['weather']
            )
        
        # Load (Merge)
        print("3. Merging datasets...")
        master_df = self.merge_datasets(transformed_data)
        
        # Validate
        print("4. Validating data quality...")
        validation_issues = self.validate_merged_data(master_df)
        
        for issue_type, issues in validation_issues.items():
            if issues:
                print(f"  {issue_type.upper()}:")
                for issue in issues:
                    print(f"    - {issue}")
        
        print(f"ETL Pipeline completed. Final dataset: {len(master_df)} records")
        
        self.processed_data = master_df
        return master_df
```

## 🔧 Configuration Management

### Settings and Parameters
```python
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages system configuration and parameters"""
    
    DEFAULT_CONFIG = {
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
            "max_inspection_days": 1095,
            "deterioration_rate_threshold": 2.0
        },
        "visualization_settings": {
            "figure_size": [12, 8],
            "style": "seaborn-v0_8-darkgrid",
            "color_palette": "husl",
            "save_dpi": 300
        },
        "data_validation": {
            "required_confidence": 0.8,
            "outlier_method": "iqr",
            "missing_value_threshold": 0.1
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Loads configuration from file or uses defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Configuration loaded from {self.config_file}")
                return config
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            print("No config file found, using defaults")
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Saves current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Gets configuration value using dot notation
        
        Example:
            config.get("priority_weights.condition") -> 0.40
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Sets configuration value using dot notation
        
        Example:
            config.set("priority_weights.condition", 0.50)
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def update_priority_weights(self, new_weights: Dict[str, float]):
        """Updates priority weights and validates they sum to 1.0"""
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Priority weights must sum to 1.0, got {total}")
        
        self.config["priority_weights"] = new_weights
        print("Priority weights updated successfully")
    
    def get_regional_config(self, region: str) -> Dict[str, Any]:
        """Gets region-specific configuration adjustments"""
        regional_configs = {
            "coastal": {
                "priority_weights": {
                    "condition": 0.50,  # Higher weight due to corrosion
                    "age": 0.15,
                    "traffic": 0.25,
                    "inspection_urgency": 0.10
                },
                "deterioration_multiplier": 1.3
            },
            "mountain": {
                "priority_weights": {
                    "condition": 0.45,
                    "age": 0.25,        # Higher weight due to freeze-thaw
                    "traffic": 0.20,
                    "inspection_urgency": 0.10
                },
                "deterioration_multiplier": 1.2
            },
            "urban": {
                "priority_weights": {
                    "condition": 0.35,
                    "age": 0.15,
                    "traffic": 0.40,    # Higher weight due to congestion impact
                    "inspection_urgency": 0.10
                },
                "deterioration_multiplier": 1.1
            }
        }
        
        return regional_configs.get(region, {})
```

## 🧪 Testing Framework

### Unit Tests
```python
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestInfrastructureAnalysis(unittest.TestCase):
    """Unit tests for infrastructure analysis components"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'asset_id': ['TEST_001', 'TEST_002', 'TEST_003'],
            'type': ['Bridge', 'Road Segment', 'Bridge'],
            'age_years': [10, 25, 45],
            'construction_material': ['Concrete', 'Asphalt', 'Steel'],
            'condition_rating_current': [85, 60, 35],
            'condition_rating_initial': [95, 80, 70],
            'traffic_volume_daily': [15000, 45000, 8000],
            'last_inspection_date': [
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=180),
                datetime.now() - timedelta(days=800)
            ]
        })
        
        self.calculator = PriorityScoreCalculator()
    
    def test_priority_score_calculation(self):
        """Test priority score calculation logic"""
        # Test asset with poor condition and high traffic
        asset_data = {
            'condition_rating_current': 30,
            'age_years': 40,
            'traffic_volume_daily': 80000,
            'days_since_last_inspection': 365
        }
        
        score = self.calculator.calculate_priority_score(asset_data)
        
        # Should have high priority score
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
    
    def test_batch_calculation(self):
        """Test batch priority score calculation"""
        # Add required columns for calculation
        self.sample_data['days_since_last_inspection'] = (
            datetime.now() - self.sample_data['last_inspection_date']
        ).dt.days
        
        result_df = self.calculator.batch_calculate(self.sample_data)
        
        # Check that priority scores are calculated
        self.assertIn('priority_score', result_df.columns)
        self.assertEqual(len(result_df), len(self.sample_data))
        
        # Check score ranges
        scores = result_df['priority_score']
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all(scores <= 1))
    
    def test_data_validation(self):
        """Test data validation functionality"""
        validator = AssetDataValidator()
        
        # Test with valid data
        errors = validator.validate_dataframe(self.sample_data, 'infrastructure')
        self.assertEqual(len(errors['missing_columns']), 0)
        
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'condition_rating_current'] = 150  # Invalid range
        
        errors = validator.validate_dataframe(invalid_data, 'infrastructure')
        self.assertGreater(len(errors['invalid_values']), 0)
    
    def test_deterioration_predictor(self):
        """Test deterioration prediction model"""
        # Add required weather columns
        self.sample_data['freeze_thaw_cycles_avg_annual'] = [5, 15, 10]
        self.sample_data['heavy_rain_days_avg_annual'] = [20, 35, 25]
        
        predictor = DeteriorationPredictor()
        predictor.train(self.sample_data)
        
        # Test prediction
        predictions = predictor.predict(self.sample_data)
        
        self.assertEqual(len(predictions), len(self.sample_data))
        self.assertTrue(all(predictions >= 0))
        self.assertTrue(all(predictions <= 100))
    
    def test_config_manager(self):
        """Test configuration management"""
        config = ConfigManager()
        
        # Test getting configuration values
        condition_weight = config.get('priority_weights.condition')
        self.assertIsNotNone(condition_weight)
        self.assertIsInstance(condition_weight, float)
        
        # Test setting configuration values
        config.set('priority_weights.condition', 0.5)
        updated_weight = config.get('priority_weights.condition')
        self.assertEqual(updated_weight, 0.5)
    
    def test_etl_pipeline(self):
        """Test ETL pipeline functionality"""
        etl = InfrastructureETL()
        
        # Test data transformation
        transformed = etl.transform_infrastructure_data(self.sample_data)
        
        # Check that derived columns are created
        self.assertIn('days_since_last_inspection', transformed.columns)
        
        # Test data validation
        issues = etl.validate_merged_data(transformed)
        self.assertIsInstance(issues, dict)

if __name__ == '__main__':
    unittest.main()
```

## 📈 Performance Optimization

### Large Dataset Handling
```python
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import dask.dataframe as dd

class PerformanceOptimizer:
    """Optimizes system performance for large datasets"""
    
    @staticmethod
    def chunk_process_dataframe(df: pd.DataFrame, 
                               processing_func: callable, 
                               chunk_size: int = 10000) -> pd.DataFrame:
        """Processes large DataFrames in chunks to manage memory"""
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            processed_chunk = processing_func(chunk)
            chunks.append(processed_chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    @staticmethod
    def parallel_priority_calculation(df: pd.DataFrame, 
                                    num_processes: int = None) -> pd.DataFrame:
        """Calculates priority scores using parallel processing"""
        if num_processes is None:
            num_processes = cpu_count() - 1
        
        # Split DataFrame into chunks for parallel processing
        chunk_size = len(df) // num_processes
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        calculator = PriorityScoreCalculator()
        
        with Pool(num_processes) as pool:
            processed_chunks = pool.map(calculator.batch_calculate, chunks)
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    @staticmethod
    def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes DataFrame memory usage by downcasting numeric types"""
        df = df.copy()
        
        # Optimize integer columns
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimize float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category where appropriate
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def create_dask_dataframe(csv_path: str) -> dd.DataFrame:
        """Creates Dask DataFrame for very large datasets"""
        return dd.read_csv(csv_path)
    
    @staticmethod
    def cache_expensive_operations(func):
        """Decorator to cache expensive computation results"""
        cache = {}
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        
        return wrapper
```

## 🔐 Security and Best Practices

### Data Security
```python
import hashlib
import logging
from typing import Any

class SecurityManager:
    """Handles data security and access control"""
    
    @staticmethod
    def anonymize_asset_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Anonymizes asset IDs for data sharing"""
        df = df.copy()
        
        def hash_id(asset_id):
            return hashlib.sha256(asset_id.encode()).hexdigest()[:10]
        
        df['asset_id_anonymous'] = df['asset_id'].apply(hash_id)
        return df
    
    @staticmethod
    def log_data_access(user: str, action: str, data_type: str):
        """Logs data access for audit trails"""
        logging.info(f"User: {user}, Action: {action}, Data: {data_type}, "
                    f"Timestamp: {datetime.now()}")
    
    @staticmethod
    def validate_data_source(data_source: str) -> bool:
        """Validates that data comes from approved sources"""
        approved_sources = [
            'national_highway_authority',
            'state_transport_dept',
            'municipal_corporation',
            'metro_rail_corporation'
        ]
        return data_source.lower() in approved_sources

# Setup logging
logging.basicConfig(
    filename='infrastructure_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

---

This technical documentation provides a comprehensive foundation for understanding, implementing, and extending the Infrastructure Asset Management System. For additional technical support or advanced customization requirements, refer to the system architecture diagrams and API documentation.
