import pandas as pd
from jinja2 import Template
from scipy import stats
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

# Template for the contextual background
template_str = """
## Contextual Background of the Dataset

### Overview
- **Dataset Name**: {{ dataset_name }}
- **Description**: {{ description }}

### Data Structure
- **Total Rows**: {{ total_rows }}
- **Total Columns**: {{ total_columns }}

### Data Collection
- **Method**: {{ method }}
- **Time Period**: {{ time_period }}
- **Geographical Scope**: {{ geographical_scope }}
- **Data Format**: {{ data_format }}

### Data Quality
- **Missing Values**: {{ missing_values }}
- **Outliers**: {{ outliers }}
- **Data Cleaning**: {{ data_cleaning }}

### Key Insights
- **Trends**: {{ trends }}
- **Patterns**: {{ patterns }}
- **Anomalies**: {{ anomalies }}

### Applications
- **Primary Use Case**: {{ primary_use_case }}
- **Other Potential Uses**: {{ other_potential_uses }}
- **Limitations**: {{ limitations }}


### Practical Recommendations
- **Further Exploration**: {{ further_exploration }}
"""

def read_chart_info(file_path):
    chart_title = None
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("Chart Title:"):
                chart_title = line.split(":", 1)[1].strip()

    if not chart_title:
        chart_title="Cannot be determined"

    return {
        "chart_title": chart_title
    }


def generate_trends_and_patterns(df):
    trends = []
    patterns = []
    # Converting column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Trends
    if 'date' in df.columns or 'time' in df.columns or 'year' in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        elif 'year' in df.columns:
            df['year'] = pd.to_datetime(df['year'], format='%Y')

        trends.append("Time-based trends are observable in the dataset.")

        # Checking for increasing or decreasing trends over time
        for col in df.select_dtypes(include=[np.number]).columns:
            time_col = 'date' if 'date' in df.columns else 'time' if 'time' in df.columns else 'year'
            df_sorted = df.sort_values(by=time_col)
            correlation = df_sorted[time_col].map(pd.Timestamp.timestamp).corr(df_sorted[col])
            if correlation > 0.5:
                trends.append(f"The column '{col}' shows an increasing trend over time.")
            elif correlation < -0.5:
                trends.append(f"The column '{col}' shows a decreasing trend over time.")

    # Patterns
    if 'height' in df.columns or 'heights' in df.columns:
        height_stats = df['height'].describe()
        patterns.append(f"Height analysis shows: Min={height_stats['min']:.2f} cm, Max={height_stats['max']:.2f} cm, Mean={height_stats['mean']:.2f} cm, Std Dev={height_stats['std']:.2f} cm.")
        patterns.append("Height data may show patterns related to age or gender, if available.")

    if 'weight' in df.columns or 'weights' in df.columns:
        weight_stats = df['weight'].describe()
        patterns.append(f"Weight analysis shows: Min={weight_stats['min']:.2f} kg, Max={weight_stats['max']:.2f} kg, Mean={weight_stats['mean']:.2f} kg, Std Dev={weight_stats['std']:.2f} kg.")
        patterns.append("Weight data may show correlations with height or age, if available.")

    if 'score' in df.columns or 'marks' in df.columns or 'mark' in df.columns:
        score_stats = df['score'].describe()
        patterns.append(f"Score analysis shows: Min={score_stats['min']:.2f}, Max={score_stats['max']:.2f}, Mean={score_stats['mean']:.2f}, Std Dev={score_stats['std']:.2f}.")
        patterns.append("Scores might show trends related to changes in teaching methods or student performance.")
    
    if 'age' in df.columns:
        age_distribution = df['age'].describe()
        patterns.append(f"Age distribution analysis shows: Min={age_distribution['min']}, Max={age_distribution['max']}, Mean={age_distribution['mean']:.2f}, Std Dev={age_distribution['std']:.2f}.")

    if 'category' in df.columns or 'type' in df.columns:
        category_column = 'category' if 'category' in df.columns else 'type'
        category_counts = df[category_column].value_counts().to_dict()
        patterns.append(f"Category/Type distribution: {', '.join([f'{k}: {v}' for k, v in category_counts.items()])}.")
        if 'date' in df.columns:
            monthly_category_counts = df.groupby(df['date'].dt.to_period("M"))[category_column].value_counts()
            patterns.append(f"Monthly category distribution: {', '.join([f'{period}: {count}' for period, count in monthly_category_counts.items()])}.")

    if 'price' in df.columns:
        price_trends = df['price'].describe()
        patterns.append(f"Price analysis shows: Min={price_trends['min']}, Max={price_trends['max']}, Mean={price_trends['mean']:.2f}, Std Dev={price_trends['std']:.2f}.")
        # Check for seasonal pricing patterns if date-related column is present
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            monthly_avg_price = df.groupby('month')['price'].mean()
            patterns.append(f"Monthly average prices: {', '.join([f'Month {i}: {v:.2f}' for i, v in monthly_avg_price.items()])}.")

    if 'revenue' in df.columns:
        revenue_trends = df['revenue'].describe()
        patterns.append(f"Revenue analysis shows: Min={revenue_trends['min']}, Max={revenue_trends['max']}, Mean={revenue_trends['mean']:.2f}, Std Dev={revenue_trends['std']:.2f}.")

    if 'humidity' in df.columns:
        humidity_trends = df['humidity'].describe()
        patterns.append(f"Humidity analysis shows: Min={humidity_trends['min']}, Max={humidity_trends['max']}, Mean={humidity_trends['mean']:.2f}, Std Dev={humidity_trends['std']:.2f}.")
        # Checking for seasonal humidity patterns if date-related column is present
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            monthly_avg_humidity = df.groupby('month')['humidity'].mean()
            patterns.append(f"Monthly average humidity: {', '.join([f'Month {i}: {v:.2f}' for i, v in monthly_avg_humidity.items()])}.")

    if 'quantity' in df.columns:
        quantity_trends = df['quantity'].describe()
        patterns.append(f"Quantity analysis shows: Min={quantity_trends['min']}, Max={quantity_trends['max']}, Mean={quantity_trends['mean']:.2f}, Std Dev={quantity_trends['std']:.2f}.")
        # Checking for trends in inventory over time if date-related column is present
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            monthly_avg_quantity = df.groupby('month')['quantity'].mean()
            patterns.append(f"Monthly average quantity: {', '.join([f'Month {i}: {v:.2f}' for i, v in monthly_avg_quantity.items()])}.")

    if not trends:
        trends.append("No trends found in the data.")
    if not patterns:
        patterns.append("No patterns found in the data.")

    return '\n'.join(trends), '\n'.join(patterns)



# Defining primary use cases and other potential uses based on data type and context
def determine_use_cases(df):
    primary_use_case = ""
    other_potential_uses = []
    # Converting column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Sales and Revenue Analysis
    if any(col in df.columns for col in ['sales', 'revenue', 'profit']):
        primary_use_case = "Sales Analysis"
        other_potential_uses = ["Market Trend Analysis", "Revenue Forecasting", "Financial Performance Review"]

    # Weather and Climate Data
    elif any(col in df.columns for col in ['temperature', 'humidity', 'precipitation', 'wind_speed']):
        primary_use_case = "Weather Analysis"
        other_potential_uses = ["Climate Research", "Agricultural Planning", "Disaster Management"]

    # Demographic Data
    elif any(col in df.columns for col in ['age', 'gender', 'income', 'education']):
        primary_use_case = "Demographic Analysis"
        other_potential_uses = ["Market Segmentation", "Healthcare Research", "Socio-economic Studies"]

    # Healthcare Data
    elif any(col in df.columns for col in ['patient_id', 'diagnosis', 'treatment', 'medication']):
        primary_use_case = "Healthcare Analysis"
        other_potential_uses = ["Medical Research", "Patient Care Improvement", "Public Health Policy Making"]

    # Financial Data
    elif any(col in df.columns for col in ['stock_price', 'market_cap', 'dividend', 'earnings']):
        primary_use_case = "Financial Market Analysis"
        other_potential_uses = ["Investment Research", "Economic Forecasting", "Portfolio Management"]

    # E-commerce Data
    elif any(col in df.columns for col in ['product_id', 'customer_id', 'purchase_date', 'quantity']):
        primary_use_case = "E-commerce Analysis"
        other_potential_uses = ["Customer Behavior Analysis", "Inventory Management", "Sales Performance Review"]

    # Social Media Data
    elif any(col in df.columns for col in ['post_id', 'user_id', 'likes', 'comments']):
        primary_use_case = "Social Media Analysis"
        other_potential_uses = ["Sentiment Analysis", "User Engagement Analysis", "Content Performance Review"]

    elif any(col in df.columns for col in ['timestamp', 'event', 'location']):
        primary_use_case = "Event and Log Analysis"
        other_potential_uses = ["Security Monitoring", "Usage Pattern Analysis", "Operational Efficiency Studies", "System Performance Monitoring"]
    
    elif any(col in df.columns for col in ['transaction_id', 'amount', 'merchant', 'card_type']):
        primary_use_case = "Transaction and Financial Analysis"
        other_potential_uses = ["Fraud Detection", "Spending Behavior Analysis", "Risk Assessment", "Financial Auditing"]
    
    elif any(col in df.columns for col in ['test_score', 'grade', 'attendance', 'subject']):
        primary_use_case = "Educational Analysis"
        other_potential_uses = ["Student Performance Tracking", "Curriculum Development", "Educational Research", "Policy Making"]
    
    elif any(col in df.columns for col in ['species', 'habitat', 'population', 'conservation_status']):
        primary_use_case = "Environmental and Ecological Analysis"
        other_potential_uses = ["Biodiversity Studies", "Conservation Planning", "Wildlife Monitoring", "Environmental Impact Assessments"]
    # General Data Analysis
    else:
        primary_use_case = "General Data Analysis"
        other_potential_uses = ["Exploratory Data Analysis", "Machine Learning Model Training", "Statistical Analysis"]

    return primary_use_case, other_potential_uses


def generate_caption(csv_path, txt_path):
    """Generate a caption based on the CSV data and append it to the chart info text file."""
    chart_info = read_chart_info(txt_path)
    df = pd.read_csv(csv_path)

    total_rows = len(df)
    total_columns = len(df.columns)
    first_column_name = df.columns[0]
    
    # Extracting key statistics
    max_value = df[first_column_name].max() if df[first_column_name].dtype in ['int64', 'float64'] else 'N/A'
    min_value = df[first_column_name].min() if df[first_column_name].dtype in ['int64', 'float64'] else 'N/A'
    
    # Generate the caption
    caption = (
        f"Here's a chart titled '{chart_info['chart_title']}. "
        f"The data in the first column '{first_column_name}'. "
        f"This chart effectively visualizes trends in the dataset, offering insights into the patterns."
    )
    
    # Append the caption to the chart info text file
    with open(txt_path, 'a') as file:
        file.write(f"\n\nGenerated Caption:\n{caption}")
    
    return caption



def display_patterns_as_text(patterns):
    summary = []
    # Identifying key correlations
    for row_name, correlations in patterns.items():
        significant_correlations = {col_name: corr for col_name, corr in correlations.items() if row_name != col_name and corr > 0.5}
        
        if significant_correlations:
            summary.append(f"Looking at the {row_name} data, we notice some important patterns:")
            for col_name, correlation in significant_correlations.items():
                summary.append(f"  - There is a strong correlation of {correlation:.2f} with {col_name}.")
            summary.append("")

    return "\n".join(summary)


def detect_anomalies(df):
    anomalies = {
        'Categorical Data': {},
        'Time Series Data': {}
    }

    # Analyzing categorical columns for anomalies
    for col in df.select_dtypes(include=['object', 'category']).columns:
        col_data = df[col].dropna()
        
        # Label Encoding for categorical data
        if col_data.size > 0:
            le = LabelEncoder()
            encoded_values = le.fit_transform(col_data)
            iso_forest = IsolationForest(contamination=0.01)
            predictions = iso_forest.fit_predict(encoded_values.reshape(-1, 1))
            anomaly_indices = predictions == -1
            if anomaly_indices.any():
                anomalies['Categorical Data'][f'{col} (Categorical)'] = col_data[anomaly_indices].tolist()
    
    # Analyzing time series data for anomalies (if applicable)
    for col in df.select_dtypes(include=['datetime64']).columns:
        col_data = df[col].dropna()
        
        if col_data.size > 1:  # Need more than 1 data point to analyze
            df_sorted = df.sort_values(by=col).reset_index(drop=True)
            values = df_sorted[col].astype(np.int64)  # Convert datetime to numerical
            rolling_mean = values.rolling(window=10).mean()
            rolling_std = values.rolling(window=10).std()
            anomaly_indices = abs(values - rolling_mean) > 3 * rolling_std
            if anomaly_indices.any():
                anomalies['Time Series Data'][f'{col} (Time Series)'] = df_sorted[col][anomaly_indices].tolist()
    
    if not any(anomalies.values()):
        return "No anomalies were detected in the dataset."

    # Format the results for user-friendly output
    anomaly_report = "Anomaly Detection Report:\n"
    
    for category, data in anomalies.items():
        anomaly_report += f"\n{category}:\n"
        if not data:
            anomaly_report += "  No anomalies found.\n"
        for feature, values in data.items():
            if values:
                anomaly_report += f"  - {feature}: Detected anomalies: {values}\n"
            else:
                anomaly_report += f"  - {feature}: No anomalies detected.\n"
    
    return anomaly_report


def infer_method(df):
    # infer the method using the data columns
    if 'method' in df.columns:
        return df['method'].dropna().unique().tolist()
    return "Not specified"


def infer_time_period(df):
        time_period = {}
        
        # Check for columns related to "year"
        year_columns = df.filter(like="Year"or"year", axis=1).columns
        if not year_columns.empty:
            year_col = year_columns[0] 
            if df[year_col].dtype in ['int64', 'float64']:
                start_year = df[year_col].min()
                end_year = df[year_col].max()
                time_period = {
                    "start_date": pd.Timestamp(year=int(start_year), month=1, day=1),
                    "end_date": pd.Timestamp(year=int(end_year), month=12, day=31)
                }
            else:
                print("Year column exists but is not numeric.")
        
        # Check for date columns
        date_columns = df.select_dtypes(include=['datetime64', 'object']).columns

        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                start_date = df[col].min()
                end_date = df[col].max()
                time_period = {
                    "start_date": start_date,
                    "end_date": end_date
                }
                break  # Assuming we only need the time period from one date column
            except Exception as e:
                print(f"Could not parse column {col} as datetime: {e}")
        
        if time_period:
            time_period_str = f"From {time_period['start_date'].strftime('%Y-%m-%d')} to {time_period['end_date'].strftime('%Y-%m-%d')}"
        else:
            time_period_str = "No date or time columns found."
        
        return time_period_str

def infer_geographical_scope(df):
    # Attempting to infer geographical scope based on location columns if available
    location_columns = ['country', 'region', 'city']  # Common geographical columns
    geo_cols = [col for col in location_columns if col in df.columns]
    if geo_cols:
        return df[geo_cols].dropna().drop_duplicates().to_dict(orient='records')
    return "Not specified"

def identify_limitations(df):
    limitations = []

    # Check for missing values
    missing_values = df.isnull().mean().mean()
    if missing_values > 0.1:
        limitations.append(f"High proportion of missing values ({missing_values:.2%})")

    # Check for potential biases
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts(normalize=True)
        if (gender_counts < 0.4).any():
            limitations.append("Gender distribution is skewed")

    if 'age' in df.columns:
        age_counts = df['age'].value_counts(bins=10, normalize=True)
        if (age_counts < 0.1).any():
            limitations.append("Age distribution is skewed")

    # Check for temporal inconsistencies (if date columns are present)
    date_cols = df.select_dtypes(include=['datetime']).columns
    for col in date_cols:
        if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
            limitations.append(f"Column '{col}' has consistent time series")
        else:
            limitations.append(f"Column '{col}' has inconsistent time series")

    # Check for columns with potential duplicates
    duplicate_cols = [col for col in df.columns if df[col].duplicated().mean() > 0.1]
    if duplicate_cols:
        limitations.append(f"Columns with potential duplicates: {', '.join(duplicate_cols)}")

    # Check for unstructured or poorly structured text data
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        avg_word_count = df[col].apply(lambda x: len(str(x).split())).mean()
        if avg_word_count > 20:
            limitations.append(f"Column '{col}' contains long text entries")

    # Check for class imbalance in target column (if applicable)
    if 'target' in df.columns:
        class_counts = df['target'].value_counts(normalize=True)
        if (class_counts < 0.1).any():
            limitations.append("Some categories in the data might be too common or too rare, which can affect the accuracy of the results.")

    # Check for high cardinality in categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() > 100:
            limitations.append(f"Column '{col}' has high cardinality")

    # Check for multicollinearity
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
        if highly_correlated:
            limitations.append(f"Some columns in the data might be too similar to each other, which can confuse the analysis. Multicollinearity detected in columns: {', '.join(highly_correlated)}")

    # Check for irrelevant features
    irrelevant_features = [col for col in df.columns if df[col].nunique() == 1]
    if irrelevant_features:
        limitations.append(f"Irrelevant features with only one unique value: {', '.join(irrelevant_features)}")

    # Check for feature scaling issues
    if len(numerical_cols) > 0:
        feature_ranges = df[numerical_cols].max() - df[numerical_cols].min()
        if (feature_ranges < 1).any():
            limitations.append("Adjusting the range of numbers in your data so that all features have an equal impact on the analysis")

    # General limitation if no specific issues found
    if not limitations:
        limitations.append("No significant limitations identified")

    return limitations


def generate_further_exploration(df, primary_use_case):
    further_exploration = []

    if primary_use_case == "Sales Analysis":
        further_exploration.append("Consider using time series forecasting models like ARIMA, Prophet, or LSTM to predict future sales trends.")
        further_exploration.append("Investigate seasonality and trend components in sales data to improve forecast accuracy.")
        further_exploration.append("Analyze the impact of marketing campaigns, holidays, and other events on sales.")

    elif primary_use_case == "Weather Analysis":
        further_exploration.append("Apply time series models to forecast future weather conditions.")
        further_exploration.append("Use regression models to predict temperature and humidity levels based on historical data.")
        further_exploration.append("Investigate the effects of climate change on local weather patterns.")

    elif primary_use_case == "Demographic Analysis":
        further_exploration.append("Use demographic data to predict population growth and demographic shifts.")
        further_exploration.append("Analyze trends in age, gender, and income distribution to inform social policies and business strategies.")
        further_exploration.append("Investigate the correlation between demographic factors and economic indicators.")

    elif primary_use_case == "General Data Analysis":
        further_exploration.append("Consider using machine learning models to identify patterns and make predictions.")
        further_exploration.append("Explore clustering techniques to segment the data into meaningful groups.")
        further_exploration.append("Apply dimensionality reduction techniques like PCA to simplify complex datasets.")

    else:
        further_exploration.append("No specific further exploration recommendations for this dataset.")

    return '\n'.join(further_exploration)


@router.post("/generate_caption")
async def detect_chart():
    csv_file_path = "extracted_table.csv"
    df = pd.read_csv(csv_file_path)

    file_path = "chart_info.txt"
    chart_info = read_chart_info(file_path)

    dataset_name = chart_info['chart_title']
    description = generate_caption(csv_file_path, file_path)
    total_rows, total_columns = df.shape
    data_format = "CSV"
    method = infer_method(df)
    time_period = infer_time_period(df)
    geographical_scope = infer_geographical_scope(df)
    primary_use_case, other_potential_uses = determine_use_cases(df)
    data_format = {col: str(df[col].dtype) for col in df.columns}
    missing_values = df.isnull().sum().sum() / (total_rows * total_columns) * 100

    outliers = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        z_scores = stats.zscore(df[col].dropna())
        outlier_count = (abs(z_scores) > 3).sum()
        if outlier_count > 0:
            outliers[col] = outlier_count

    data_cleaning = "Removed rows with missing values" if df.isnull().any().any() else "No missing values detected"

    trends, patterns = generate_trends_and_patterns(df)
    anomalies = detect_anomalies(df)
    limitations = identify_limitations(df)
    further_exploration = generate_further_exploration(df, primary_use_case)

    context = {
        "dataset_name": dataset_name,
        "description": description,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "method": method,
        "time_period": time_period,
        "geographical_scope": geographical_scope,
        "data_format": data_format,
        "missing_values": f"{missing_values:.2f}%",
        "outliers": outliers,
        "data_cleaning": data_cleaning,
        "trends": trends,
        "patterns": patterns,
        "anomalies": anomalies,
        "primary_use_case": primary_use_case,
        "other_potential_uses": other_potential_uses,
        "limitations": limitations,
        "further_exploration": further_exploration
    }

    # Rendering the template
    template = Template(template_str)
    report = template.render(context)
    
    # Saving the report to a markdown file
    with open("contextual_background_report.md", "w") as file:
        file.write(report)

    return {"message": "Contextual background report generated successfully."}

