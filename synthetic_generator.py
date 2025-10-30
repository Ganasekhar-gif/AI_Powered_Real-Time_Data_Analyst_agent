# synthetic_generator.py
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re

# Configuration
NUM_SAMPLES = 10000  # Target number of samples
OUTPUT_DIR = Path("training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample data patterns
FINANCIAL_METRICS = ["revenue", "profit", "sales", "expenses", "margin", "growth", "ROI", "EBITDA"]
TIME_PERIODS = ["daily", "weekly", "monthly", "quarterly", "yearly", "YTD"]
COMPARISONS = ["vs last period", "vs same period last year", "vs target", "vs forecast", "vs industry average"]
DIMENSIONS = ["region", "product", "category", "channel", "customer_segment", "department"]
CHART_TYPES = ["line", "bar", "pie", "scatter", "area", "histogram", "box", "heatmap"]

class SyntheticDataGenerator:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.datasets = self._load_datasets()
        self.generated_samples = 0
        
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the data directory"""
        datasets = {}
        for file in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                # Basic cleaning
                df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
                datasets[file.stem] = df
                print(f"Loaded {file.name} with shape {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return datasets
    
    def generate_sql_agent_output(self) -> Dict[str, Any]:
        """Generate synthetic SQL agent output"""
        metric = random.choice(FINANCIAL_METRICS)
        dimension = random.choice(DIMENSIONS)
        time_period = random.choice(TIME_PERIODS)
        
        # Generate a realistic SQL query based on available columns
        tables = list(self.datasets.keys())
        if not tables:
            tables = ["financial_data"]
            
        table = random.choice(tables)
        columns = self.datasets.get(table, pd.DataFrame()).columns.tolist()[:5] or ["*"]
        
        # Simple WHERE clause
        where_clause = ""
        if "date" in columns:
            where_clause = "WHERE date >= CURRENT_DATE - INTERVAL '1 year'"
        
        sql = f"SELECT {', '.join(columns)} FROM {table} {where_clause} LIMIT 100;"
        
        # Generate a sample result
        preview = self._generate_sample_data(columns)
        
        return {
            "type": "sql",
            "question": f"Show me {metric} by {dimension} {random.choice(COMPARISONS)}",
            "sql": sql,
            "rows": random.randint(50, 1000),
            "preview": preview.to_dict(orient="records")[:5]  # First 5 rows as dict
        }
    
    def generate_python_agent_output(self) -> Dict[str, Any]:
        """Generate synthetic Python agent output"""
        operation = random.choice(["clean", "transform", "analyze", "calculate"])
        metric = random.choice(FINANCIAL_METRICS)
        dimension = random.choice(DIMENSIONS)
        
        # Generate sample code
        code = f"""# {operation.capitalize()} {metric} data
import pandas as pd
import numpy as np

# Assuming df is your DataFrame
df = df.copy()

# {operation} operation
"""
        
        if operation == "clean":
            code += f"""# Clean {metric} data
df['{metric}'] = pd.to_numeric(df['{metric}'].astype(str).str.replace('[^0-9.-]', ''), errors='coerce')
df = df.dropna(subset=['{metric}'])"""
        
        elif operation == "transform":
            code += f"""# Transform {metric} data
df['{metric}_log'] = np.log1p(df['{metric}'])
df['{metric}_pct_change'] = df.groupby('{dimension}')['{metric}'].pct_change()"""
        
        elif operation == "analyze":
            code += f"""# Analyze {metric} by {dimension}
result = df.groupby('{dimension}')['{metric}'].agg(['mean', 'sum', 'count', 'std']).reset_index()
result = result.rename(columns={{
    'mean': 'avg_{metric}',
    'sum': 'total_{metric}',
    'count': 'record_count',
    'std': 'std_dev_{metric}'
}})
result = result.sort_values('total_{metric}', ascending=False)"""
        
        else:  # calculate
            code += f"""# Calculate {metric} metrics
result = pd.DataFrame({{
    'total_{metric}': [df['{metric}'].sum()],
    'avg_{metric}': [df['{metric}'].mean()],
    'min_{metric}': [df['{metric}'].min()],
    'max_{metric}': [df['{metric}'].max()],
    'std_dev_{metric}': [df['{metric}'].std()]
}})"""
        
        return {
            "type": "python",
            "question": f"{operation.capitalize()} {metric} data{f' by {dimension}' if random.random() > 0.5 else ''}",
            "code": code,
            "operation": operation
        }
    
    def generate_viz_agent_output(self) -> Dict[str, Any]:
        """Generate synthetic visualization agent output"""
        chart_type = random.choice(CHART_TYPES)
        metric = random.choice(FINANCIAL_METRICS)
        dimension = random.choice(DIMENSIONS)
        
        # Generate sample visualization code
        if chart_type in ["line", "bar", "area"]:
            code = f"""import plotly.express as px

# Create {chart_type} chart of {metric} by {dimension}
fig = px.{chart_type}(
    df,
    x='{dimension}',
    y='{metric}',
    title='{metric.capitalize()} by {dimension}'
)
fig.show()"""
        
        elif chart_type == "pie":
            code = f"""import plotly.express as px

# Create pie chart of {metric} distribution
fig = px.pie(
    df,
    names='{dimension}',
    values='{metric}',
    title='{metric.capitalize()} Distribution by {dimension}'
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()"""
        
        else:  # scatter, histogram, box, heatmap
            code = f"""import plotly.express as px

# Create {chart_type} plot
fig = px.{chart_type}(
    df,
    x='{dimension}',
    y='{metric}',
    title='{metric.capitalize()} {chart_type.capitalize()} Plot'
)
fig.show()"""
        
        return {
            "type": "visualization",
            "question": f"Create a {chart_type} chart showing {metric} by {dimension}",
            "chart_type": chart_type,
            "code": code
        }
    
    def generate_data_examiner_output(self) -> Dict[str, Any]:
        """Generate synthetic data examiner output"""
        issues = random.sample([
            "missing_values", "outliers", "data_type_mismatch",
            "duplicates", "inconsistent_formatting", "anomalies"
        ], k=random.randint(1, 3))
        
        recommendations = random.sample([
            "impute missing values", "remove outliers", "convert data types",
            "standardize formats", "normalize values", "create derived features"
        ], k=random.randint(2, 4))
        
        return {
            "type": "examination",
            "issues_found": issues,
            "recommendations": recommendations,
            "dataset_type": random.choice(["transaction", "customer", "sales", "inventory"])
        }
    
    def _generate_sample_data(self, columns: List[str], n: int = 5) -> pd.DataFrame:
        """Generate sample data with realistic values based on column names"""
        data = {}
        for col in columns:
            col = str(col).lower()
            if any(x in col for x in ["date", "time", "day", "month", "year"]):
                dates = pd.date_range(end=datetime.now(), periods=n).strftime('%Y-%m-%d')
                data[col] = dates
            elif any(x in col for x in ["id", "num", "code", "no", "#"]):
                data[col] = [f"{random.randint(1000, 9999)}" for _ in range(n)]
            elif any(x in col for x in ["amount", "price", "revenue", "sales", "profit", "cost"]):
                data[col] = [round(random.uniform(100, 10000), 2) for _ in range(n)]
            elif any(x in col for x in ["qty", "quantity", "count", "units"]):
                data[col] = [random.randint(1, 100) for _ in range(n)]
            elif any(x in col for x in ["name", "product", "category", "type", "status"]):
                samples = ["Premium", "Standard", "Basic", "Economy", "Deluxe", "Limited"]
                data[col] = random.choices(samples, k=n)
            elif any(x in col for x in ["region", "country", "city", "location"]):
                samples = ["North", "South", "East", "West", "Central"]
                data[col] = random.choices(samples, k=n)
            else:
                # Default to string if we don't recognize the column
                data[col] = [f"Sample {i+1}" for i in range(n)]
        
        return pd.DataFrame(data)
    
    def generate_explanation(self, agent_output: Dict[str, Any]) -> Dict[str, str]:
        """Generate a synthetic explanation for the agent's output"""
        if agent_output["type"] == "sql":
            return {
                "summary": f"Executed SQL query to retrieve {agent_output.get('rows', 0)} rows of {agent_output['question'].split()[-1]} data.",
                "next_steps": [
                    "Consider filtering the results further to focus on specific segments",
                    "Create visualizations to better understand the data distribution",
                    "Calculate additional metrics or KPIs from this dataset"
                ]
            }
        elif agent_output["type"] == "python":
            return {
                "summary": f"Performed {agent_output.get('operation', 'data processing')} operation on the dataset.",
                "next_steps": [
                    "Review the transformed data for any anomalies",
                    "Save the processed data for future analysis",
                    "Create visualizations to explore the results"
                ]
            }
        elif agent_output["type"] == "visualization":
            return {
                "summary": f"Generated a {agent_output.get('chart_type', 'chart')} showing {agent_output['question'].split('showing')[-1].strip()}.",
                "next_steps": [
                    "Adjust the chart parameters to highlight different aspects of the data",
                    "Save the visualization for your report or presentation",
                    "Consider creating additional visualizations with different dimensions"
                ]
            }
        else:  # examination
            return {
                "summary": f"Examined the dataset and identified {len(agent_output.get('issues_found', []))} potential issues.",
                "next_steps": agent_output.get("recommendations", []) + [
                    "Review the data quality issues in more detail",
                    "Document any data cleaning steps taken",
                    "Update data validation rules if needed"
                ]
            }
    
    def generate_training_example(self) -> Dict[str, Any]:
        """Generate a complete training example with agent output and explanation"""
        # Randomly select an agent type
        agent_type = random.choices(
            ["sql", "python", "visualization", "examination"],
            weights=[0.4, 0.3, 0.2, 0.1]  # Weighted distribution
        )[0]
        
        # Generate agent output
        if agent_type == "sql":
            agent_output = self.generate_sql_agent_output()
        elif agent_type == "python":
            agent_output = self.generate_python_agent_output()
        elif agent_type == "visualization":
            agent_output = self.generate_viz_agent_output()
        else:  # examination
            agent_output = self.generate_data_examiner_output()
        
        # Generate explanation
        explanation = self.generate_explanation(agent_output)
        
        return {
            "agent_output": agent_output,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_dataset(self, num_samples: int) -> None:
        """Generate a dataset of training examples"""
        output_file = OUTPUT_DIR / "explainer_training_data.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(num_samples):
                try:
                    example = self.generate_training_example()
                    f.write(json.dumps(example) + "\n")
                    
                    # Print progress
                    if (i + 1) % 100 == 0:
                        print(f"Generated {i + 1}/{num_samples} examples...")
                
                except Exception as e:
                    print(f"Error generating example {i + 1}: {e}")
        
        print(f"\nSuccessfully generated {num_samples} training examples in {output_file}")

if __name__ == "__main__":
    # Initialize the generator with your data directory
    data_dir = Path("data")  # Update this to your data directory
    generator = SyntheticDataGenerator(data_dir)
    
    # Generate the dataset
    generator.generate_dataset(NUM_SAMPLES)