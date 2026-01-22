import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
import os

# Using consistent paths based on your repo structure
REFERENCE_DATA_PATH = "data/processed/train.csv"
LATEST_DATA_PATH = "data/processed/latest_logs.csv" # This is where your API saves new data
REPORT_OUTPUT = "monitoring_report.html"

def generate_drift_report(reference_path=REFERENCE_DATA_PATH, current_path=LATEST_DATA_PATH):
    """Generates an Evidently HTML report comparing training data to production data."""
    
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        return None

    # Load data - Ensure these match the features of your Emotion Classification model
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    # Initialize the report with presets from the DTU guide
    report = Report(metrics=[
        DataDriftPreset(), 
        DataQualityPreset(), 
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(REPORT_OUTPUT)
    
    return REPORT_OUTPUT
