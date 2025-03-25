import pandas as pd

def ingest_csv(filepath):
    """Ingests data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

data = ingest_csv("notebook/data/stud.csv")
if data is not None:
    print(data.head(10))