import pandas as pd
from pathlib import Path

def load_raw_data(base_dir: Path):
    train = pd.read_csv(base_dir / "data" / "drugsComTrain_raw.csv")
    test = pd.read_csv(base_dir / "data" / "drugsComTest_raw.csv")
    return train, test

def clean_data(df: pd.DataFrame):
    df = df[['drugName', 'condition', 'review', 'rating']]
    df.dropna(inplace=True)
    
    def label_rating(rating):
        if rating >= 7:
            return 0   # Safe
        elif rating <= 4:
            return 1   # Adverse
        else:
            return None

    df['label'] = df['rating'].apply(label_rating)
    df.dropna(subset=['label'], inplace=True)
    return df

def save_clean_data(df: pd.DataFrame, base_dir: Path):
    df.to_csv(base_dir / "data" / "clean_drug_data.csv", index=False)
