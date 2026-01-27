import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train = pd.read_csv(DATA_DIR / "drugsComTrain_raw.csv")
test = pd.read_csv(DATA_DIR / "drugsComTest_raw.csv")

print(train.head())
print(train.shape)
