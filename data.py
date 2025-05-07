from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("dair-ai/emotion")

# Convert to DataFrame
df = pd.DataFrame(dataset["train"])

# Save locally
df.to_csv("emotion_dataset.csv", index=False)
print("Saved emotion_dataset.csv")
