import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("student_performance_academic_5000.csv")

print(df.head())

# Split data into training and testing subsets (70% training, 30% testing)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")
