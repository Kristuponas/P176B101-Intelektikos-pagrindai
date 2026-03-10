import pandas as pd
import math
import graphviz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# sudo pacman -S nodejs npm

df = pd.read_csv("student_performance_academic_5000.csv")

print(df.head())

# Split data into training and testing subsets (70% training, 30% testing)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(train_df)}")
print(f"\nTesting set size: {len(test_df)}")

# Calculate missing values
missing_values = df.isnull().sum()
print(f"\n{missing_values}\n")

# One hot encoding
scaler = StandardScaler()
df_hot = pd.get_dummies(df)
df_scaled = pd.DataFrame(scaler.fit_transform(df_hot), columns = df_hot.columns)
df_hot.describe()
print(df_scaled.head())

# Correlation graphs
corr = df_hot.corr()
m = np.triu(corr)
plt.figure(figsize = (12, 12))
sns.heatmap(corr, annot = True, mask = m)

# entropy calculation
def entropy(data, label = 'final_grade'):
    counts = data[label].value_counts()

    total = len(data)
    entropy = 0

    for value in counts:
        prob = value / total
        entropy = entropy - prob * np.log2(prob)

    return entropy

# information gain calculation
def cal_info_gain(data):
    base_entropy = entropy(data)
    total = len(data)

    info_gain = []

    for column in data.columns[:-1]:
        attribute_values = data[column].unique()

        new_entropy = 0.0

        for value in attribute_values:
            subset = data[data[column] == value]
            prob = len(subset)/total
            new_entropy += prob*entropy(subset)

        info_gain.append([column, base_entropy-new_entropy])

    return info_gain

print(cal_info_gain(df))

#def build_tree_ID3(data, root=None):
