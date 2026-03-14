import pandas as pd
import math
import graphviz
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------
#                 METHODS
# -----------------------------------------

# Bin continuous features into categories
def preprocess_data(df):
    df = df.copy()
    
    df['age'] = pd.cut(df['age'], bins=[0, 16, 18, 21, 100], 
                       labels=['<=16', '17-18', '19-21', '>21'])
    
    df['study_hours_per_week'] = pd.cut(df['study_hours_per_week'], 
                                         bins=[0, 10, 20, 30, 100],
                                         labels=['low', 'medium', 'high', 'very_high'])
    
    df['attendance_percentage'] = pd.cut(df['attendance_percentage'], 
                                          bins=[0, 60, 75, 90, 100],
                                          labels=['poor', 'average', 'good', 'excellent'])
    
    df['assignment_score'] = pd.cut(df['assignment_score'], 
                                     bins=[0, 50, 65, 80, 100],
                                     labels=['low', 'medium', 'high', 'very_high'])
    
    df['exam_score'] = pd.cut(df['exam_score'], 
                               bins=[0, 50, 65, 80, 100],
                               labels=['low', 'medium', 'high', 'very_high'])
    
    df = df.drop('student_id', axis=1)
    
    return df

# Gini Index calculation
def gini_index(data, label):
    counts = data[label].value_counts()
    total = len(data)
    gini = 1.0

    for value in counts:
        prob = value / total
        gini -= prob ** 2

    return gini

# Entropy calculation
def entropy(data, label):
    counts = data[label].value_counts()

    total = len(data)
    entropy = 0

    for value in counts:
        prob = value / total
        entropy = entropy - prob * np.log2(prob)

    return entropy

# information gain calculation
def cal_info_gain(data, method, label):
    if method == 'entropy':
        base_value = entropy(data, label)
    elif method == 'gini':
        base_value = gini_index(data, label)

    total = len(data)
    info_gain = []

    for column in data.columns[:-1]:
        attribute_values = data[column].unique()
        new_value = 0.0

        for value in attribute_values:
            subset = data[data[column] == value]
            prob = len(subset)/ total

            if method == 'entropy':
                new_value += prob * entropy(subset, label)
            elif method == 'gini':
                new_value += prob * gini_index(subset, label)

        info_gain.append([column, base_value - new_value])

    return info_gain

def build_tree_ID3(data, method, label, root=None, max_depth=None, depth=0):
    if max_depth and depth >= max_depth:
        count = data[label].value_counts()
        return count.idxmax()
    
    info_gain = cal_info_gain(data, method, label)
    info_gain = sorted(info_gain, key=lambda x: x[1], reverse=True)
    column_name = info_gain[0][0]

    root = {column_name: {}}

    for attr in data[column_name].unique():
        new_data = data[data[column_name] == attr]
        new_data = new_data.drop(column_name, axis=1)

        if len(new_data.columns) < 2:
            count = new_data[label].value_counts()
            count = count.sort_values(ascending=False)
            root[column_name][attr] = count.index[0]

        elif len(new_data) > 1 and len(new_data[label].unique()) > 1:
            new = build_tree_ID3(new_data, method, label, root, max_depth, depth + 1)
            root[column_name][attr] = new
        
        else:
            output = new_data[label].unique()
            root[column_name][attr] = output[0]
    
    return root

def predict_decision_tree(data_point, decision_tree):
    node = decision_tree
    default_prediction = 'C'
    
    while isinstance(node, dict):
        feature = list(node.keys())[0]
        value = data_point[feature]
        
        if value is None:
            return default_prediction
        
        try:
            node = node[feature][value]
        except KeyError:
            return default_prediction
        
    return node

def plot_matrix(matrix):
    labels = [False, True]
    sns.heatmap(matrix, annot = True, fmt="d", cmap='Greens', xticklabels = labels, yticklabels = labels)
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()

def generate_report(data, tree):
    predictions = []
    actual = data.iloc[:,-1].astype(str)
    for _, row in data.iterrows():
        row = row.drop(data.columns[-1], axis = 0)
        pre = predict_decision_tree(row, tree)
        predictions.append(str(pre))

    report = classification_report(actual, predictions, zero_division=0)
    matrix = confusion_matrix(actual, predictions)

    return report, matrix

# -----------------------------------------
#                MAIN PROCESS
# -----------------------------------------

df = pd.read_csv("student_performance_academic_5000.csv")
df = preprocess_data(df)
print(df.head())

df_raw = pd.read_csv("student_performance_academic_5000.csv")
print(df_raw.groupby('final_grade')[['exam_score', 'assignment_score', 
                                      'attendance_percentage', 
                                      'study_hours_per_week']].mean().round(1))

'''
    Bad data set:
        - A student with ~68% exam score is equally likely to get and A,B,C,D or F. So no algorithm can learn from this data set...

            exam_score  assignment_score  attendance_percentage  study_hours_per_week
final_grade                                                                           
A              68.2           70.8                74.4                  19.5
B              68.4           69.3                74.4                  19.6
C              66.8           70.3                75.1                  20.3
D              67.7           70.6                74.8                  19.8
F              67.2           70.3                75.2                  20.1
'''

# Split data into training and testing subsets (80% training, 20% testing)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

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
# plt.show()

# Calculate information gain
info_gain = cal_info_gain(df, 'gini', 'final_grade')
max_width = max(len(item[0]) for item in info_gain)

for item in info_gain:
    print(f"{item[0].ljust(max_width)}  {float(item[1]):.6f}")

# Training
id3_tree = build_tree_ID3(train_df, 'gini', 'final_grade', max_depth=3)

# Train report
report, matrix = generate_report(train_df, id3_tree)
print('\nTrain Report\n')
print(report)
plot_matrix(matrix)

# Test report
report, matrix = generate_report(test_df, id3_tree)
print('\nTest Report\n')
print(report)
plot_matrix(matrix)