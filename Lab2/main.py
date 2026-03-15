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

def predict_decision_tree(data_point, decision_tree, default_prediction=None):
    node = decision_tree
    
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

def generate_report(data, tree, label):
    predictions = []
    actual = data[label].astype(str)
    default_prediction = data[label].mode()[0]

    for _, row in data.iterrows():
        row = row.drop(label, axis = 0)
        pre = predict_decision_tree(row, tree, default_prediction)
        predictions.append(str(pre))

    report = classification_report(actual, predictions, zero_division=0)
    matrix = confusion_matrix(actual, predictions)

    return report, matrix

def visualize_tree(tree, data, label, graph=None, parent=None, 
                   edge_label='', max_depth=None, depth=0):
    if graph is None:
        graph = graphviz.Digraph(graph_attr={
            'rankdir': 'TB'
        })
    
    class_colors = {
        'unacc':  '#F4A460',
        'acc':    '#87CEEB', 
        'good':   '#90EE90',
        'vgood':  "#24F224"
    }

    def get_node_info(subset, label):
        total = len(subset)
        counts = subset[label].value_counts()
        majority = counts.idxmax()

        # Gini
        gini = 1.0
        for c in counts:
            gini -= (c / total) ** 2

        dist = [int(counts.get(c, 0)) for c in sorted(subset[label].unique())]
        return total, gini, majority, dist

    if max_depth is not None and depth >= max_depth:
        node_id = f'trunc_{parent}_{edge_label}'
        graph.node(node_id, label='...', shape='ellipse',
                   style='filled', fillcolor='lightgray',
                   fontname='Helvetica')
        if parent is not None:
            graph.edge(parent, node_id, label=edge_label, fontsize='10')
        return graph

    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        node_id = str(id(tree))

        total, gini, majority, dist = get_node_info(data, label)
        color = class_colors.get(majority, '#87CEEB')

        node_label = (
            f"{feature}\n"
            f"gini = {gini:.4f}\n"
            f"samples = {total}\n"
            f"value = {dist}\n"
            f"class = {majority}"
        )

        graph.node(node_id, label=node_label,
                   shape='box', style='filled,rounded',
                   fillcolor=color, fontname='Helvetica', fontsize='11')

        if parent is not None:
            graph.edge(parent, node_id, label=edge_label,
                       fontsize='10', fontname='Helvetica')

        for value, subtree in tree[feature].items():
            subset = data[data[feature] == value]
            visualize_tree(subtree, subset, label, graph,
                          parent=node_id, edge_label=str(value),
                          max_depth=max_depth, depth=depth + 1)

    else:
        # Leaf node
        leaf_id = f'leaf_{parent}_{edge_label}'
        total, gini, majority, dist = get_node_info(data, label)
        color = class_colors.get(str(tree), 'white')

        leaf_label = (
            f"gini = {gini:.4f}\n"
            f"samples = {total}\n"
            f"value = {dist}\n"
            f"class = {tree}"
        )

        graph.node(leaf_id, label=leaf_label,
                   shape='box', style='filled,rounded',
                   fillcolor=color, fontname='Helvetica', fontsize='11')

        if parent is not None:
            graph.edge(parent, leaf_id, label=edge_label,
                       fontsize='10', fontname='Helvetica')

    return graph

# -----------------------------------------
#                MAIN PROCESS
# -----------------------------------------

df = pd.read_csv('car.data', header=None, names=[
    'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'
])
print('df.head():\n')
print(df.head())

print('\nClass distribution:\n')
print(df['class'].value_counts())

# Split data into training and testing subsets (80% training, 20% testing)
train_df, test_df = train_test_split(df, test_size=0.2, 
                                      random_state=42, 
                                      stratify=df['class'])

print('\nTraining set class distribution:\n')
print(train_df['class'].value_counts())

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
plt.show()

# Calculate information gain
print("\nInformation Gain (Gini Index):\n")
info_gain = cal_info_gain(df, 'gini', 'class')
info_gain_sorted = sorted(info_gain, key=lambda x: x[1], reverse=True)
for item in info_gain_sorted:
    print(f"{item[0]:30s} {item[1]:.6f}")

# Training

# At depth 5, tree has best accuracy: training 0.96, testing 0.95
# More than 5, tree starts to overfit, at depth 6, training accuracy is 1.0, but testing accuracy drops to 0.90
id3_tree = build_tree_ID3(train_df, 'gini', 'class', max_depth=5)

# Train report
report, matrix = generate_report(train_df, id3_tree, 'class')
print('\nTrain Report\n')
print(report)
plot_matrix(matrix)

# Test report
report, matrix = generate_report(test_df, id3_tree, 'class')
print('\nTest Report\n')
print(report)
plot_matrix(matrix)

# Visualize tree
graph_full = visualize_tree(id3_tree, train_df, 'class')
graph_full.render('car_tree_full', format='png', cleanup=True)
graph_full.view()

# Render readable fragment for report (first 3 levels)
graph_preview = visualize_tree(id3_tree, train_df, 'class', max_depth=3)
graph_preview.render('car_tree_preview', format='png', cleanup=True)
graph_preview.view()