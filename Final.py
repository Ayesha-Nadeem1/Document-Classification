# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:54:10 2024

@author: HP
"""

import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix

# Read data from CSV files
health_df = pd.read_csv("train_health.csv")
disease_df = pd.read_csv("train_disease.csv")
travel_df = pd.read_csv("train_travel.csv")

# Concatenate all dataframes
all_data = pd.concat([health_df, disease_df, travel_df], ignore_index=True)

# Preprocessing function
def preprocess(text):
    # Check if text is not NaN
    if isinstance(text, str):
        # Tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Stop-word removal and stemming can be added here if needed
        return " ".join(tokens)
    else:
        return "doctor"

# Preprocess text data
all_data['text'] = all_data['text'].apply(preprocess)

# Make a Directed Graph according to the paper
def make_graph(string):
    # Split the string into words
    chunks = string.split()
    # Create a directed graph
    G = nx.DiGraph()
    # Add nodes for each unique word
    for chunk in set(chunks):
        G.add_node(chunk)
    # Add edges between adjacent words
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G

# Calculate graph distance
def graph_distance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())

class GraphKNN:
    def __init__(self, k:int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []
    
    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels
    
    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = graph_distance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        return prediction

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Prepare training data
train_texts = all_data['text'].tolist()
train_labels = all_data['label'].tolist()
train_graphs = [make_graph(text) for text in train_texts]

# Train the model
graph_classifier = GraphKNN(k=1)
graph_classifier.fit(train_graphs, train_labels)

# Read data from CSV files
health_df = pd.read_csv("test_health.csv")
disease_df = pd.read_csv("test_disease.csv")
travel_df = pd.read_csv("test_travel.csv")

# Concatenate all dataframes
test_data = pd.concat([health_df, disease_df, travel_df], ignore_index=True)
test_data['text'] = test_data['text'].apply(preprocess)
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

test_graphs = [make_graph(text) for text in test_texts]

# Predict
predictions = [graph_classifier.predict(graph) for graph in test_graphs]
predictions = predictions[:len(test_labels)]
# Evaluate
#test_labels = ["health", "disease", "travel"]
# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
# Calculate accuracy
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate F1 Score for the second class
f1_scores = f1_score(test_labels, predictions, average=None)

f1_score_percentage0 = f1_scores[0] * 100
f1_score_percentage1 = f1_scores[1] * 100
f1_score_percentage2 = f1_scores[2] * 100

print("F1 Score: "
      "health: {:.2f}% "
      "disease: {:.2f}% "
      "travel: {:.2f}%".format(f1_score_percentage0, f1_score_percentage1, f1_score_percentage2))

# Calculate Jaccard similarity for the second class
jaccard = jaccard_score(test_labels, predictions, average=None)

jaccard_percentage0 = jaccard[0] * 100
jaccard_percentage1 = jaccard[1] * 100
jaccard_percentage2 = jaccard[2] * 100

print("Jaccard Similarity: "
      "health: {:.2f}% "
      "disease: {:.2f}% "
      "travel: {:.2f}%".format(jaccard_percentage0, jaccard_percentage1, jaccard_percentage2))

# Plot confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions, labels=list(set(test_labels)))

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(test_labels), yticklabels=set(test_labels))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()