# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:34:28 2024

@author: Ayesha Nadeem
"""

import requests
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pandas as pd
import os
import csv
import networkx as nx
import random
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt

# Make a Directed Graph according to the paper.
def make_graph(string):
    if string is None or not isinstance(string, str):
        pass
    
    # Split the string into words
    chunks = re.findall(r'\b\w+\b', str(string))  # Ensure string type
    # Create a directed graph
    G = nx.DiGraph()
    # Add nodes for each unique word
    for chunk in set(chunks):
        G.add_node(chunk)
    # Add edges between adjacent words
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G

def extract_common_subgraphs(g1, g2):
    # subgraphs = []
    # for graph in graphs:
    #     if graph is not None:  # Check if graph is not None
    #         # Extract all subgraphs
    #         undirected_graph = graph.to_undirected()
    #         for subgraph in nx.connected_components(undirected_graph):
    #             subgraphs.append(tuple(subgraph))  # Convert set to tuple
    # # Count occurrences of each subgraph
    # subgraph_counts = Counter(subgraphs)
    # # Filter common subgraphs
    # common_subgraphs = [set(subgraph) for subgraph, count in subgraph_counts.items() if count > 1]  # Convert back to set
    # return common_subgraphs
    print(g1)
    print(g2)
    edge_set1 = set(g1.edges())
    edge_set2 = set(g2.edges())
    common = edge_set1.intersection(edge_set2)
    max_graph = nx.Graph(list(common))
    return -len(max_graph.edges())

class GraphKNN:
  def __init__(self, k: int):
    self.k = k
    self.train_graphs = []
    self.train_labels = []

  def fit(self, train_graphs, train_labels):
    self.train_graphs = train_graphs
    self.train_labels = train_labels

  def predict(self, graph):
    distances = []
    for train_graph in self.train_graphs:
      distance = extract_common_subgraphs(graph, train_graph)
      distances.append(distance)

    nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
    # Get labels of nearest neighbors
    nearest_labels = [self.train_labels[i] for i in nearest_indices]
    # Choose the majority class label
    prediction = max(set(nearest_labels), key=nearest_labels.count)
    return prediction

train_df = pd.read_csv('merged_data.csv')
print(train_df.shape)
#print(train_df)
train_text=list(train_df["text"].values)
training_graphs = [ make_graph(text) for text in train_text]
training_labels = list(train_df["label"].values)

graph = GraphKNN(k=3)
graph.fit(training_graphs, training_labels)

test_df = pd.read_csv('merged_test_data.csv')
#test_graphs = [ make_graph(text) for text in list(test_df["text"].values)]
test_graphs = [make_graph(text) for text in list(test_df["text"].values) if text is not None]

test_labels = list(test_df["label"].values)

# Predict labels for all test graphs
predictions = [graph.predict(test_graph) for test_graph in test_graphs]
# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
accuracy_percentage = accuracy * 100
print("Accuracy:", accuracy_percentage, "%")
# Calculate F1 Score
f1_s = f1_score(test_labels, predictions, average=None)
f1_s *= 100
print(f'F1 Score is {f1_s[0]}') # The Lowest Confidence Bound
# Calculate the jaccard similarity
jaccard = jaccard_score(test_labels, predictions, average=None)
jaccard *= 100
print(f'Jaccard Similarity is {jaccard[1]}')

conf_matrix = confusion_matrix(test_labels, predictions)
# Plot

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(test_labels), yticklabels=set(test_labels))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
