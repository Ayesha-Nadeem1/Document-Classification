# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:04:32 2024

@author: Ayesha Nadeem
"""
import pandas as pd
import re
import networkx as nx
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to make a directed graph from text
def make_graph(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return None
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)
    # Create a directed graph
    G = nx.DiGraph()
    # Add nodes for each unique word
    for word in set(words):
        G.add_node(word)
    # Add edges between adjacent words
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i + 1])
    return G

# Function to extract common subgraphs
def extract_common_subgraphs(graphs):
    subgraphs = []
    for graph in graphs:
        if graph is not None:  # Check if graph is not None
            # Extract all subgraphs
            undirected_graph = graph.to_undirected()
            for subgraph in nx.connected_components(undirected_graph):
                subgraphs.append(tuple(subgraph))  # Convert set to tuple
    # Count occurrences of each subgraph
    subgraph_counts = Counter(subgraphs)
    # Filter common subgraphs
    common_subgraphs = [set(subgraph) for subgraph, count in subgraph_counts.items() if count > 1]  # Convert back to set
    return common_subgraphs


# Function to compute graph distance (MCS)
def graph_distance(graph1, graph2):
    # Ensure that graph1 and graph2 are NetworkX graph objects
    if not isinstance(graph1, nx.Graph) or not isinstance(graph2, nx.Graph):
        raise ValueError("Input should be NetworkX graph objects")
    
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())
# Function to implement KNN classifier
class GraphKNN:
    def __init__(self, k):
        self.k = k
        self.train_graphs = []
        self.train_labels = []

    def fit(self, train_graphs, train_labels):
        # Convert input graphs to NetworkX graph objects
        self.train_graphs = [self._convert_to_networkx(graph) for graph in train_graphs]
        self.train_labels = train_labels

    def predict(self, graph):
        # Convert input graph to NetworkX graph object
        graph = self._convert_to_networkx(graph)
        
        distances = [graph_distance(graph, train_graph) for train_graph in self.train_graphs]
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        return prediction

    def _convert_to_networkx(self, graph):
        # If graph is already a NetworkX graph object, return it
        if isinstance(graph, nx.Graph):
            return graph
        
        # Otherwise, create a new NetworkX graph object
        G = nx.DiGraph()
        if isinstance(graph, list):  # Check if the input is a list
            for edge in graph:
                if isinstance(edge, tuple) and len(edge) == 2:  # Ensure each edge is a tuple of length 2
                    G.add_edge(edge[0], edge[1])
        return G

# Read merged data
merged_df = pd.read_csv("merged_data.csv")


# Separate features and labels
texts = merged_df["text"]

labels = merged_df["label"]

# Convert labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
labels = labels.map(label_mapping)

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
# print(train_texts)
# print(test_texts)

# Construct graphs for training and test data
train_graphs = [make_graph(text) for text in train_texts]
test_graphs = [make_graph(text) for text in test_texts]

# Extract common subgraphs from training set
common_subgraphs = extract_common_subgraphs(train_graphs)

# Convert common subgraphs to features
#train_features = [[int(common_subgraphs.has_node(node)) for node in common_subgraph.nodes()] for common_subgraph in common_subgraphs]
all_nodes = set()
for common_subgraph in common_subgraphs:
    all_nodes.update(common_subgraph)
train_features = []
for common_subgraph in common_subgraphs:
    feature = [int(node in common_subgraph) for node in all_nodes]  # Use 'in' operator to check membership
    train_features.append(feature)
    
    
#test_features = [[int(common_subgraph.has_node(node)) for node in common_subgraph.nodes()] for common_subgraph in common_subgraphs]
test_features = []
for common_subgraph in common_subgraphs:
    feature = [int(node in common_subgraph) for node in all_nodes]
    test_features.append(feature)

# Initialize and fit KNN classifier
k = 3  # You can change the value of k here
graph_classifier = GraphKNN(k)
graph_classifier.fit(train_features, train_labels)

# Predict labels for test data
predictions = [graph_classifier.predict(test_feature) for test_feature in test_features]

# Evaluate the classifier
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')
conf_matrix = confusion_matrix(test_labels, predictions)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
