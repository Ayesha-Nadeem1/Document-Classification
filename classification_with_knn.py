# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:43:38 2024

@author: Ayesha Nadeem
"""


import os
import pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')

# Define preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words and perform stemming
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    
    return filtered_tokens

# Define graph construction function with preprocessing
def construct_graph(text):
    # Preprocess text
    terms = preprocess(text)
    
    # Create directed graph
    graph = {}
    for i in range(len(terms)-1):
        term1 = terms[i]
        term2 = terms[i+1]
        if term1 not in graph:
            graph[term1] = [term2]
        else:
            graph[term1].append(term2)
    return graph

# Read data from the train folder
def read_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            data.append(df)
    print(pd.concat(data, ignore_index=True))
    return pd.concat(data, ignore_index=True)

train_data = read_data("Train")

# Construct graphs for each document
train_data['graph'] = train_data['text'].apply(construct_graph)

# Define function to find common subgraphs
def find_common_subgraphs(graphs):
    subgraphs = Counter()
    for graph in graphs:
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                subgraph = (node, neighbor)
                subgraphs[subgraph] += 1
    # Filter common subgraphs based on frequency threshold
    frequency_threshold = 5  # You can adjust this threshold as needed
    common_subgraphs = [subgraph for subgraph, count in subgraphs.items() if count >= frequency_threshold]
    return common_subgraphs

# Find common subgraphs in the training set
common_subgraphs = find_common_subgraphs(train_data['graph'])

# Define function to compute maximal common subgraph
def compute_mcs(graph1, graph2):
    common_nodes = set(graph1.keys()) & set(graph2.keys())
    mcs = len(common_nodes)
    return mcs

# Define KNN classification function
def knn_classification(test_graph, train_data, k):
    distances = []
    for _, train_row in train_data.iterrows():
        train_graph = train_row['graph']
        distance = compute_mcs(test_graph, train_graph)
        distances.append((distance, train_row['topic']))
    distances.sort(reverse=True)  # Sort distances in descending order
    neighbors = distances[:k]  # Select top k nearest neighbors
    counts = Counter(neighbor[1] for neighbor in neighbors)
    predicted_topic = counts.most_common(1)[0][0]  # Get the most common topic
    return predicted_topic

# Define function to evaluate performance
def evaluate_performance(test_data, train_data, k):
    y_true = []
    y_pred = []
    for _, test_row in test_data.iterrows():
        test_graph = test_row['graph']
        predicted_topic = knn_classification(test_graph, train_data, k)
        y_true.append(test_row['topic'])
        y_pred.append(predicted_topic)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm

# Read data from the test folder
def read_test_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Assuming you have split your test data into separate documents
test_data = read_test_data("Test")
accuracy, precision, recall, f1, cm = evaluate_performance(test_data, train_data, k=5)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(test_data['topic']), yticklabels=set(test_data['topic']))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
