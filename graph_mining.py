import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix

# Read data from CSV files
health_df = pd.read_csv("train_health_data_preprocessed.csv")
disease_df = pd.read_csv("train_disease_data_preprocessed.csv")
travel_df = pd.read_csv("train_travel_data.csv")

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

# Prepare training data
train_texts = all_data['text'].tolist()
# Select the first 1000 labels
train_labels_first = all_data['label'][:1000].tolist()

# Calculate the start index for the next 1000 labels from the center
center_start_index = len(all_data) // 2 - 500

# Select the next 1000 labels from the center
train_labels_center = all_data['label'][center_start_index:center_start_index + 1000].tolist()

# Select the last 1500 labels
train_labels_last = all_data['label'][-1500:].tolist()

# Concatenate the three segments to get the final train labels
train_labels = train_labels_first + train_labels_center + train_labels_last
print(len(train_labels))
train_graphs = [make_graph(text) for text in train_texts]

# Train the model
graph_classifier = GraphKNN(k=3)
graph_classifier.fit(train_graphs, train_labels)

# Test data

# Specify the path to your test data CSV file
# Specify the path to your test data CSV file
test_data_path = "merged_test_data.csv"

# Read the test data from the CSV file
test_df = pd.read_csv(test_data_path)

# Assuming your CSV file has a column named 'text' containing the text samples
test_texts = test_df['text'].tolist()
test_labels = test_df['label'][:3500].tolist()
print(len(test_labels))
# Define a regular expression pattern to match words
pattern = re.compile(r'\b[A-Za-z]+\b')

# Extract only words from the text samples
filtered_texts = []
for text in test_texts:
    words = pattern.findall(text)
    filtered_texts.extend(words)

# Preprocess test data
test_texts = [preprocess(text) for text in filtered_texts]
# Make graphs for test data
test_graphs = [make_graph(text) for text in test_texts]

# Predict
predictions = [graph_classifier.predict(graph) for graph in test_graphs]
predictions = predictions[:len(test_labels)]

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)

# Calculate accuracy
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate F1 Score for the second class
f1_scores = f1_score(test_labels, predictions, average=None)
f1_score_percentage = f1_scores[1] * 100
print("F1 Score:", "{:.2f}%".format(f1_score_percentage))

# Calculate Jaccard similarity for the second class
jaccard = jaccard_score(test_labels, predictions, average=None)
jaccard_percentage = jaccard[1] * 100
print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(test_labels))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()