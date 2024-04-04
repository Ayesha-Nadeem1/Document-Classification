# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:49:40 2024

@author: HP
"""

from bs4 import BeautifulSoup
import requests
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os.path
import nltk

# Set NLTK data path and download necessary resources
nltk.data.path.append("D:\\nltk_data")  # Change this to your desired directory
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize text
def lemmatize_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Fetch the HTML content from the URL
url = "https://www.mayoclinic.org/diseases-conditions/index?letter=Y"
response = requests.get(url)
html_content = response.text

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find all <a> tags within <div class="cmp-result-name"> and <p> tags within <div class="cmp-results-with-primary-name">
links = soup.find_all('div', class_='cmp-result-name')
secondary_names = soup.find_all('div', class_='cmp-results-with-primary-name')

# Open the CSV file in append mode if it exists, otherwise create a new file
file_exists = os.path.isfile('disease_data_2.csv')
with open('disease_data_2.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # If the file doesn't exist, write the header
    if not file_exists:
        writer.writerow(['Primary Name', 'Secondary Name'])
    
    # Iterate over each link and associated secondary name
    for link, secondary_name in zip(links, secondary_names):
        primary_name = link.a.text
        
        # Extracting the secondary name and its associated URL, if available
        secondary_name_text = secondary_name.p.text if secondary_name.p else None
        
        # Lemmatize the primary and secondary names
        lemmatized_primary_name = lemmatize_text(primary_name)
        lemmatized_secondary_name = lemmatize_text(secondary_name_text) if secondary_name_text else None
        
        # Write the extracted and lemmatized data to the CSV file
        writer.writerow([lemmatized_primary_name, lemmatized_secondary_name])

print("Data has been written to disease_data.csv")
