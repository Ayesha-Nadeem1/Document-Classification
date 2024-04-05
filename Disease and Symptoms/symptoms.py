import requests
from bs4 import BeautifulSoup
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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

# Define the URL
url = "https://www.mayoclinic.org/diseases-conditions/kidney-cancer/symptoms-causes/syc-20352664"

# Send a GET request
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all <ul> elements
ul_elements = soup.find_all('ul')

# Check if there are at least 7 <ul> elements
if len(ul_elements) >= 7:
    # Get the 7th <ul> element
    seventh_ul = ul_elements[6]  # Indexing starts from 0

    # Initialize a list to store list items
    list_items = []

    # Extract and lemmatize the text from list items
    for li in seventh_ul.find_all('li'):
        list_item_text = li.get_text()
        lemmatized_text = lemmatize_text(list_item_text)
        list_items.append(lemmatized_text)

    # Write the list items to a CSV file
    file_exists = os.path.isfile('disease_data_14.csv')
    with open('disease_data_14.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(['List Item'])
        # Write the list items to the CSV file
        for item in list_items:
            writer.writerow([item])

    print("List items have been written to list_items.csv")
else:
    print("There are less than 7 <ul> elements on the page.")
