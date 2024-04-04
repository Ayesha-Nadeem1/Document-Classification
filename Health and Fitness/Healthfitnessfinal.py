import requests
from bs4 import BeautifulSoup
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

# Function to lemmatize title
def lemmatize_title(title):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(title)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_title = ' '.join(lemmatized_words)
    return lemmatized_title

# Define the URL
url = "https://www.aliexpress.com/w/wholesale-health-and-fitness-equipment.html?page=62&g=y&SearchText=health+and+fitness+equipment"

# Send a GET request
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all div elements with the specified class
divs = soup.find_all('div', class_='list--gallery--C2f2tvm search-item-card-wrapper-gallery')

# Open the CSV file in append mode if it exists, otherwise create a new file
file_exists = os.path.isfile('sanitized_data_14.csv')
with open('sanitized_data_14.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # If the file doesn't exist, write the header
    if not file_exists:
        writer.writerow(['Title'])
    
    # Loop through each div and extract information
    for div in divs:
        # Extract the title
        title_div = div.find('div', class_='multi--content--11nFIBL').find('div', class_='multi--title--G7dOCj3').h3
        title = title_div.text.strip() if title_div else "N/A"
        
        # Lemmatize the title
        lemmatized_title = lemmatize_title(title)
        
        # Write the lemmatized title to the CSV file
        writer.writerow([lemmatized_title])

print("Sanitized data has been written to sanitized_data.csv")
