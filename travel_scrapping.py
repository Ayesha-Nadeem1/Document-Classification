# 



import requests
from bs4 import BeautifulSoup
import csv
import os

def scrape_travel_data(url, word_count):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Assuming text is contained within <p> tags
        text_elements = soup.find_all('p')
        
        # Extract text from elements and concatenate into a single string
        text_data = ' '.join(element.get_text() for element in text_elements)
        
        # Split text into words
        words = text_data.split()
        
        # Take only the required number of words
        selected_words = words[:word_count]
        
        return selected_words
    except Exception as e:
        print(f"Error occurred while scraping {url}: {e}")
        return []

# Function to append data to CSV
def append_to_csv(data, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for word in data:
            writer.writerow([word])

# Example usage with multiple URLs
urls = [
    'https://www.lonelyplanet.com/france/paris',
    'https://www.lonelyplanet.com/england/london',
    'https://www.lonelyplanet.com/japan/tokyo',
    'https://www.lonelyplanet.com/pakistan'
    
]
word_count = 7500

# Define the path for the CSV file
csv_file_path = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel_words.csv'

# Loop through each URL and scrape data
for url in urls:
    print(f"Scraping data from URL: {url}")
    travel_words = scrape_travel_data(url, word_count)
    
    # Print the number of words scraped
    print(f"Scraped {len(travel_words)} words from URL: {url}")
    
    # Append data to CSV if there are words scraped
    if travel_words:
        append_to_csv(travel_words, csv_file_path)
        print("Data appended to CSV file.")
    else:
        print("No words scraped from this URL.")
