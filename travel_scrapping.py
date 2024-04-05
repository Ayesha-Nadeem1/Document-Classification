
import requests
from bs4 import BeautifulSoup
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')

nltk.download('wordnet')

def sanitize_text(text, stopwords_dict, lemmatizer):
    tokens = word_tokenize(text)  # Tokenize the text
    # Remove stopwords, non-alphanumeric characters, and lemmatize words
    sanitized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stopwords_dict]
    return ' '.join(sanitized_tokens)

def scrape_travel_data(url, word_count,stopwords_dict):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Assuming text is contained within <p> tags
        text_elements = soup.find_all('p')
        
        # Extract text from elements and concatenate into a single string
        text_data = ' '.join(element.get_text() for element in text_elements if element.get_text() not in stopwords_dict)
        
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
    
    
    # for i in range(1, 16):  # Creating 15 files
    #     filename = f"travel_words_{i}.csv"
    #     file_path = os.path.join(folder, filename)
        
    #     with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for j in range(500):  # Writing 500 words to each file
    #             writer.writerow([data[(i-1)*500 + j]])
    
    # num_files = 15
    # words_per_file = 500
    
    # for i in range(num_files):
    #     filename = f"travel_words_{i+1}.csv"
    #     file_path = os.path.join(folder_path, filename)
        
    #     with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         start_index = i * words_per_file
    #         end_index = min((i + 1) * words_per_file, len(data))
    #         for j in range(start_index, end_index):
    #             writer.writerow([data[j]])

#multiple URLs
urls = [
    'https://www.lonelyplanet.com/france/paris',
    'https://www.lonelyplanet.com/england/london',
    'https://www.lonelyplanet.com/japan/tokyo',
    'https://www.lonelyplanet.com/pakistan',
    'https://www.lonelyplanet.com/india',
    'https://www.lonelyplanet.com/greece',
    'https://www.lonelyplanet.com/oman',
    'https://www.lonelyplanet.com/morocco',
    'https://www.lonelyplanet.com/brazil',
    'https://www.lonelyplanet.com/fiji'
    
    
]
word_count = 7500

# Define the path for the CSV file
csv_folder_path = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel words'
csv_file_path = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel_words.csv'
csv_file_path_sanitized = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel_sanitized.csv'
stopwords_dict = set(stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
# Loop through each URL and scrape data
for url in urls:
    print(f"Scraping data from URL: {url}")
    travel_words = scrape_travel_data(url, word_count,stopwords_dict)
    
    # Print the number of words scraped
    print(f"Scraped {len(travel_words)} words from URL: {url}")
    #open('travel_words.csv', 'w').close()
    # Append data to CSV if there are words scraped
    if travel_words:
        append_to_csv(travel_words, csv_file_path)
        print("Data appended to CSV file.")
    else:
        print("No words scraped from this URL.")
        
with open(csv_file_path, 'r', newline='', encoding='utf-8') as input_file, \
     open(csv_file_path_sanitized, 'w', newline='', encoding='utf-8') as output_file:
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

    for row in csv_reader:
        if row:  # Check if the row is not empty
            sanitized_text = sanitize_text(row[0], stopwords_dict, lemmatizer)
            if sanitized_text:  # Check if sanitized text is not empty
                csv_writer.writerow([sanitized_text])

print("Sanitized data written to:", csv_file_path_sanitized)

input_csv_file_path = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel_sanitized.csv'
output_folder_path = r'C:\Users\Ayesha Nadeem\OneDrive\Documents\semester 6\GT\travel words'

# Function to write a chunk of data to a CSV file
def write_chunk_to_csv(chunk, file_number):
    output_file_path = os.path.join(output_folder_path, f'travel_words_{file_number}.csv')
    with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        for word in chunk:
            csv_writer.writerow([word])

# Read data from input CSV file and split into chunks of 500 words each
with open(input_csv_file_path, 'r', newline='', encoding='utf-8') as input_file:
    csv_reader = csv.reader(input_file)
    all_words = [row[0] for row in csv_reader]

# Calculate number of chunks
num_chunks = len(all_words) // 500 + (len(all_words) % 500 > 0)

# Write each chunk to a separate CSV file
for i in range(num_chunks):
    start_index = i * 500
    end_index = min((i + 1) * 500, len(all_words))
    chunk = all_words[start_index:end_index]
    write_chunk_to_csv(chunk, i + 1)

print("Data split into chunks and written to separate CSV files.")
