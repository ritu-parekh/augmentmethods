import requests
import random
import os
from bs4 import BeautifulSoup

# Base URLs
search_url = "https://openclipart.org/search"
image_base_url = "https://openclipart.org"


for i in range(5):
    # Choose a random page (up to ~5669 according to HTML)
    page_num = random.randint(1, 5669)
    params = {'p': page_num}

    # Make the request
    response = requests.get(search_url, params=params)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all image URLs
    image_tags = soup.select('.gallery .artwork img')
    image_urls = [
        image_base_url + tag['src'] for tag in image_tags if tag.get('src')
    ]

    # Shuffle and pick a few random ones
    random.shuffle(image_urls)
    downloaded = []

    for i, img_url in enumerate(image_urls[:15]):  # Download first 
        filename = f"clipart/clipart_{page_num}{i+1}.png"
        img_data = requests.get(img_url).content
        with open(filename, 'wb') as f:
            f.write(img_data)
        downloaded.append(filename)

    print(f"Downloaded: {downloaded}")
