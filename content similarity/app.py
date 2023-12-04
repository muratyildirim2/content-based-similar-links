from flask import Flask, render_template
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

def get_links_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [loc.text for loc in soup.find_all('loc')]
        return links
    except requests.exceptions.RequestException as e:
        return []

def get_text_from_link(link_url):
    try:
        response = requests.get(link_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        h1_text = soup.find('h1').get_text(separator=' ') if soup.find('h1') else ''
        h2_text = soup.find('h2').get_text(separator=' ') if soup.find('h2') else ''

        header_text = f"{h1_text} {h2_text}"
        
        return header_text
    except requests.exceptions.RequestException as e:
        return f"Error fetching content from {link_url}: {e}"



def calculate_similarity(text1, text2, model):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity

def get_most_similar_links(links, model, max_links=50, top_k=15):
    most_similar_links = []

    similarity_scores = []

    for i, link1 in enumerate(links[:max_links]):
        for j, link2 in enumerate(links[i+1:max_links]):
            text1 = get_text_from_link(link1)
            text2 = get_text_from_link(link2)
            similarity = calculate_similarity(text1, text2, model)
            similarity_scores.append((link1, link2, similarity))

    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    most_similar_links = similarity_scores[:top_k]

    return most_similar_links

@app.route('/')
def index():
    sitemap_url = 'https://www.f5haber.com/export/sitemap'
    links = get_links_from_sitemap(sitemap_url)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    most_similar_links = get_most_similar_links(links, model)

    return render_template('index.html', most_similar_links=most_similar_links)

if __name__ == '__main__':
    app.run(debug=True)
