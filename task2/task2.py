import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline


def crawl_and_extract_text(urls, max_pages=3):
    all_chunks = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_chunks = [p.get_text().strip() for p in paragraphs]
        all_chunks.extend(text_chunks)
    return all_chunks


def embed_chunks(chunks, model_name='bert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    return embeddings


def store_embeddings(embeddings, metadata, index_file='faiss_index'):
    index = faiss.IndexFlatL2(embeddings.size(1))  
    index.add(embeddings.cpu().numpy())
    faiss.write_index(index, index_file)


def search_query(query, model_name='bert-base-nli-mean-tokens', index_file='faiss_index', top_k=5):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    index = faiss.read_index(index_file)
    D, I = index.search(query_embedding.cpu().numpy(), top_k)
    
    return I.flatten(), D.flatten()


def generate_response(query, context_chunks, model_name='gpt-4', max_length=150):
    context = " ".join(context_chunks)
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuery: {query}\nAnswer:"
    
    generator = pipeline('text-generation', model=model_name)
    response = generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True)
    
    return response[0]['generated_text']

# Example usage
urls = [
    'https://www.uchicago.edu/',
    'https://www.washington.edu/',
    'https://www.stanford.edu/',
    'https://und.edu/'
]
query = "campus facilities information"

chunks = crawl_and_extract_text(urls)
embeddings = embed_chunks(chunks)
store_embeddings(embeddings, chunks)

indices, distances = search_query(query)
retrieved_chunks = [chunks[i] for i in indices]

response = generate_response(query, retrieved_chunks)
print("Response:")
print(response)
