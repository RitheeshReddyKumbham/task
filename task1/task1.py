import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import faiss
import requests
from bs4 import BeautifulSoup

def extract_pdf_text(file_path):
    pdf_reader = PyPDF2.PdfFileReader(open(file_path, 'rb'))
    text_chunks = []
    for page in range(pdf_reader.getNumPages()):
        text = pdf_reader.getPage(page).extractText().strip()
        text_chunks.extend(text.split('\n'))  
    return text_chunks

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

from transformers import pipeline

def generate_response(query, context_chunks, model_name='gpt-4', max_length=150):
    context = " ".join(context_chunks)
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuery: {query}\nAnswer:"
    
    generator = pipeline('text-generation', model=model_name)
    response = generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True)
    
    return response[0]['generated_text']

# Example usage
pdf_file_path = 'example.pdf'
query = "unemployment information by degree type"

chunks = extract_pdf_text(pdf_file_path)
embeddings = embed_chunks(chunks)
store_embeddings(embeddings, chunks)

indices, distances = search_query(query)
retrieved_chunks = [chunks[i] for i in indices]

response = generate_response(query, retrieved_chunks)
print("Response:")
print(response)
