import pandas as pd
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import os
import requests
import json
import nltk
nltk.download('punkt')

def load_config(config_file):
    """Loads the JSON configuration file."""
    try:
        print(f"Loading configuration from {config_file}...")
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file {config_file}: {e}")

def preprocess_text(text):
    """Removes extra whitespace and empty lines from text."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())

def extract_text(pdf_path):
    """Extracts and preprocesses text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    extracted_text = [preprocess_text(page.get_text()) for page in doc]
    doc.close()
    return "\n".join(extracted_text)

def split_into_segments(text, max_tokens):
    """Splits text into segments of a specified maximum token length."""
    print(f"Splitting text into segments with max {max_tokens} tokens each...")
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def create_or_load_chromadb_collection(segments, collection_name="pdf_segments"):
    """Creates or loads a ChromaDB collection and stores the text segments."""
    print(f"Creating or loading ChromaDB collection: {collection_name}...")
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        print("Adding segments to collection...")
        collection.add(documents=segments, ids=[str(i) for i in range(len(segments))])
    return collection

def preprocess_query(query):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(token) for token in word_tokenize(query.lower()))

def normalize_scores(results):
    distances = results["distances"][0]
    max_dist, min_dist = max(distances), min(distances)
    return list(zip(results["documents"][0], [(max_dist - d) / (max_dist - min_dist) for d in distances]))

def query_chromadb_collection(collection, query, top_k):
    """Queries the ChromaDB collection and returns the top matching segments."""
    print(f"Querying ChromaDB for: {query} (top {top_k} results)...")
    results = collection.query(query_texts=[preprocess_query(query)], n_results=top_k)
    return normalize_scores(results)

def query_llm(content, subindex_name, description, temperature, llm_endpoint):
    """Queries an external LLM API to assess compliance with sustainability criteria."""
    print(f"Querying LLM for subindex: {subindex_name}...")
    prompt = (
        f"""
        You are an experienced sustainability researcher. I will give you the following:

        1. Subindex: "{subindex_name}"
        2. Description: "{description}"
        3. Content: "{content}"
        Your task is to evaluate how relevant is the given subindex and description to the content.
        Please provide a compliance score from 0 (completely irrelevant) to 10 (fully relevant).
        Additionally, answer with Yes or No if the response is relevant.

        Your answer should be in the following format:
        Compliance Score: [score]
        Relevant: [Yes/No]
        """
    )

    payload = {"prompt": prompt, "temperature": temperature}
    try:
        response = requests.post(url=llm_endpoint, json=payload)
        response.raise_for_status()
        response_data = response.json()
        answer_text = response_data.get("choices", [{}])[0].get("text", "").strip()
        compliance_line = next((line for line in answer_text.split("\n") if "Compliance Score" in line), "")
        relevant_line = next((line for line in answer_text.split("\n") if "Relevant:" in line), "")
        return f"{compliance_line} {relevant_line}"
    except requests.RequestException as e:
        return {"error": f"Error querying LLM: {str(e)}"}

def process_excel_and_search(config_file):
    """Processes an Excel file, searches the extracted text, and queries an LLM."""
    config = load_config(config_file)
    pdf_path, excel_path, output_dir, llm_endpoint = (
        config.get("pdf_path"), config.get("excel_path"), config.get("output_dir"), config.get("llm_endpoint")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

        
    print("Processing Excel file and searching documents...")
    text = extract_text(pdf_path)
    segments = split_into_segments(text, config.get("max_tokens", 50))
    collection = create_or_load_chromadb_collection(segments)
    df = pd.read_excel(excel_path)
    results_by_subindex = {}

    for index, row in df.iterrows():
        subindex, description, llm_queries = row["Subindex"], row["Description"], row["LLM Query"].split(",")
        paragraph_counter = Counter()
        for query in [subindex] + llm_queries:
            for segment, score in query_chromadb_collection(collection, query, config.get("top_k", 20)):
                paragraph_counter[segment] += 1

        combined_text = "\n".join(paragraph for paragraph, _ in paragraph_counter.most_common(config.get("paragraph_limit", 20)))
        results_by_subindex[subindex] = combined_text

        with open(os.path.join(output_dir, f"{subindex}.txt"), "w", encoding="utf-8") as f:
            f.write(combined_text)

        llm_response = query_llm(combined_text, subindex, description, config.get("temperature", 0.2), llm_endpoint)
        if llm_response:
            with open(os.path.join(output_dir, f"llm_score_{subindex}.txt"), "w", encoding="utf-8") as f:
                json.dump(llm_response, f, indent=4)

    return results_by_subindex

if __name__ == "__main__":
    print("Starting processing...")
    results = process_excel_and_search("config_2.json")
    for subindex, combined_text in results.items():
        print(f"Subindex: {subindex}\n{combined_text}")
    print("Processing complete.")
