import logging
import json
import fitz
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import asyncio
import aiohttp
import nltk

# Ενεργοποίηση του NLTK tokenizer
nltk.download('punkt')

# Ρύθμιση Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Φόρτωση ρυθμίσεων από config_2.json
def load_config(config_path="config_1a.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()

LLAMA_API_URL = config["LLAMA_API_URL"]
BATCH_SIZE = config["BATCH_SIZE"]
HEADER_HEIGHT_RATIO = config["HEADER_HEIGHT_RATIO"]
MAX_TOKENS = config["MAX_TOKENS"]
PDF_PATH = config["PDF_PATH"]
EXCEL_PATH = config["EXCEL_PATH"]
RESULTS_PATH = config["RESULTS_PATH"]
SEGMENTS_PATH = config["SEGMENTS_PATH"]


def preprocess_text(text):
    lines = text.splitlines()
    processed_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(processed_lines)


# Εξαγωγή κειμένου από PDF χωρίς headers
def extract_text_without_headers(pdf_path: str, header_height_ratio: float) -> str:
    logging.info(f"Extracting text from PDF: {pdf_path}")
    extracted_text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page in pdf_document:
            page_rect = page.rect
            header_rect = fitz.Rect(0, 0, page_rect.width, page_rect.height * header_height_ratio)

            filtered_blocks = [block[4] for block in page.get_text("blocks") if
                               not fitz.Rect(block[:4]).intersects(header_rect)]
            extracted_text += "\n".join(filtered_blocks) + "\n"
        pdf_document.close()
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
    logging.info("Text extraction completed successfully.")
    return preprocess_text(extracted_text)


# Τμηματοποίηση του κειμένου
def segment_text(text: str, max_tokens: int) -> list:
    logging.info("Segmenting text...")
    sentences = sent_tokenize(text)
    segments, current_segment, token_count = [], [], 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if token_count + len(tokens) > max_tokens:
            segments.append(' '.join(current_segment))
            current_segment, token_count = [], 0
        current_segment.append(sentence)
        token_count += len(tokens)

    if current_segment:
        segments.append(' '.join(current_segment))
    logging.info(f"Created {len(segments)} segments.")
    return segments


# Αποθήκευση segments σε TXT
def save_segments_to_txt(segments: list, output_path: str):
    logging.info(f"Saving segments to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            f.write(f"Segment {i + 1}:\n{segment}\n{'=' * 50}\n")
    logging.info("Segments saved successfully.")


# Ανάγνωση δεδομένων από Excel
def process_excel_data(excel_path: str) -> pd.DataFrame:
    logging.info(f"Reading Excel file: {excel_path}")
    data = pd.read_excel(excel_path)
    logging.info(f"Loaded {len(data)} rows from Excel file.")
    return data.dropna(subset=["Index", "Description"])


# Αποστολή αιτήματος στο API
async def make_request_async(payload: dict) -> dict:
    logging.info("Sending request to LM Studio API...")
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=260)) as session:
        async with session.post(LLAMA_API_URL, json=payload, headers=headers) as response:
            response.raise_for_status()
            logging.info("Request completed successfully.")
            return await response.json()


# Δημιουργία payload
def generate_payload(prompt: str) -> dict:
    return {"prompt": prompt, "stream": False, "temperature": 0.0, "max_tokens": 1000}


# Επεξεργασία segment για subindex
async def process_segment_for_subindex(segment: str, subindex_name: str, description: str) -> str:
    logging.info(f"Processing segment for subindex: {subindex_name}")
    prompt = (
        f"""You are an expert in sustainability auditing. Your task is to extract sentences from the provided text 
            that directly relate to the subindex '{subindex_name}'. 

            The description of '{subindex_name}' is: '{description}'.

            If no relevant sentences are found in the text, return an empty response. 
            Do not include any generic statements, explanations, or questions. 

            Provided Text: {segment}
            """
    )
    response = await make_request_async(generate_payload(prompt))
    return response.get("choices", [{}])[0].get("text", "").strip() if response else ""


# Παράλληλη επεξεργασία όλων των segments για subindex
async def process_all_segments_for_subindex(segments: list, subindex_name: str, description: str, results_file: str):
    logging.info(f"Processing all segments for Subindex: {subindex_name}")
    batches = [segments[i:i + BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"Subindex: {subindex_name}\nDescription: {description}\nRelevant Sentences:\n")

        for batch in batches:
            results = await asyncio.gather(
                *[process_segment_for_subindex(seg, subindex_name, description) for seg in batch])
            f.writelines(f"- {res}\n" for res in results if res)
        f.write("=" * 50 + "\n")
    logging.info(f"Completed processing for Subindex: {subindex_name}")


# Κύρια συνάρτηση
async def main():
    logging.info("Starting main process...")
    text = extract_text_without_headers(PDF_PATH, HEADER_HEIGHT_RATIO)
    segments = segment_text(text, MAX_TOKENS)
    save_segments_to_txt(segments, SEGMENTS_PATH)
    df = process_excel_data(EXCEL_PATH)

    for _, row in df.iterrows():
        logging.info(f"Processing subindex: {row['Index']}")
        await process_all_segments_for_subindex(segments, row["Index"], row["Description"], RESULTS_PATH)
    logging.info("Main process completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
