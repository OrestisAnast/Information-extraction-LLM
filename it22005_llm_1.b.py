import logging
import aiohttp
import asyncio
import re
import json

# Φόρτωση παραμέτρων από config_2.json
with open("config_1b.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

# Ρύθμιση Logging
logging.basicConfig(level=getattr(logging, config["log_level"], logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")

LLM_API_URL = config["LLM_API_URL"]


def generate_relevance_payload(subindex_name: str, description: str, llm_response: str) -> dict:
    prompt = (
        f"""
        You are an experienced sustainability researcher. I will give you the following:

        1. Subindex: "{subindex_name}"
        2. Description: "{description}"
        3. LLM Response: "{llm_response}"

        Your task is to evaluate how relevant the LLM Response is to the given subindex and description.
        Please provide a compliance score from 0 (completely irrelevant) to 10 (fully relevant).
        Additionally, answer with Yes or No if the response is relevant.

        Your answer should be in the following format:
        Compliance Score: [score]
        Relevant: [Yes/No]
        """
    )
    return {
        "prompt": prompt,
        "stream": False,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }


async def get_relevance_score(subindex_name: str, description: str, llm_response: str) -> dict:
    payload = generate_relevance_payload(subindex_name, description, llm_response)
    headers = {"Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=config["request_timeout"])

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(LLM_API_URL, json=payload, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


def parse_llm_response(response_text: str) -> dict:
    compliance_score_match = re.search(r"Compliance Score:\s*(\d+)", response_text)
    relevant_match = re.search(r"Relevant:\s*(Yes|No)", response_text, re.IGNORECASE)

    compliance_score = int(compliance_score_match.group(1)) if compliance_score_match else None
    relevant = relevant_match.group(1).capitalize() if relevant_match else "Unknown"

    return {
        "compliance_score": compliance_score,
        "relevant": relevant,
    }


def extract_llm_responses(entry: str) -> str:
    extracted_sentences = re.findall(r'\* "(.*?)"', entry, re.DOTALL)
    return " ".join(extracted_sentences)


async def evaluate_relevance_scores():
    logging.info(f"Reading input results file: {config['input_file']}")
    with open(config["input_file"], "r", encoding="utf-8") as file:
        data = file.read()

    final_results = []
    entries = data.split(config["delimiter"])

    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) < 2:
            continue

        subindex_name = lines[0].replace("Subindex:", "").strip()
        description = lines[1].replace("Description:", "").strip()
        llm_response = extract_llm_responses(entry)

        if not llm_response:
            logging.warning(f"No responses found for Subindex: {subindex_name}")
            continue

        logging.info(f"Evaluating relevance for Subindex: {subindex_name}")
        response = await get_relevance_score(subindex_name, description, llm_response)

        response_text = response.get("choices", [{}])[0].get("text", "").strip()
        parsed_response = parse_llm_response(response_text)

        final_results.append({
            "subindex": subindex_name,
            "description": description,
            "llm_response": llm_response,
            "compliance_score": parsed_response["compliance_score"],
            "relevant": parsed_response["relevant"],
        })

    logging.info(f"Saving final relevance evaluations to {config['output_file']}")
    with open(config["output_file"], "w", encoding="utf-8") as file:
        for result in final_results:
            file.write(f"Subindex: {result['subindex']}\n")
            file.write(f"Description: {result['description']}\n")
            file.write(f"Response: {result['llm_response']}\n")
            file.write(f"Compliance Score: {result['compliance_score']}\n")
            file.write(f"Relevant: {result['relevant']}\n")
            file.write(config["delimiter"])


if __name__ == "__main__":
    asyncio.run(evaluate_relevance_scores())