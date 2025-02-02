import pandas as pd
import requests

# Ρυθμίσεις LLM Studio API
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"  # Προσαρμόστε την URL βάση των ρυθμίσεων σας

# Φόρτωση δεδομένων από Excel
excel_file = "indices.test.xlsx"  # Αντικαταστήστε με το όνομα του αρχείου
data = pd.read_excel(excel_file)

# Λίστα για αποθήκευση επεξεργασμένων δεδομένων
processed_data = []

# Μεταβλητή για παρακολούθηση του τρέχοντος θέματος
current_topic = None

# Επεξεργασία γραμμών
for _, row in data.iterrows():
    index_value = str(row["Index"]).strip()
    description_value = str(row["Description"]).strip()

    # Αν είναι θέμα (δεν είναι αριθμημένο και δεν έχει περιγραφή)
    if not index_value.startswith("(") and pd.isna(description_value):
        current_topic = index_value  # Ενημερώνουμε το τρέχον θέμα

    # Αν είναι υποθέμα (αριθμημένο και έχει περιγραφή)
    elif index_value.startswith("(") and not pd.isna(description_value):
        processed_data.append({
            "Topic": current_topic,
            "Subtopic": index_value,
            "Description": description_value
        })

# Μετατροπή των δεδομένων σε DataFrame
processed_df = pd.DataFrame(processed_data)


def clean_llm_response(llm_response):
    """
    Καθαρίζει την έξοδο του LLM για να διατηρήσει μόνο το query.
    Αφαιρεί περιττές εισαγωγές, εξηγήσεις ή κενά.
    """
    # Αφαιρούμε κενά και ειδικούς χαρακτήρες
    cleaned_response = llm_response.strip()

    # Αν υπάρχουν πολλαπλές γραμμές, κρατάμε την πρώτη μόνο
    cleaned_response = cleaned_response.split("\n")[0]

    # Αφαιρούμε πιθανά σημεία στίξης στο τέλος
    cleaned_response = cleaned_response.rstrip(".")

    return cleaned_response


# Λειτουργία για τη δημιουργία ενός semantic search query μέσω API
def get_semantic_search_query(subindex_name):
    # Prompt για αυστηρά έως 3 queries
    prompt = (
        f"You are an expert in sustainability auditing and semantic search, with extensive experience in analyzing company reports on Sustainable Development Goals (SDGs). \n\n"
        f"Your task is to generate highly specific, concise, and relevant semantic search keywords based on the provided subindex to enable accurate retrieval of relevant text segments.\n\n"
        f"- Focus on keywords that are contextually relevant to the subindex.\n"
        f"- Ensure that the keywords are aligned with the subindex's theme and are tailored to retrieve highly specific information.\n"
        f"- Provide strictly only 3 keywords in a comma-separated format. Avoid including any explanations, introductory text, or the subindex itself.\n\n"
        f"Subindex: \"{subindex_name}\"\n\n"
        f"Output: Only the keywords as a comma-separated list.Do not include the word Keywords."
    )

    # Αποστολή στο LLM
    payload = {
        "prompt": prompt,
        "max_tokens": 100,  # Επαρκές για έως 3 queries
        "temperature": 0.0
    }
    response = requests.post(LM_STUDIO_API_URL, json=payload)
    if response.status_code == 200:
        raw_response = response.json().get("choices", [{}])[0].get("text", "").strip()
        return clean_llm_response(raw_response)  # Καθαρισμός και διαχείριση των queries
    else:
        raise Exception(f"Error from LLM API: {response.status_code}, {response.text}")

# Λίστα για αποθήκευση των semantic search απαντήσεων
semantic_search_responses = []

# Σειριακή επεξεργασία κάθε γραμμής
for _, row in processed_df.iterrows():
    subindex_name = row["Subtopic"]
    description = row["Description"]
    try:
        print(f"Generating semantic search query for Subindex: {subindex_name}")
        llm_query = get_semantic_search_query(subindex_name)  # Ένα query ανά υποθέμα
        semantic_search_responses.append({
            "Subindex": subindex_name,
            "Description": description,
            "LLM Query": llm_query
        })
    except Exception as e:
        print(f"Error for Subindex: {subindex_name} - {e}")
        semantic_search_responses.append({
            "Subindex": subindex_name,
            "Description": description,
            "LLM Query": f"Error: {e}"
        })

# Αποθήκευση των semantic search queries σε αρχείο
semantic_output_file = "semantic_search_queries.xlsx"
pd.DataFrame(semantic_search_responses).to_excel(semantic_output_file, index=False)

print(f"Τα semantic search queries αποθηκεύτηκαν στο: {semantic_output_file}")
