from datasets import load_dataset
import textwrap
import json
import itertools
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Load the dataset
middle_english_pd = load_dataset(
    "PleIAs/Middle-English-PD", 
    data_files=[f"middle_english_pd_{i}.parquet" for i in range(1, 6)], 
    split="train"
)

# print some general information about the dataset
print("Dataset Summary:")
print(f"The dataset has {len(middle_english_pd)} examples, and the publication dates of the dataset range between {min(middle_english_pd['publication_date'])} and {max(middle_english_pd['publication_date'])}.")

# write a function to print an example from the dataset in a readable format
def print_wrapped_item(item, limit=80, trim_length=200):
    def wrap_text(text, width):
        return "\n".join(textwrap.fill(line, width) for line in text.splitlines())

    def trim_value(value, length):
        if isinstance(value, str) and len(value) > length:
            return value[:length] + "..."
        return value
    
    # Prepare the item for pretty-printing with trimmed values
    trimmed_item = {k: trim_value(v, trim_length) for k, v in item.items()}
    pretty_json = json.dumps(trimmed_item, ensure_ascii=False, indent=4)
    
    # Split the JSON string into lines and wrap each line
    wrapped_lines = []
    for line in pretty_json.splitlines():
        wrapped_lines.extend(wrap_text(line, limit).splitlines())

    # Print the wrapped lines
    for line in wrapped_lines:
        print(line)

print("Example from the dataset:")
print_wrapped_item(middle_english_pd[0])

# Transform the dataset into a list of dictionaries
middle_english_pd = [
    {
        "id": entry["identifier"],
        "text": entry["title"] + "\n\n" + entry["text"],
        "source": "PleIAs/Middle-English-PD"
    }
    for entry in tqdm(middle_english_pd, desc="Transforming dataset")
]

print(f"And the total number of words in the dataset is {round(sum(len(entry['text'].split()) for entry in middle_english_pd) / 1e6, 2)} million.")

def process_document(entry, n_low, n_high):
    document_id = entry["id"]
    text = entry["text"]

    # TODO: Improve upon the text processing pipeline
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # remove special characters 
    sentences = [re.sub(r"[^a-zA-Z0-9 ]", "", sentence) for sentence in sentences]
    # remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]
    # make all sentences lowercase
    sentences = [sentence.lower() for sentence in sentences]

    # Initialize a dictionary for the current document's n-grams
    document_ngram_dict = defaultdict(set)

    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = sentence.split()

        # Generate n-grams for the given range (n_low to n_high)
        for n in range(n_low, n_high + 1):
            ngrams = list(itertools.zip_longest(*[words[i:] for i in range(n)]))
            ngrams = [" ".join(ngram).strip() for ngram in ngrams if None not in ngram]

            for ngram in ngrams:
                document_ngram_dict[ngram].add(document_id)

    return document_ngram_dict

"""Converts the dataset into n-grams and saves the results to a file.
"""
def n_grams(dataset, n_low=2, n_high=5, file_path="../data/corpus/sentence_ngrams.json"):
    # Initialize a dictionary to store combined n-grams for all documents
    combined_ngram_dict = defaultdict(lambda: {"count": 0, "documents": set()})

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, entry, n_low, n_high) for entry in dataset]

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            document_ngram_dict = future.result()

            # Combine document-level n-grams into the global combined dictionary
            for ngram, document_ids in document_ngram_dict.items():
                combined_ngram_dict[ngram]["count"] += len(document_ids)
                combined_ngram_dict[ngram]["documents"].update(document_ids)

    # Finalize the combined_ngram_dict to make it JSON serializable
    processed_data = {
        ngram: {
            "count": data["count"],
            "documents": list(data["documents"])
        }
        for ngram, data in combined_ngram_dict.items()
    }

    # Save the processed data to a file
    with open(file_path, "w") as f:
        json.dump(processed_data, f, indent=4)

# Usage example:
n_grams(middle_english_pd[:200], n_low=2, n_high=5, file_path="../data/corpus/middle_english_pd_sentence_ngrams.json")

# clear memory
gc.collect()
# clear large variables
del middle_english_pd

print("Data collection completed!")
