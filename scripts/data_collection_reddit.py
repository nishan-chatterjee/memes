from datasets import load_dataset
import textwrap
import json
import itertools
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Load 10% of the dataset
reddit = load_dataset("webis/tldr-17", trust_remote_code=True, split="train[:10%]")
# print some general information about the dataset
print("Dataset Summary:")
print(f"The dataset has {round(len(reddit)/1e6, 2)} million examples, and the publication dates range between 2006 and 2016.")

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
print_wrapped_item(reddit[0])

# Transform the dataset into a list of dictionaries
reddit = [
    {
        "id": entry["id"],
        "text": entry["content"],
        "source": "webis/tldr-17"
    }
    for entry in tqdm(reddit, desc="Transforming dataset")
]

print(f"And the total number of words in the dataset is {round(sum(len(entry['text'].split()) for entry in reddit) / 1e6, 2)} million.")

def process_document(entry, n_low, n_high):
    text = entry["text"]

    # TODO: Improve upon the text processing pipeline
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # Remove special characters 
    sentences = [re.sub(r"[^a-zA-Z0-9 ]", "", sentence) for sentence in sentences]
    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]
    # Make all sentences lowercase
    sentences = [sentence.lower() for sentence in sentences]

    # Initialize a dictionary for the current document's n-grams
    document_ngram_dict = defaultdict(int)

    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = sentence.split()

        # Generate n-grams for the given range (n_low to n_high)
        for n in range(n_low, n_high + 1):
            ngrams = list(itertools.zip_longest(*[words[i:] for i in range(n)]))
            ngrams = [" ".join(ngram).strip() for ngram in ngrams if None not in ngram]

            for ngram in ngrams:
                document_ngram_dict[ngram] += 1

    return document_ngram_dict

"""Converts the dataset into n-grams and saves the results to a file."""
def n_grams(dataset, n_low=2, n_high=5, file_path="../data/corpus/sentence_ngrams.json"):
    # Initialize a dictionary to store combined n-grams for all documents
    combined_ngram_dict = defaultdict(int)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, entry, n_low, n_high) for entry in dataset]

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            document_ngram_dict = future.result()

            # Combine document-level n-grams into the global combined dictionary
            for ngram, count in document_ngram_dict.items():
                combined_ngram_dict[ngram] += count

    # Save the processed data to a file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_ngram_dict, f, indent=4)

# Convert the dataset into n-grams and save the results to a file
n_grams(reddit, n_low=2, n_high=5, file_path="../data/corpus/reddit_sentence_ngrams_10.json")

# clear memory
gc.collect()
# clear large variables
del reddit

print("Data collection completed!")
