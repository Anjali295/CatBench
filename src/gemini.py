
import google.generativeai as genai
import jsonlines
import hashlib
import time
import pickle
import atexit
from config import config
import os

# Set up the Gemini model

genai.configure(api_key=config["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")

#response = model.generate_content("Write a story about a magic backpack.")
#print(response.text)

# Load cache if exists, otherwise create an empty cache
def load_cache_if_exists():
    cache_path = 'cache.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            return pickle.load(handle)
    return {}

# Save cache before exiting
def save_cache_on_exit(cache):
    print('Saving cache...')
    with open('cache.pkl', 'wb') as cache_file:
        pickle.dump(cache, cache_file)

# Read the JSONL data
def read_jsonl_file(filepath):
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data

# Define the path to your files
files = [
    'data/train_must_why/dependent_real_after.jsonl',
    'data/train_must_why/dependent_real_before.jsonl',
    'data/train_must_why/nondependent_real_after.jsonl',
    'data/train_must_why/nondependent_real_before.jsonl'
]

# Extract relevant data for binary classification
def extract_binary_data(data):
    binary_data = []
    for entry in data:
        question = entry['binary_question']
        steps = entry['steps']
        step_pair = entry['step_pair_idx_asked_about']
        # Create a prompt based on the step pair and question ------
        prompt = f"Step {step_pair[0]+1}: {steps[step_pair[0]]}\nStep {step_pair[1]+1}: {steps[step_pair[1]]}\nQuestion: {question}"
        binary_data.append({'prompt': prompt, 'label': question})
    return binary_data

# Call the Gemini model with exception handling and caching
def call_gemini(prompt, cache_key, cached_responses, idx):
    if cache_key in cached_responses:
        print(f"Cache hit for record {idx}")
        return cached_responses[cache_key]['text']
    else:
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f'Exception: {e} for record {idx}. Retrying...')
            time.sleep(60)  # Wait before retrying
            return call_gemini(prompt, cache_key, cached_responses, idx)


# Process results to get yes/no answers
def  process_responses(response):
    if "yes" in response.lower():
        return 1  # Yes
    elif "no" in response.lower():
        return 0  # No
    else:
        return -1  # Unclear or unknown response


def main():

    # Load cached responses if they exist
    cached_responses = load_cache_if_exists()

    # Register saving the cache before exiting
    atexit.register(save_cache_on_exit, cached_responses)

    # Read all the data from your JSONL files
    data_dependent_real_after = read_jsonl_file(files[0])
    data_dependent_real_before = read_jsonl_file(files[1])
    data_nondependent_real_after = read_jsonl_file(files[2])
    data_nondependent_real_before = read_jsonl_file(files[3])

    # Extract binary data
    binary_data_after = extract_binary_data(data_dependent_real_after[:10])  #for now trying 10 entries.

    responses = []
    binary_results = []


    for idx, entry in enumerate(binary_data_after):
        prompt = entry['prompt']

        # Create a hash for the current prompt to check the cache
        hashed_prompt = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        cache_key = (hashed_prompt, model.model_name)

        # Call the Gemini model (with caching and retries)
        model_answer = call_gemini(prompt, cache_key, cached_responses, idx)

        # Cache the response if it's a new one
        cached_responses[cache_key] = {'text': model_answer}

        # Process the response to extract yes/no label
        binary_result = process_responses(model_answer)

        # Store the response and result
        responses.append({
            'prompt': prompt,
            'response': model_answer
        })
        binary_results.append(binary_result)

        # Print the result for the current entry
        print(f"Prompt: {prompt}\nResponse: {model_answer}\nBinary Result: {binary_result}\n")


if __name__ == "__main__":
    main()
