
import google.generativeai as genai
import jsonlines
import hashlib
import time
import pickle
import atexit
from config import config
import os
import argparse

# Set up the Gemini model

genai.configure(api_key=config["GEMINI_API_KEY"])


def parse_args():
    parser = argparse.ArgumentParser(description='Binary Classification Using Gemini Models')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data (JSONL files)')
    parser.add_argument('--cache_path', type=str, default='cache.pkl', help='Path to save cache data')
    parser.add_argument('--log_path', type=str, default='gemini_log.txt', help='Path to log the data')
    parser.add_argument('--max_tokens', type=int, default=2, help='Maximum tokens for the generated response')
    parser.add_argument('--model_name', type=str, default='gemini-1.5-pro', help='Model name')
    args = parser.parse_args()
    return args

# Load cache if exists, otherwise create an empty cache
def load_cache_if_exists(cache_path):  
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            return pickle.load(handle)
    return {}

# Save cache before exiting
def save_cache_on_exit(cache, cache_path):
    print('Saving cache...')
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(cache, cache_file)

# Read the JSONL data
def read_jsonl_file(filepath):
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data


# Extract relevant data for binary classification
def extract_binary_data(data):
    binary_data = []
    for entry in data:
        goal = entry['title']
        question = entry['binary_question']
        steps = entry['steps']
        step_pair = entry['step_pair_idx_asked_about']

        procedure = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])

        # Create a prompt based on the step pair and question ------
        prompt = f"Given a goal, a procedure to achieve that goal and a question about the steps in the procedure, you are required to answer the question in one sentence.\n\nGoal: {goal}\nProcedure:\n{procedure}\n\nMust Step {step_pair[0]+1} happen before Step {step_pair[1]+1}? Select between yes or no\n\n"

        binary_data.append({'prompt': prompt, 'label': question})
    return binary_data

# Call the Gemini model with exception handling and caching
def call_gemini(prompt, cache_key, cached_responses, idx, model):
    # break point for prompt checking 
    if cache_key in cached_responses:
        print(f"Cache hit for record {idx}")
        return cached_responses[cache_key]['text']
    else:
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f'Exception: {e} for record {idx}. Retrying...')
            time.sleep(20)  # Wait before retrying
            return call_gemini(prompt, cache_key, cached_responses, idx, model)


# Process results to get yes/no answers
def  process_responses(response):

    if "yes" in response:
        return 1  # Yes
    elif "no" in response:
        return 0  # No
    else:
        return -1  # Unclear or unknown response


# Function to log data to a file
def log_to_file(prompt, response, binary_result, idx, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Record {idx+1}:\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Response: {response}\n")
        f.write(f"Binary Result: {binary_result}\n")
        f.write("\n" + "="*50 + "\n\n")

def main():

    args = parse_args()

    # Load cached responses if they exist
    cached_responses = load_cache_if_exists(args.cache_path)

    # Register saving the cache before exiting
    atexit.register(save_cache_on_exit, cached_responses, args.cache_path)

    data = read_jsonl_file(args.train_data_path)

    # Extract binary data
    binary_data_after = extract_binary_data(data[5:10])  #for now trying 10 entries.

    responses = []
    binary_results = []
    
    candidatecount, temp, topp, topk = 1, 0.0, 1.0, 1
    generation_config = genai.GenerationConfig(
        candidate_count = candidatecount,
        max_output_tokens = 2,
        temperature = temp,
        top_p = topp,
        top_k = topk
    )

    model = genai.GenerativeModel(model_name=args.model_name, generation_config=generation_config)



    for idx, entry in enumerate(binary_data_after):
        prompt = entry['prompt']
        # add a break point and see what the prompt is , do not run the model and go through the prompts one by one for each of these 10 entries
        # idx, raw input, prompt and the model output
       
        # import pdb; pdb.set_trace() #for breakpoint use
        # print(prompt)

        # Create a hash for the current prompt to check the cache
        hashed_prompt = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        cache_key = (hashed_prompt, model.model_name)

        # Call the Gemini model (with caching and retries)
        # breakpoint()
        model_answer = call_gemini(prompt, cache_key, cached_responses, idx, model)

        cleaned_response = model_answer.strip().replace(",", "").replace(".", "").lower()

        # breakpoint()

        # Cache the response if it's a new one
        cached_responses[cache_key] = {'text': cleaned_response}

        # Process the response to extract yes/no label
        binary_result = process_responses(cleaned_response)

        # Log the prompt, response, and binary result to a file
        log_to_file(prompt, cleaned_response, binary_result, idx, args.log_path)

        # Store the response and result
        responses.append({
            'prompt': prompt,
            'response': cleaned_response
        })
        binary_results.append(binary_result)

        # Print the result for the current entry
        print(f"Prompt: {prompt}\nResponse: {cleaned_response}\nBinary Result: {binary_result}\n")


if __name__ == "__main__":
    main()
