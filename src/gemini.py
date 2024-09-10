import google.generativeai as genai
import jsonlines
from config import config
#import os

# Set up the Gemini model

genai.configure(api_key=config["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")

#response = model.generate_content("Write a story about a magic backpack.")
#print(response.text)


# Step 1: Read the JSONL data
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

# Step 2: Extract relevant data for binary classification
def extract_binary_data(data):
    binary_data = []
    for entry in data:
        question = entry['binary_question']
        steps = entry['steps']
        step_pair = entry['step_pair_idx_asked_about']
        # Create a prompt based on the step pair and question
        prompt = f"Step {step_pair[0]+1}: {steps[step_pair[0]]}\nStep {step_pair[1]+1}: {steps[step_pair[1]]}\nQuestion: {question}"
        binary_data.append({'prompt': prompt, 'label': question})
    return binary_data

# Step 3: Send prompts to the Gemini model
def test_gemini_model(binary_data):
    results = []
    for entry in binary_data:
        prompt = entry['prompt']
        response = model.generate_content(prompt)
        results.append({
            'prompt': prompt,
            'response': response.text
        })
    return results

# Step 4: Process results to get yes/no answers
def process_responses(responses):
    processed = []
    for response in responses:
        if "yes" in response['response'].lower():
            processed.append(1)  # Yes
        elif "no" in response['response'].lower():
            processed.append(0)  # No
        else:
            processed.append(-1)  # Unclear or unknown response
    return processed

# Step 5: Main function to execute the workflow
def main():
    # Read all the data from your JSONL files
    data_dependent_real_after = read_jsonl_file(files[0])
    data_dependent_real_before = read_jsonl_file(files[1])
    data_nondependent_real_after = read_jsonl_file(files[2])
    data_nondependent_real_before = read_jsonl_file(files[3])

    # Extract binary data
    binary_data_after = extract_binary_data(data_dependent_real_after[:10])  #for now trying 10 entries.

    # Initialize an empty list to store responses and results
    responses = []
    binary_results = []


    for entry in binary_data_after:
        # Test each entry individually using the Gemini model
        response = test_gemini_model([entry])
        
        # Process the response to extract yes/no label
        binary_result = process_responses(response)
        
        # Store the response and result
        responses.append(response[0]) 
        binary_results.append(binary_result[0])

        # Print the result for the current entry
        print(f"Prompt: {response[0]['prompt']}\nResponse: {response[0]['response']}\nBinary Result: {binary_result[0]}\n")


    # # Test the binary data using the Gemini model
    # responses = test_gemini_model(binary_data_after)

    # # Process responses to extract yes/no labels
    # binary_results = process_responses(responses)
    
    # # Print the results
    # for result, response in zip(binary_results, responses):
    #     print(f"Prompt: {response['prompt']}\nResponse: {response['response']}\nBinary Result: {result}\n")

if __name__ == "__main__":
    main()
