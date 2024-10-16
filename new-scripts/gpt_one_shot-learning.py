import pandas as pd
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
import json
import ast
# Initialize OpenAI client
client = OpenAI(api_key="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")

def query_llm(prompt):
    """
    Send a prompt to the LLM and return the response.
    """
    print(prompt)
    response = client.chat.completions.create(
        messages=prompt,
        model="gpt-4o", #change the model to model="gpt-3.5-turbo if you want to use gpt-3.5 
    )
    return {"response": response.choices[0].message.content}

def read_first_few_lines(file_path, num_lines=0):
    lines = []
    try:
        with open(file_path, 'r') as file:
            if num_lines==0:
                lines = file.readlines()
            else:
                for _ in range(num_lines):
                    lines.append(file.readline().strip())
    except Exception as e:
        return str(e)
    return lines


# Read and display the first few lines of the file
domain = "specific-domain"
file_path = f"../al-quran-dataset-formatted/114-An_Nas.txt.jsonl"
first_few_lines = read_first_few_lines(file_path)
predict_dict = dict()
num = 1
for line in tqdm(first_few_lines):
    line = ast.literal_eval(line)
    results = query_llm(line["content"]["messages"])
    predict_dict[num]=results['response']
    num +=1

# Save the dictionary to a JSON file
with open("results-zeroshot-attempt-1-gpt.json", "w") as json_file:
    json.dump(predict_dict, json_file, indent=4)  # 'indent' adds formatting for readability

print("Results saved")