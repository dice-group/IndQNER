import pandas as pd
from datasets import Dataset
import json

# Function to prepare each example
def prepare_examples(sentence):
    system_message = """
    Given the following entity classes and sentences, label entity mentions with their respective classes in sentences according to the sentences' context. 
    In the output, only include entity mentions and their respective class in the given output format. No needed further explanation.
    CONTEXT: entity classes: {text["named_entity_classes"]}. 
    Example sentence: Jika kamu (tetap) dalam keraguan tentang apa (Al-Qur’an) yang Kami turunkan kepada hamba Kami (Nabi Muhammad), buatlah satu surah yang semisal dengannya dan ajaklah penolong-penolongmu selain Allah, jika kamu orang-orang yang benar.
    Example output: Jika/O kamu/O (/O tetap/O )/O dalam/O keraguan/O tentang/O apa/O (/O Al-Qur’an/HolyBook )/O yang/O Kami/O turunkan/O kepada/O hamba/O Kami/O (/O Nabi/O Muhammad/Messenger )/O ,/O buatlah/O satu/O surah/O yang/O semisal/O dengannya/O dan/O ajaklah/O penolong-penolongmu/O selain/O Allah/Allah ,/O jika/O kamu/O orang-orang/O yang/O benar/O ./O
    """
    
    test_sentence =f"""
    Test sentence: {sentence}
    Test output:
    """
    return {
        "content" :{
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": test_sentence}
        ]}
    }

domain = "specific-domain"

# Read CSV files
test_df = pd.read_csv(f"../al-quran/114-An_Nas.txt", sep="\t", header=None, names=["sentence"])
test_df.head()

# Convert DataFrame to Dataset
test_dataset = Dataset.from_pandas(test_df)

# Apply the prepare_examples function
test_dataset = test_dataset.map(prepare_examples, remove_columns=['sentence'])

# Extract only the messages field
messages_list = [example for example in test_dataset]

# Save the messages to a JSONL file
json_output = f"../new-datasets/114-An_Nas.txt.jsonl"
with open(json_output, "w") as json_file:
    for message in messages_list:
        json.dump(message, json_file)
        json_file.write("\n")

print(f"Messages saved to {json_output}")
