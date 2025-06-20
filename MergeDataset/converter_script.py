import json
import os

def clean_gpt_response(response_text):
    """
    Cleans the GPT response by extracting the core answer and removing
    meta-text like 'Thought Process', 'Answer:', and 'Sources Used'.
    """
    # If the response contains the "Answer:" block, we prioritize extracting it.
    if "\n\nAnswer:\n" in response_text:
        # Get the text after "Answer:"
        main_content = response_text.split("\n\nAnswer:\n", 1)[1]
    else:
        # If there is no "Answer:" block, use the original text.
        # This handles cases where the response is direct.
        main_content = response_text

    # Remove the "Sources Used:" section if it exists in the result
    if "\n\nSources Used:" in main_content:
        main_content = main_content.split("\n\nSources Used:", 1)[0]
        
    # A final cleanup to remove any leading/trailing whitespace
    return main_content.strip()

def convert_conversational_to_instruction(input_path, output_path):
    """
    Reads a JSONL file with conversational data and converts it into a 
    standard instruction-response JSON file.
    """
    # This list will hold all our formatted instruction-response pairs
    final_dataset = []

    print(f"Starting conversion of '{input_path}'...")

    # Open the input file and process it line by line
    with open(input_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            try:
                # Each line is a JSON object
                data = json.loads(line)
                conversations = data.get('conversations', [])

                # Iterate through the conversation in pairs of (human, gpt)
                for j in range(0, len(conversations), 2):
                    # Ensure we don't go out of bounds and that we have a valid pair
                    if (j + 1 < len(conversations) and 
                        conversations[j]['from'] == 'human' and 
                        conversations[j+1]['from'] == 'gpt'):
                        
                        # The human's turn is the instruction
                        instruction = conversations[j]['value'].strip()
                        
                        # The gpt's turn is the raw response
                        raw_response = conversations[j+1]['value']
                        
                        # Clean the raw response to get the core answer
                        cleaned_response = clean_gpt_response(raw_response)

                        # Add the clean pair to our final dataset if both parts have content
                        if instruction and cleaned_response:
                            final_dataset.append({
                                "Instruction": instruction,
                                "Response": cleaned_response
                            })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line {i+1} in {input_path}. Error: {e}")
                continue

    # Now, save the final list to the output JSON file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Use indent=4 for a pretty, human-readable file
        json.dump(final_dataset, outfile, indent=4)

    print("-" * 50)
    print(f"Conversion successful!")
    print(f"Processed {len(final_dataset)} instruction-response pairs.")
    print(f"Formatted data saved to '{output_path}'.")
    print("-" * 50)

# --- Main execution block ---
if __name__ == '__main__':
    # --- CONFIGURATION ---
    input_file = 'conversational_data.jsonl'  # The name of your source file
    output_file = 'formatted_finetuning_data.json' # The name of the file to be created

    # Check if the input file exists before running
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        print("Please make sure your dataset is in the same directory and named correctly.")
    else:
        convert_conversational_to_instruction(input_file, output_file)