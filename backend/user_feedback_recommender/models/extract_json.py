import os
import json
import pandas as pd
import regex

# Define input folder and output file
input_folder = "outputs"
output_file = "restaurant_pages.json"
unprocessed_files_log = "unprocessed_files.txt"

data_list = []
unprocessed_files = []

def extract_json(text):
    """Extracts a JSON object from the given text using regex."""
    json_pattern = regex.compile(r'\{(?:[^{}]+|(?R))*\}')  # Match nested JSON
    match = json_pattern.search(text)  # Search for JSON in text
    if match:
        json_str = match.group(0)  # Extract the JSON substring
        try:
            return json.loads(json_str)  # Convert string to JSON
        except json.JSONDecodeError:
            print("⚠️ Warning: Failed to parse JSON response.")
            return text
    else:
        print("No JSON object found in the text.")
        return text
    
def extract_filename_info(filename):
    """Extracts i, j, and identifier from the filename using regex."""
    match = regex.search(r'output_text_(\d+)_(\d+)_([a-f0-9\-]+)', filename)
    if match:
        return {
            # "i": int(match.group(1)),  # Convert to integer
            # "j": int(match.group(2)),  # Convert to integer
            "record_id": match.group(3)
        }
    return None

# Iterate through files
for filename in os.listdir(input_folder):
    if filename.startswith("output_text_") and filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing {file_path}...")
        file_info = extract_filename_info(filename)

        if file_info:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    extracted_data = extract_json(content)

                    if extracted_data:
                        # Add extracted filename info (i, j, identifier) to JSON
                        extracted_data.update(file_info)
                        data_list.append(extracted_data)
                    else:
                        unprocessed_files.append(filename)
                        print(f"Skipping {filename}: No valid JSON extracted.")

            except Exception as e:
                unprocessed_files.append(filename)
                print(f"Error processing {filename}: {e}")

# Save unprocessed file names
with open(unprocessed_files_log, "w", encoding="utf-8") as f:
    f.write("\n".join(unprocessed_files))
    
# Convert to Pandas DataFrame
df = pd.DataFrame(data_list)

# Save DataFrame as JSON
df.to_json(output_file, orient="records", indent=4)