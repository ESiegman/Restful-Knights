##
# @file llama_client.py
# @brief Client for interacting with the Ollama LLM to analyze sleep data.

import csv
from ollama import chat
import markdown
from bs4 import BeautifulSoup

##
# @brief Gets a response from the LLM analyzing EEG and SpO₂ data.
# @param eeg_csv Path to EEG results CSV file.
# @param spo_csv Path to SpO₂ results CSV file.
# @return Plain text analysis and summary from the LLM.
def get_response(eeg_csv="EEG_results.csv", spo_csv="SPO_results.csv"):
    data_lines = []
    spo_lines = []

    # Read EEG CSV
    try:
        with open(eeg_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_lines.append(','.join(row))
        eeg_data = '\n'.join(data_lines)
    except FileNotFoundError:
        eeg_data = "Error: EEG_results.csv not found."
    except Exception as e:
        eeg_data = f"Error reading EEG CSV: {str(e)}"

    # Read SpO2 CSV
    try:
        with open(spo_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                spo_lines.append(','.join(row))
        spo_data = '\n'.join(spo_lines)
    except FileNotFoundError:
        spo_data = "Error: SPO_results.csv not found."
    except Exception as e:
        spo_data = f"Error reading SPO CSV: {str(e)}"

    prompt = f"""
You are an expert somnologist analyzing sleep data.
EEG Data:
{eeg_data}

SpO₂ Data:
{spo_data}

Analyze both datasets together to identify sleep patterns, abnormalities, and provide recommendations. Output results in CSV format as before, then a summary.
"""

    try:
        response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        markdown_txt =  response['message']["content"]
        reg_txt = markdown.markdown(markdown_txt)

        soup = BeautifulSoup(reg_txt, 'html.parser')
        plain_txt = soup.get_text(separator='\n')

        return plain_txt
    except Exception as e:
        return f"Error getting response: {str(e)}"
