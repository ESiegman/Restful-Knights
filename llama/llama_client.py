import csv
from ollama import chat
import markdown
from bs4 import BeautifulSoup


def get_response():
    csv_file = "EEG_results.csv"
    data_lines = []

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_lines.append(','.join(row))
        data = '\n'.join(data_lines)
    except FileNotFoundError:
        return "Error: EEG_results.csv not found."
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

    prompt = f"""
You are an expert somnologist analyzing EEG data to identify sleep patterns, sleep stages, and any abnormalities.
The CSV data has columns:
- time (float, in seconds)
- label (int, current stage: 0=Awake, 1=N1, 2=N2, 3=N3, 4=REM)

Your task:
1. Analyze the EEG data to detect overall sleep structure and key patterns.
2. Identify any irregularities or abnormalities in stage distribution or transitions (e.g., fragmented REM, excessive awakenings, missing deep sleep).
3. If the results are abnormally irregular or potentially concerning, include a gentle note suggesting the user should consider consulting a sleep specialist or doctor.
4. Offer practical suggestions for improving sleep quality based on the detected issues (e.g., bedtime routine, screen time, caffeine intake, etc.).
5. Present the analysis results in **true CSV format** (no Markdown or tables), columns:
   - x: description or pattern (e.g., "Total REM Duration", "Number of Awakenings", "Abnormal Sleep Fragmentation Detected")
   - y: value or interpretation (e.g., 90, "Moderate", "Yes")

After the CSV, include a short summary paragraph describing the user's sleep quality and any recommendations.

Data:

{data}

Provide a concise summary followed by the CSV results.
"""

    try:
        response = chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
        markdown_txt =  response['message']["content"]
        reg_txt = markdown.markdown(markdown_txt)

        soup = BeautifulSoup(reg_txt, 'html.parser')
        plain_txt = soup.get_text(separator='\n')

        return plain_txt
    except Exception as e:
        return f"Error getting response: {str(e)}"
