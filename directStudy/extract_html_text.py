import os
import re
from bs4 import BeautifulSoup

def extract_text_from_html(html_file, output_file):
    """
    Extracts visible text from an HTML file and saves it to a text file.

    Args:
        html_file: Path to the HTML file.
        output_file: Path to the output text file.
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get visible text
        text = soup.get_text(separator=' ', strip=True)

        # Remove extra whitespace and newlines (optional)
        text = re.sub(r'\s+', ' ', text).strip()

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(text)

        print(f"Text extracted from '{html_file}' and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: File not found: {html_file}")
    except Exception as e:
        print(f"An error occurred while processing {html_file}: {e}")

def process_html_files_in_directory(directory):
    """
    Processes all HTML files in a directory and extracts visible text.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            html_file = os.path.join(directory, filename)
            output_file = os.path.splitext(html_file)[0] + ".txt"
            extract_text_from_html(html_file, output_file)

if __name__ == "__main__":
    current_directory = os.getcwd()  # Get the current working directory
    process_html_files_in_directory(current_directory)