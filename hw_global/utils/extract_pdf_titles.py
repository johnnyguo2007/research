import os
import re
import requests
import logging
from pdfminer.high_level import extract_text
from doi2bib.crossref import get_bib
import openpyxl
from datetime import datetime

# Configure logging
log_filename = f"pdf_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_doi_from_pdf(pdf_path):
    """
    Extracts the DOI from a PDF file using regex.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The DOI string if found, otherwise None.
    """
    logger.info(f"Extracting DOI from: {pdf_path}")
    try:
        text = extract_text(pdf_path)
        # Regex to find DOI (improved to handle more variations)
        doi_regex = r"(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)"
        match = re.search(doi_regex, text)
        if match:
            doi = match.group(0)
            logger.info(f"Found DOI: {doi}")
            return doi
        else:
            logger.warning(f"No DOI found in {pdf_path}")
            return None
    except Exception as e:
        logger.error(f"Error extracting DOI from {pdf_path}: {str(e)}")
        return None

def format_bibtex(response_text, author=None, title=None, journal=None, year=None):
    """
    Formats BibTeX entry in a standardized format.
    """
    if not response_text:
        logger.warning("No response text provided for BibTeX formatting")
        return None
        
    # Extract the entry type and make it lowercase
    entry_match = re.match(r'@(\w+)\s*{\s*([^,]*),', response_text)
    if not entry_match:
        logger.warning("Could not extract entry type and citation key from BibTeX")
        return None
    entry_type = entry_match.group(1).lower()
    logger.debug(f"Found entry type: {entry_type}")
    
    # Extract fields from the response text
    fields = {}
    field_patterns = {
        'author': r'author\s*=\s*[{"](.+?)[}"],?',
        'title': r'title\s*=\s*[{"](.+?)[}"],?',
        'journal': r'(?:journal|journaltitle)\s*=\s*[{"](.+?)[}"],?',
        'year': r'year\s*=\s*[{"]?(\d{4})[}"]?,?',
        'volume': r'volume\s*=\s*[{"](.+?)[}"],?',
        'number': r'number\s*=\s*[{"](.+?)[}"],?',
        'pages': r'pages\s*=\s*[{"](.+?)[}"],?',
        'publisher': r'publisher\s*=\s*[{"](.+?)[}"],?'
    }
    
    for field, pattern in field_patterns.items():
        match = re.search(pattern, response_text)
        if match:
            fields[field] = match.group(1)
            logger.debug(f"Found {field}: {fields[field]}")
    
    # Use backup values if not found in response_text
    if 'author' not in fields and author:
        fields['author'] = author
        logger.debug(f"Using backup author: {author}")
    if 'title' not in fields and title:
        fields['title'] = title
        logger.debug(f"Using backup title: {title}")
    if 'journal' not in fields and journal:
        fields['journal'] = journal
        logger.debug(f"Using backup journal: {journal}")
    if 'year' not in fields and year:
        fields['year'] = year
        logger.debug(f"Using backup year: {year}")
    
    # Generate citation key in the format: firstauthor+year+firstword
    try:
        if 'author' in fields and 'year' in fields and 'title' in fields:
            # Get first author's lastname
            first_author = fields['author'].split(' and ')[0].split(',')[0].split()[-1].lower()
            # Get first meaningful word from title (skip articles like "a", "an", "the")
            title_words = fields['title'].lower().split()
            skip_words = {'a', 'an', 'the', 'on', 'in', 'at', 'to', 'for', 'of', 'and'}
            first_word = next((word for word in title_words if word not in skip_words), title_words[0])
            citation_key = f"{first_author}{fields['year']}{first_word}"
            # Remove any non-alphanumeric characters
            citation_key = re.sub(r'[^\w]', '', citation_key)
        else:
            citation_key = "unknown"
    except Exception as e:
        logger.warning(f"Error generating citation key: {str(e)}")
        citation_key = "unknown"
    
    # Format the BibTeX entry with two-space indentation
    formatted = f"@{entry_type}{{{citation_key},\n"
    # Define field order
    field_order = ['title', 'author', 'journal', 'volume', 'number', 'pages', 'year', 'publisher']
    for field in field_order:
        if field in fields and fields[field]:
            formatted += f"  {field}={{{fields[field]}}},\n"
    formatted = formatted.rstrip(',\n') + "\n}"
    
    logger.debug("Successfully formatted BibTeX entry")
    return formatted

def get_paper_info_from_doi(doi):
    """
    Fetches paper title, author, year, and journal from a DOI using doi2bib.
    """
    try:
        logger.info(f"Fetching information for DOI: {doi}")
        bib_entry = get_bib(doi)
        if bib_entry:
            if isinstance(bib_entry, dict):
                logger.info("Received dictionary response from doi2bib")
                info = {
                    "title": bib_entry.get('title', None),
                    "author": bib_entry.get('author', None),
                    "year": bib_entry.get('year', None),
                    "journal": bib_entry.get('journal', None),
                }
                logger.debug(f"Extracted info from dictionary: {info}")
                
                # Try to create a formatted BibTeX even for dictionary response
                citation_key = re.sub(r'[^\w]', '', doi.split('/')[-1])[:10]
                dummy_bibtex = f"@ARTICLE{{{citation_key},\n"
                for field, value in info.items():
                    if value:
                        dummy_bibtex += f"    {field} = {{{value}}},\n"
                dummy_bibtex += "}"
                info["bibtex"] = dummy_bibtex
                logger.debug("Created dummy BibTeX entry")
                return info
                
            elif isinstance(bib_entry, tuple):
                logger.info("Received tuple response from doi2bib")
                status_code, response_text = bib_entry
                logger.debug(f"Status code: {status_code}")
                logger.info(f"Response text: {response_text}")
                
                if status_code == True and response_text:
                    # Parse the BibTeX response text
                    title_match = re.search(r'title\s*=\s*[{"](.+?)[}"],?', response_text)
                    author_match = re.search(r'author\s*=\s*[{"](.+?)[}"],?', response_text)
                    year_match = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?,?', response_text)
                    journal_match = re.search(r'journal\s*=\s*[{"](.+?)[}"],?', response_text)
                    
                    logger.info("Matched fields:")
                    logger.info(f"Title match: {title_match.group(1) if title_match else 'Not found'}")
                    logger.info(f"Author match: {author_match.group(1) if author_match else 'Not found'}")
                    logger.info(f"Year match: {year_match.group(1) if year_match else 'Not found'}")
                    logger.info(f"Journal match: {journal_match.group(1) if journal_match else 'Not found'}")
                    
                    # Try alternative patterns if initial ones fail
                    if not journal_match:
                        journal_match = re.search(r'journaltitle\s*=\s*[{"](.+?)[}"],?', response_text)
                        logger.debug(f"Alternative journal match: {journal_match.group(1) if journal_match else 'Not found'}")
                    
                    title = title_match.group(1) if title_match else None
                    author = author_match.group(1) if author_match else None
                    year = year_match.group(1) if year_match else None
                    journal = journal_match.group(1) if journal_match else None
                    
                    formatted_bibtex = format_bibtex(
                        response_text,
                        author=author,
                        title=title,
                        journal=journal,
                        year=year
                    )
                    
                    if not formatted_bibtex:
                        logger.warning("Failed to format BibTeX entry")
                    
                    return {
                        "title": title,
                        "author": author,
                        "year": year,
                        "journal": journal,
                        "bibtex": formatted_bibtex
                    }
                logger.warning(f"Invalid response: status_code={status_code}")
                return None
        logger.warning("No response from doi2bib")
        return None
    except Exception as e:
        logger.error(f"Error fetching info for DOI {doi}: {str(e)}")
        if 'response_text' in locals():
            logger.error(f"Full response text when error occurred: {response_text}")
        return None

def main():
    """
    Processes all PDF files in the current directory, extracts DOIs,
    fetches paper information, and writes the data to an Excel file.
    """
    logger.info("Starting PDF processing")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Filename", "DOI", "Title", "Author", "Year", "Journal", "BibTeX"])

    pdf_count = 0
    success_count = 0
    error_count = 0

    for filename in os.listdir("."):
        if filename.endswith(".pdf"):
            pdf_count += 1
            pdf_path = os.path.join(".", filename)
            logger.info(f"Processing PDF {pdf_count}: {filename}")
            doi = extract_doi_from_pdf(pdf_path)

            if doi:
                doi_link = f"https://doi.org/{doi}"
                paper_info = get_paper_info_from_doi(doi)
                if paper_info:
                    success_count += 1
                    sheet.append([
                        filename,
                        doi_link,
                        paper_info["title"],
                        paper_info["author"],
                        paper_info["year"],
                        paper_info["journal"],
                        paper_info["bibtex"]
                    ])
                    logger.info(f"Successfully processed {filename}")
                else:
                    error_count += 1
                    sheet.append([filename, doi_link, "Not Found", "Not Found", "Not Found", "Not Found", "Not Found"])
                    logger.warning(f"Failed to get paper info for {filename}")
            else:
                error_count += 1
                logger.warning(f"No DOI found in {filename}")
                sheet.append([filename, "Not Found", "", "", "", "", ""])

    # Adjust column width for better readability
    for column in sheet.columns:
        max_length = 0
        column_letter = openpyxl.utils.get_column_letter(column[0].column)
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 100)  # Cap width at 100 characters
        sheet.column_dimensions[column_letter].width = adjusted_width

    workbook.save("paper_information.xlsx")
    logger.info(f"Excel file 'paper_information.xlsx' created successfully")
    logger.info(f"Processing complete. Total PDFs: {pdf_count}, Successful: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()