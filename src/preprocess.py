import os
import re
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"Found {total_pages} pages. Extracting...")
            for page_num in range(total_pages):
                page = reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except FileNotFoundError:
        print(f"Error: Could not find the file at {pdf_path}")
    return text

def clean_text(text):
    # Remove multiple spaces and excessive newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # Remove leading and trailing whitespaces
    text = text.strip()
    return text

def main():
    # Use absolute paths based on the script's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, 'data', '7323175-Doh-Communicable-Diseases.pdf')
    output_path = os.path.join(base_dir, 'data', 'cleaned_doh_manual.txt')

    print("Extracting text from DOH manual PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text:
        print("Extraction failed. Please check the PDF file.")
        return

    print("Cleaning extracted text...")
    cleaned = clean_text(raw_text)

    print(f"Saving cleaned text to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()