import io
import pdfplumber

def extract_text_from_pdf(content: bytes) -> str:
    try:
        pdf_stream = io.BytesIO(content)
        with pdfplumber.open(pdf_stream) as pdf:
            text = "\n\n".join(
                page.extract_text() or "" 
                for page in pdf.pages
            )

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("The PDF file is empty or text cannot be extracted (it may be a scan).")
        return clean_text

    except ValueError:
        raise    
    except Exception as e:
        raise ValueError("The PDF file is corrupted or in an incorrect format.")

    
    

