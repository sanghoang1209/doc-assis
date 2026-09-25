import pytest
from unittest.mock import MagicMock, patch
from app.services.extractor import extract_document_content, extract_text_from_pdf

def test_extract_empty_content_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        extract_document_content(b"", "sample.pdf")

@patch("app.services.extractor.get_converter")
def test_extract_pdf_markdown_success(mock_get_converter):
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Header\n\nSample extracted content from PDF."
    mock_converter.convert.return_value = mock_result
    mock_get_converter.return_value = mock_converter

    dummy_pdf_bytes = b"%PDF-1.4 sample pdf content"
    result = extract_document_content(dummy_pdf_bytes, "test.pdf")

    assert "# Header" in result
    assert "Sample extracted content from PDF." in result
    mock_converter.convert.assert_called_once()

@patch("app.services.extractor.get_converter")
def test_extract_docx_markdown_success(mock_get_converter):
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "| Table Header |\n| --- |\n| Cell Data |"
    mock_converter.convert.return_value = mock_result
    mock_get_converter.return_value = mock_converter

    dummy_docx_bytes = b"PK\x03\x04 dummy docx zip bytes"
    result = extract_document_content(dummy_docx_bytes, "test.docx")

    assert "Table Header" in result
    assert "Cell Data" in result
    mock_converter.convert.assert_called_once()

@patch("app.services.extractor.extract_document_content")
def test_extract_text_from_pdf_legacy_alias(mock_extract_doc):
    mock_extract_doc.return_value = "Sample legacy text"
    res = extract_text_from_pdf(b"dummy pdf bytes")
    assert res == "Sample legacy text"
    mock_extract_doc.assert_called_once_with(b"dummy pdf bytes", "document.pdf")
