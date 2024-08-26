from typing import Dict, List, Union, Tuple

import camelot
import pdfplumber
from langchain.schema.document import Document

BBox = Tuple[float, float, float, float]
TableInfo = Dict[str, Union[int, str, BBox]]


def get_tables(pdf_path: str) -> List[TableInfo]:
    """
    Extract tables from the PDF using Camelot and return a list of table information.
    """
    samples = camelot.read_pdf(pdf_path, pages="all")
    tables = []
    for sample in samples:
        info = {}
        info['page'] = sample.parsing_report['page']
        info['order'] = sample.parsing_report['order']
        info['table'] = sample.df.to_markdown(index=False)
        info['bbox'] = sample._bbox
        tables.append(info)

    return reorder_table(tables)


def extract_contents(
    page: pdfplumber.page.Page,
    page_num: int,
    tables: List[TableInfo],
    bbox: BBox
) -> str:
    """
    Extract contents from a page, considering tables within the page.
    """
    text = ""
    x0, _, w, h = bbox
    y_start = 0

    current_page_tables = [
        table for table in tables
        if table['page'] == page_num + 1 and x0 < table['bbox'][0] < w
    ]

    for table in current_page_tables:
        x0_table, y0, _, y1 = table['bbox']
        bbox = (x0, y_start, w, h - y1)
        y_start = h - y0
        if bbox[1] > bbox[-1]:
            bbox = (x0, bbox[-1], w, bbox[1])
        text += f"\n{page.within_bbox(bbox).extract_text()}"            
        text += f"\n{table['table']}"

    if not current_page_tables or y_start > 0:
        text += f"\n{page.within_bbox((x0, y_start, w, h)).extract_text()}"

    return text


def reorder_table(tables) -> List[TableInfo]:
    """
    Reorder tables based on their position on the page.
    """
    tables.sort(key=lambda x: (x['page'], x['bbox'][0], -x['bbox'][1]))

    current_page = None
    current_order = 1
    
    for table in tables:
        if table['page'] != current_page:
            current_page = table['page']
            current_order = 1  
        table['order'] = current_order
        current_order += 1

    return tables


def is_two_up_layout(width: float, height: float) -> bool:
    """
    Determine if the page layout is a two-up layout.
    """
    return width > height


def parsing(pdf_path: str) -> List[Document]:
    """
    Parse the entire PDF, extracting text and tables, and returning a list of documents.
    """
    documents = []
    pdf = pdfplumber.open(pdf_path)
    tables = get_tables(pdf_path)
    for page_num, page in enumerate(pdf.pages):
        w, h = page.width, page.height
        text = ""
        if is_two_up_layout(w, h):
            bbox_left = (0, 0, w / 2, h)
            bbox_right = (w / 2, 0, w, h)
            text += extract_contents(page, page_num, tables, bbox_left)
            text += extract_contents(page, page_num, tables, bbox_right)
        else:
            bbox = (0, 0, w, h)
            text += extract_contents(page, page_num, tables, bbox)

        documents.append(Document(metadata={'source': pdf_path, 'page': page_num}, page_content=text))

    return documents