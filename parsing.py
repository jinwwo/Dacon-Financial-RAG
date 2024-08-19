from typing import Dict, List

import camelot
import pdfplumber
from langchain.schema.document import Document


def get_tables(pdf_path: str) -> List[Dict]:
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


def text_extractor(
    page,
    page_num: int,
    tables,
    bbox: tuple
) -> str:
    text = ""
    y_start = 0
    has_table = False
    initial_table = True
    x0, _, w, h = bbox
    for table in tables:
        if table['page'] != page_num + 1:
            continue
        x0_table, y0, _, y1 = table['bbox']
        if x0 < x0_table < w:
            has_table = True
            if initial_table:
                initial_table = False
                bbox = (x0, 0, w, h - y1)
                y_start = h - y0
            else:
                bbox = (x0, y_start, w, h - y1)
                y_start = h - y0
            if bbox[1] > bbox[-1]:
                bbox = (x0, bbox[-1], w, bbox[1])
            text += f"\n{page.within_bbox(bbox).extract_text()}"            
            text += f"\n{table['table']}"
            
    if not has_table or y_start > 0:
        bbox = (x0, y_start, w, h)
        text += f"\n{page.within_bbox(bbox).extract_text()}"
    return text


def reorder_table(tables) -> List[Dict]:
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


def is_two_up_layout(w: int, h: int) -> bool:
    return w > h


def parsing(pdf_path: str) -> List[Document]:
    documents = []
    pdf = pdfplumber.open(pdf_path)
    tables = get_tables(pdf_path)
    for page_num in range(len(pdf.pages)):
        text = ""
        page = pdf.pages[page_num]
        w, h = page.width, page.height
        if is_two_up_layout(w, h):
            bbox_left = (0, 0, w / 2, h)
            bbox_right = (w / 2, 0, w, h)
            text += text_extractor(page, page_num, tables, bbox_left)
            text += text_extractor(page, page_num, tables, bbox_right)
        else:
            bbox = (0, 0, w, h)
            text += text_extractor(page, page_num, tables, bbox)
        documents.append(Document(metadata={'source': pdf_path, 'page': page_num}, page_content=text))
    return documents