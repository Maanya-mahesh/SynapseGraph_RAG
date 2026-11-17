#Description: Extracts 2D bounding box coordinates from PDFs & builds hierarchical section trees
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF

@dataclass
class TextSpan:# -> extracted text span with geometric bounding box metadata
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    is_bold: bool
    @property
    def x0(self) -> float:
        return self.bbox[0]
    @property
    def y0(self) -> float:
        return self.bbox[1]
    @property
    def x1(self) -> float:
        return self.bbox[2]
    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class SectionNode:# -> hierarchical node = section header and its nested contents
    title: str
    level: int  # 0: Root, 1: H1, 2: H2
    content: List[str] = field(default_factory=list)
    subsections: List['SectionNode'] = field(default_factory=list)

class LayoutAwarePDFParser: # parses PDF documents using 2D bounding box coordinates & font size heuristics
    def __init__(self, pdf_path: str, body_font_size: float = 10.0):
        self.pdf_path = pdf_path
        self.body_font_size = body_font_size

    def _extract_page_spans(self, page: fitz.Page) -> List[TextSpan]: 
        spans = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    # now check font flags for bold attribute (bit 4)
                    is_bold = bool(span.get("flags", 0) & (1 << 4))
                    spans.append(
                        TextSpan(
                            text=text,
                            bbox=span["bbox"],
                            font_size=round(span["size"], 1),
                            is_bold=is_bold
                        )
                    )
        return spans

    def parse_structure(self) -> SectionNode:# now parse the PDF & build a hierarchical SectionNode document tree
        doc = fitz.open(self.pdf_path)
        root = SectionNode(title="Document Root", level=0)
        stack = [root]
        last_y1 = 0.0
        for page in doc:
            spans = self._extract_page_spans(page) #this sorts spans by column x0 then top vertical coordinate y0
            spans.sort(key=lambda s: (0 if s.x0 < 280 else 1, s.y0))
            for span in spans:
                delta_y = span.y0 - last_y1 if last_y1 > 0 else 0.0
                last_y1 = span.y1
                gamma = span.font_size / self.body_font_size
                if gamma >= 1.5 or (span.is_bold and gamma >= 1.3):
                    level = 1
                elif 1.2 <= gamma < 1.5 or (span.is_bold and gamma > 1.0):
                    level = 2
                else:
                    level = 0
                if level > 0:
                    new_node = SectionNode(title=span.text, level=level)
                    # now keep poping stack until we find the parent level
                    while stack and stack[-1].level >= level:
                        stack.pop()
                    stack[-1].subsections.append(new_node)
                    stack.append(new_node)
                else:# add paragraph content 
                    annotation = f"{span.text} [y0={span.y0:.1f}, Δy={delta_y:.1f}pt]"
                    stack[-1].content.append(annotation)
        doc.close()
        return root
