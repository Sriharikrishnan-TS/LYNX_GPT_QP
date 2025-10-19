from pdf2image import convert_from_path
import numpy as np
from paddleocr import PPStructureV3

pdf = "lynx_gpt_qp-2.pdf"
def paddleocr(pdf_path):
    ocr = PPStructureV3(lang="en", use_textline_orientation= True, 
                        use_doc_orientation_classify=False, use_doc_unwarping=False)
    imgs = convert_from_path(pdf, 500)
    all_txt = []
    for img in imgs:
        rgb = np.array(img.convert('RGB'))
        op = ocr.predict(rgb)
        page_txt = []
        for line in op[0]:
            txt_line = line[1][0]
            if txt_line.strip():
                page_txt.append(txt_line)
        
        all_txt.append( " ".join(page_txt))

    return "\n\n".join(all_txt)

txt = paddleocr(pdf)
print(txt[:1500])

