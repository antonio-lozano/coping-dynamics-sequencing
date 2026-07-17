import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
doc = Document(r'C:\Users\jenif\Downloads\Coping_strategies_manuscript\Coping_strategies_dynamics_without_methods_JSG_01.04.docx')
paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
print(f'Total non-empty paragraphs: {len(paras)}')
for i, t in enumerate(paras):
    print(f'[{i}] {t[:200]}')
