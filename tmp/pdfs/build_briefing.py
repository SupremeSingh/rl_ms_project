from pathlib import Path
import re, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
fontroot=Path('/Users/manmit/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
for name,file in [('D','DejaVuSans.ttf'),('DB','DejaVuSans-Bold.ttf'),('DI','DejaVuSans-Oblique.ttf'),('DBI','DejaVuSans-BoldOblique.ttf')]: pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('D',normal='D',bold='DB',italic='DI',boldItalic='DBI')
navy=colors.HexColor('#17354b')
body=ParagraphStyle('body',fontName='D',fontSize=9,leading=13,spaceAfter=7,allowWidows=0,allowOrphans=0,textColor=colors.HexColor('#24333c'))
styles={'body':body,'title':ParagraphStyle('title',parent=body,fontName='DB',fontSize=23,leading=29,spaceAfter=12,textColor=navy),'heading':ParagraphStyle('heading',parent=body,fontName='DB',fontSize=13.5,leading=18,spaceBefore=13,spaceAfter=8,textColor=navy,keepWithNext=True),'meta':ParagraphStyle('meta',parent=body,fontName='DI',fontSize=8.5,leading=12,spaceAfter=13),'cell':ParagraphStyle('cell',parent=body,fontSize=8,leading=11,spaceAfter=0),'th':ParagraphStyle('th',parent=body,fontName='DB',fontSize=8,leading=11,textColor=colors.white),'equation':ParagraphStyle('equation',parent=body,fontSize=10,leading=16,spaceBefore=4,spaceAfter=12),'bullet':ParagraphStyle('bullet',parent=body,leftIndent=13,firstLineIndent=-10)}
def inline(s):
 s=html.escape(s.replace('—',' - ').replace('–','-').replace('−','-').replace('‑','-'))
 s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#137c82"><u>\1</u></link>',s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'(?<!\*)\*([^*]+)\*',r'<i>\1</i>',s)
 return re.sub(r'`([^`]+)`',r'<font color="#137c82">\1</font>',s)
story=[];lines=Path('docs/project-briefing.md').read_text().splitlines();i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line=='\\[':
  while lines[i].strip()!='\\]':i+=1
  story.append(Paragraph('V<super>π</super>(s) = Pr(eventual correct answer | s, π)',styles['equation']));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[:\-]+',x) for x in cells):rows.append(cells)
   i+=1
  n=len(rows[0]);ratios={2:[.29,.71],3:[.48,.26,.26],4:[.31,.20,.20,.29]}.get(n,[1/n]*n)
  if rows[0][0]=='Method':ratios=[.34,.24,.16,.26]
  data=[[Paragraph(inline(c),styles['th' if r==0 else 'cell']) for c in row] for r,row in enumerate(rows)]
  table=Table(data,colWidths=[492*r for r in ratios],repeatRows=1,hAlign='LEFT',splitByRow=0)
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f7'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c8d6dd'))]))
  story.extend([table,Spacer(1,10)]);continue
 if line.startswith('# '):typ='title';line=line[2:]
 elif line.startswith('## '):
  if line.startswith('## 8.'):story.append(PageBreak())
  typ='heading';line=line[3:]
 elif line.startswith('*Project briefing'):typ='meta'
 elif line.startswith('- '):typ='bullet';line='• '+line[2:]
 elif re.match(r'^\d\. ',line):typ='bullet'
 else:typ='body'
 story.append(Paragraph(inline(line),styles[typ]));i+=1
class NumberedCanvas(canvas.Canvas):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.states=[]
 def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
 def save(self):
  total=len(self.states)
  for state in self.states:
   self.__dict__.update(state);self.setStrokeColor(colors.HexColor('#d4dfe4'));self.setLineWidth(.5);self.line(52,43,544,43)
   self.setFont('D',7.5);self.setFillColor(colors.HexColor('#566874'));self.drawString(52,30,'MATH RL  /  PROJECT BRIEFING');self.drawRightString(544,30,f'{self._pageNumber} / {total}');super().showPage()
  super().save()
out='output/pdf/project-briefing.pdf'
SimpleDocTemplate(out,pagesize=(596,842),leftMargin=52,rightMargin=52,topMargin=48,bottomMargin=58,title='From next-token prediction to cheap critics and formal proof search',author='Math RL Project').build(story,canvasmaker=NumberedCanvas)
pdf=PdfReader(out);print(f'Created {len(pdf.pages)} pages')
for j,p in enumerate(pdf.pages):print(j+1,len(p.extract_text()),p.extract_text()[-100:].replace('\n',' '))
