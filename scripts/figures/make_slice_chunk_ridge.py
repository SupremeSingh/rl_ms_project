"""Paper figure: the implemented causal SLICE/chunk/Ridge path, without RL."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/slice-figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                     'mathtext.fontset':'dejavusans', 'pdf.fonttype':42,
                     'svg.fonttype':'none'})
C = dict(ink='#142C40', muted='#526674', line='#CBD5DC', blue='#246495',
         bluebg='#F0F6FA', teal='#167D80', tealbg='#EDF7F5', orange='#A75C20',
         orangebg='#FCF4EB', graybg='#F5F7F9', white='#FFFFFF')
fig, ax = plt.subplots(figsize=(15.6,7.8))
fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.98)
ax.set(xlim=(0,15.4),ylim=(0,7.45)); ax.axis('off')

def text(x,y,s,size=12,color='ink',weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=C.get(color,color),fontweight=weight,
                   ha=ha,va=va,linespacing=1.45,**kw)
def box(x,y,w,h,fill='white',edge='line',lw=1.1,r=.10,**kw):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=C.get(fill,fill),edgecolor=C.get(edge,edge),linewidth=lw,**kw)
    ax.add_patch(p); return p
def arrow(x1,y1,x2,y2,color='ink',style='-',lw=1.4,**kw):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
                     color=C.get(color,color),linewidth=lw,linestyle=style,
                     shrinkA=0,shrinkB=0,**kw)
    ax.add_patch(p)
def path(points,color='ink',style='-',lw=1.4):
    ax.plot(*zip(*points[:-1]),color=C[color],lw=lw,ls=style,solid_capstyle='round')
    arrow(*points[-2],*points[-1],color=color,style=style,lw=lw)


fig.set_size_inches(15.6,6.6)
ax.set_ylim(0,6.35)
text(.15,5.98,'SLICE + chunking + Ridge',24,weight='bold')
text(.15,5.52,'Reasoning so far → compact features → predicted correctness',13,color='muted')
for x,n,label,col in [(0.15,'1','Choose prefix states','blue'),(4.10,'2','Read frozen features','teal'),(7.95,'3','Compress','teal'),(11.75,'4','Predict','orange')]:
    text(x,4.98,n,16,color=col,weight='bold')
    text(x+.32,4.98,label,14,weight='bold')

box(.15,3.50,3.4,1.03,'bluebg')
text(.35,4.25,r'Solve $2x+3=11$',14,weight='bold')
for x,w,label in [(.35,.92,r'$c_1$: $2x=8$'),(1.38,.90,r'$c_2$: $x=4$'),(2.39,.94,r'$c_3$: answer')]:
    box(x,3.70,w,.31,'white',r=.04)
    text(x+w/2,3.855,label,11.4,ha='center')
arrow(1.85,3.50,1.85,3.23,'blue')
box(.15,1.93,3.4,1.30)
for j in range(3):
    y=2.86-j*.35
    text(.35,y,rf'$s_{j+1}$',12,color='blue')
    for k in range(j+1):
        x=.91+k*.73
        box(x,y-.13,.63,.26,'bluebg',edge='none',r=.03)
        text(x+.315,y,r'$q$' if k==0 else rf'$c_{k}$',12,color='blue',ha='center')
text(.35,1.65,'Question + preceding chunks',10.8,color='muted')
path([(3.55,2.58),(3.81,2.58),(3.81,3.43),(4.10,3.43)],color='blue')

box(4.10,2.34,2.95,2.19,'tealbg',edge='#BDD8D4')
text(5.575,4.21,'Causal Qwen',15,weight='bold',ha='center')
for i in range(5):
    y=2.97+i*.16
    box(4.55,y,1.62,.11,'white',edge='teal',r=.02,lw=.9)
    ax.add_patch(Rectangle((6.32,y+.01),.23,.09,facecolor=C['teal'],edgecolor='none'))
text(5.575,2.60,'5 layer readouts',11.5,color='teal',ha='center')
arrow(7.05,3.43,7.80,3.43,'teal')

from matplotlib.patches import Polygon
ax.add_patch(Polygon([(7.80,4.09),(10.40,4.09),(9.55,2.95),(8.65,2.95)],closed=True,facecolor=C['tealbg'],edgecolor=C['teal'],lw=1.4))
text(9.10,3.67,'SLICE',19,color='teal',weight='bold',ha='center')
text(9.10,3.25,'7,680 → 768',12,color='teal',ha='center')
for i in range(11):
    ax.add_patch(Rectangle((8.24+i*.16,2.34),.11,.29,facecolor=C['teal'] if i in (0,3,7,9) else '#D9E5E2',edgecolor='none'))
text(9.10,2.09,'768 sparse features',12,color='teal',ha='center')
path([(10.03,2.48),(10.74,2.48),(10.74,3.43),(11.75,3.43)],color='teal')

box(11.75,2.69,1.61,1.52,'orangebg',edge='#E4CDB6')
text(12.555,3.82,'Ridge',16,color='orange',weight='bold',ha='center')
text(12.555,3.24,r'$w^{\mathsf{T}}x_j$',18,ha='center')
arrow(13.36,3.43,13.72,3.43,'orange')
for i in range(3):
    y=3.86-i*.43
    box(13.72,y-.16,1.48,.32,'orangebg',edge='none',r=.03)
    text(14.46,y,rf'$s_{i+1}\to\widehat{{V}}_{i+1}$',12.7,color='orange',ha='center')
text(14.46,2.38,'One value\nper prefix',11.2,color='muted',ha='center')
ax.plot([4.28,4.28,10.34,10.34],[1.85,1.69,1.69,1.85],color=C['teal'],lw=1)
text(7.31,1.39,'Frozen during Ridge updates',12.5,color='teal',weight='bold',ha='center')

box(.15,.18,3.40,.75,'graybg')
text(.35,.66,'Correct / incorrect outcome',12.3,weight='bold')
text(.35,.36,'Training supervision',10.8,color='muted')
arrow(3.55,.555,7.84,.555,'orange')
text(5.58,.80,'verified label',11,color='orange',ha='center')
box(7.84,.18,2.50,.75,'orangebg',edge='#E4CDB6')
text(9.09,.66,r'Accumulate $A,b$',13,color='orange',weight='bold',ha='center')
text(9.09,.36,'Keep past statistics',10.8,color='muted',ha='center')
path([(10.03,2.48),(10.46,2.48),(10.46,1.07),(9.75,1.07),(9.75,.93)],color='teal')
arrow(10.34,.555,11.75,.555,'orange')
box(11.75,.18,1.61,.75,'white',edge='orange')
text(12.555,.555,'Solve for w',12,color='orange',ha='center')
arrow(12.555,.93,12.555,2.69,'orange')
text(12.72,1.52,'update',10.8,color='orange')

target=ROOT/'docs/figures/slice-chunk-ridge'
fig.savefig(target.with_suffix('.svg'),facecolor='white')
fig.savefig(target.with_suffix('.png'),dpi=220,facecolor='white')
pdf=ROOT/'output/pdf/slice-chunk-ridge.pdf'
pdf.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(pdf,facecolor='white')
plt.close(fig)
print(target.with_suffix('.svg')); print(pdf)
