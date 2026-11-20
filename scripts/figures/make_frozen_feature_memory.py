"""Calculated feature-cache storage; excludes frozen backbone and working memory."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
values=[w*4*1_000_000/2**30 for w in [7680,1536,768]]
fig,ax=plt.subplots(figsize=(8,4.1))
ax.bar(range(3),values,color=['#A0AAB4','#356A9B','#188B86'],width=.52,zorder=3)
ax.set_xticks(range(3),['Five-layer readout\n7,680 features','Single-layer readout\n1,536 features','SLICE\n768 features'])
ax.set_ylabel('Feature-cache memory (GiB)');ax.set_ylim(0,35)
ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E5E5',linewidth=.8)
for i,v in enumerate(values):ax.text(i,v+.7,f'{v:.2f}',ha='center')
ax.set_title('Frozen feature storage',fontsize=16,pad=14)
fig.text(.53,.045,'1 million prefix vectors · float32\nCalculated feature storage only; shared transformer weights and working memory excluded.',ha='center',fontsize=9,color='#555555',linespacing=1.5)
fig.subplots_adjust(left=.12,right=.98,top=.82,bottom=.28)
for ext in ['png','pdf','svg']:fig.savefig((ROOT/'docs/figures/frozen-feature-memory').with_suffix('.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
plt.close(fig)
