"""Faithful three-panel reformat of the supplied scatter figure.

Point marks are raster crops, not reconstructed observations. Axes/labels are
redrawn at the original pixel grid positions. No jitter, clipping, or new points.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/slice-figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42,
                    'svg.fonttype':'none', 'axes.labelcolor':'#263A4A',
                    'xtick.color':'#415362', 'ytick.color':'#415362'})
PANELS = [
    ('slice-chunk', 'SLICE + Ridge / chunk', '#70518D',
     (1736,91,2511,776), [34.2,175.5,316.5,458.5,599.5,740.5]),
    ('ppo', 'PPO value network', '#B57811',
     (935,841,1709,1526), [32.2,173.5,315.5,456.5,597.5,738.5]),
    ('grpo', 'GRPO question baseline', '#008967',
     (1736,841,2511,1526), [34.2,175.5,316.5,458.5,599.5,740.5]),
]


def panel_image(source, bounds):
    crop = np.array(source.crop(bounds).convert('RGB'))
    # The old MAE annotation occupies this otherwise empty area above y=1.1.
    # Remove only its lettering; keep the grid above it and all plotted marks.
    crop[39:68, 20:650] = 255
    # Increase only chromatic mark contrast; leave the gray grid/black line intact.
    colorful = (crop.max(2).astype(int)-crop.min(2).astype(int)) > 8
    enhanced = np.clip(255-(255-crop.astype(float))*1.8,0,255).astype('uint8')
    crop[colorful] = enhanced[colorful]
    return crop


def render(source, metrics, vertical):
    fig, axes = plt.subplots(3,1,figsize=(6.6,14.2)) if vertical else plt.subplots(1,3,figsize=(15.6,6.35))
    if vertical:
        fig.subplots_adjust(left=.17,right=.97,bottom=.14,top=.92,hspace=.46)
        title_size, tick_size, metric_size = 18, 13, 14
    else:
        fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.78,wspace=.11)
        title_size, tick_size, metric_size = 19, 14, 16
    for index, (ax, (key,title,color,bounds,xticks)) in enumerate(zip(axes,PANELS)):
        crop = panel_image(source,bounds)
        ax.imshow(crop,extent=(0,crop.shape[1],crop.shape[0],0),aspect='auto',interpolation='none')
        ax.set_xlim(0,crop.shape[1]); ax.set_ylim(crop.shape[0],0)
        ax.set_xticks(xticks,['0','0.2','0.4','0.6','0.8','1.0'])
        ax.set_yticks([628.5,543.5,458.5,373.5,289,204.5,119.5,34.5],
                      ['−0.2','0','0.2','0.4','0.6','0.8','1.0','1.2'])
        if not vertical and index:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=tick_size,length=4,width=.8,pad=6)
        for spine in ax.spines.values():
            spine.set_color('#536777'); spine.set_linewidth(.9)
        ax.set_title(title,fontsize=title_size,fontweight='bold',color=color,pad=44 if vertical else 49)
        r=metrics[key]
        ax.text(.5,1.045,f'MAE {r["mae_mean"]:.3f}  ·  seed SD {r["mae_seed_sd"]:.3f}',
                transform=ax.transAxes,ha='center',va='bottom',fontsize=metric_size,color='#304657')
        if vertical and index == 2:
            ax.set_xlabel('Monte Carlo reference value',fontsize=13,labelpad=7)
    if vertical:
        fig.text(.028,.55,'Predicted value / sampled baseline',rotation=90,ha='center',va='center',fontsize=15)
        legend_y, detail_y, caveat_y = .083,.052,.028
        detail='3 seeds · 50 questions/seed · 128 MC continuations/state'
        caveat='Actor-specific references; no architecture-only ranking.\nGRPO uses 4 extra answers/question; no learned critic.'
        suffix='-column'
    else:
        fig.text(.018,.515,'Predicted value / sampled baseline',rotation=90,ha='center',va='center',fontsize=17)
        fig.text(.525,.155,'Monte Carlo reference value',ha='center',fontsize=18,color='#263A4A')
        legend_y, detail_y, caveat_y = .105,.055,.018
        detail='3 actor seeds · 50 questions and 200 states per seed · 128 MC continuations per state'
        caveat='Each panel uses its own actor’s reference values. GRPO is a question baseline using 4 extra answers/question, not a learned critic.'
        suffix=''
    handles=[Line2D([],[],linestyle='none',marker=m,markersize=7,color='#536777',label=str(s))
             for m,s in zip(['o','s','^'],[1101,1102,1103])]
    handles.append(Line2D([],[],color='black',linestyle='--',lw=1.4,label='Ideal agreement'))
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,legend_y),
               ncol=2 if vertical else 4,frameon=False,fontsize=11 if vertical else 13,
               columnspacing=1.4,handlelength=1.8)
    fig.text(.53,detail_y,detail,ha='center',fontsize=9.6 if vertical else 12,color='#415362')
    fig.text(.53,caveat_y,caveat,ha='center',fontsize=9.2 if vertical else 11,color='#415362',linespacing=1.5)
    stem=ROOT/'docs/figures'/f'poster-value-scatter{suffix}'
    fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    pdf=ROOT/'output/pdf'/f'poster-value-scatter{suffix}.pdf'
    pdf.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdf,facecolor='white')
    plt.close(fig)
    print(stem.with_suffix('.png')); print(pdf)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    args=p.parse_args()
    source=Image.open(args.source)
    if source.size != (2520,1620):
        raise ValueError('This layout requires the original 2520 x 1620 source image')
    record=json.loads((ROOT/'docs/results/multiseed-value-audit.json').read_text())
    for vertical in (False,True):
        render(source,record['metrics'],vertical)
    (ROOT/'docs/figures/poster-value-scatter-source.json').write_text(json.dumps(dict(
        source_name=args.source.name,sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        rendering='Raster point crops with vector axes and labels; chromatic mark contrast enhanced 1.8x.',
        annotation_removal='Original MAE lettering above y=1.1 removed; annotation relocated above each panel.',
        data='Original marks retained, no digitized/reconstructed coordinates, jitter, or prediction clipping.',
        metrics_source='docs/results/multiseed-value-audit.json'),indent=2)+'\n')


if __name__=='__main__':
    main()
