"""Add three comparison panels, preserving original raster scatter marks."""
import argparse
import hashlib
import json
from pathlib import Path

from make_poster_value_scatter import ROOT, PANELS, panel_image, plt, np, Image, Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = Image.open(args.source)
    if source.size != (2520, 1620):
        raise ValueError('Expected original 2520 x 1620 scatter image')
    control_path = ROOT/'docs/figures/question-versus-prefix-mc4.png'
    control = Image.open(control_path)
    if control.size != (2400, 1280):
        raise ValueError('Unexpected control image dimensions')
    metrics = json.loads((ROOT/'docs/results/multiseed-value-audit.json').read_text())['metrics']
    diagnostic = json.loads((ROOT/'docs/results/prefix-mc-diagnostic.json').read_text())
    panels = PANELS + [
        ('slice-token', 'SLICE + Ridge / token', '#70518D',
         (131,91,906,776), [34.2,175.5,316.5,458.5,599.5,740.5]),
        ('rms-layer-token', 'RMS layer + Ridge / token', '#2374AB',
         (935,91,1709,776), [32.2,173.5,315.5,456.5,597.5,738.5]),
        ('prefix-mc4', 'Prefix MC control / n = 4', '#70518D', None, None),
    ]
    fig, axes = plt.subplots(2,3,figsize=(15.6,11.9))
    fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.89,wspace=.11,hspace=.43)
    for index, (ax, (key,title,color,bounds,xticks)) in enumerate(zip(axes.flat,panels)):
        if key == 'prefix-mc4':
            crop = np.array(control.crop((1455,206,2234,986)).convert('RGB'))
            # Relocate the source annotation, between the y=1 and y=.75 marks.
            crop[38:83,25:650] = 255
            chromatic = crop.max(2).astype(int)-crop.min(2).astype(int)>8
            enhanced = np.clip(255-(255-crop.astype(float))*1.8,0,255).astype('uint8')
            crop[chromatic] = enhanced[chromatic]
            extent = (-29/716,750/716,-33/716,747/716)
            r = diagnostic['shown_split_metrics']['prefix_mc4']
            annotation = f'MAE {r["mae"]:.3f}  ·  RMSE {r["rmse"]:.3f}'
            # Same plot window as other panels; original diagnostic covers [0,1].
            ax.set_facecolor('white')
            ax.set_axisbelow(True)
            ax.grid(color='#E5E5E5',linewidth=.6)
        else:
            crop = panel_image(source,bounds)
            if key == 'slice-token':
                # Old in-panel seed legend occupies empty negative-value space.
                crop[550:678,615:770] = 255
            scale = xticks[-1]-xticks[0]
            extent = (-xticks[0]/scale,(crop.shape[1]-xticks[0])/scale,
                      (543.5-crop.shape[0])/422.5,543.5/422.5)
            r=metrics[key]
            annotation=f'MAE {r["mae_mean"]:.3f}  ·  seed SD {r["mae_seed_sd"]:.3f}'
        ax.imshow(crop,extent=extent,aspect='auto',interpolation='none',zorder=2)
        ax.set_xlim(-.0485,1.049); ax.set_ylim(-.335,1.2864)
        ax.set_xticks([0,.2,.4,.6,.8,1],['0','0.2','0.4','0.6','0.8','1.0'])
        ax.set_yticks([-.2,0,.2,.4,.6,.8,1,1.2],['−0.2','0','0.2','0.4','0.6','0.8','1.0','1.2'])
        if index%3:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=14,length=4,width=.8,pad=6)
        for spine in ax.spines.values():
            spine.set_color('#536777'); spine.set_linewidth(.9)
        ax.set_title(title,fontsize=19,fontweight='bold',color=color,pad=49)
        ax.text(.5,1.045,annotation,transform=ax.transAxes,ha='center',va='bottom',
                fontsize=16,color='#304657')
    fig.text(.018,.54,'Predicted value / sampled baseline',rotation=90,ha='center',va='center',fontsize=17)
    fig.text(.525,.13,'Monte Carlo reference value',ha='center',fontsize=18,color='#263A4A')
    handles=[Line2D([],[],linestyle='none',marker=m,markersize=7,color='#536777',label=str(s))
             for m,s in zip(['o','s','^'],[1101,1102,1103])]
    handles.append(Line2D([],[],color='black',linestyle='--',lw=1.4,label='Ideal agreement'))
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,.097),ncol=4,
               frameon=False,fontsize=13,columnspacing=1.4,handlelength=1.8)
    notes = [
        '3 actor seeds · 50 questions and 200 states per seed · actor-specific MC references',
        'Original panels: 128 MC continuations/state. Prefix control: 4 sampled continuations vs. 124 held-out, GRPO actors only.',
        'GRPO uses 4 extra answers/question; prefix control uses 4/state. Control is exploratory sampling, not a learned critic or PPO run.',
    ]
    for y,note in zip([.066,.043,.02],notes):
        fig.text(.53,y,note,ha='center',fontsize=11,color='#415362')
    stem=ROOT/'docs/figures/poster-value-scatter-six'
    fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    pdf=ROOT/'output/pdf/poster-value-scatter-six.pdf'
    pdf.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdf,facecolor='white')
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({
        'original_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'control_sha256': hashlib.sha256(control_path.read_bytes()).hexdigest(),
        'rendering': 'Preserved raster points, contrast enhanced; vector axes and labels. No reconstructed points.',
        'normalized_layer': 'RMS-normalized token-level layer readout',
        'control': diagnostic,
    },indent=2)+'\n')
    print(stem.with_suffix('.png')); print(pdf)


if __name__ == '__main__':
    main()
