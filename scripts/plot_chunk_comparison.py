"""Plot only the saved report; no actor execution, fitting or test reselection."""
import argparse
import json
from pathlib import Path


def plot(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((study/'summary.json').read_text());methods=summary['methods']
    labels={'slice-token':'SLICE / token','slice-chunk':'SLICE / chunk','layer-token':'Layer / token','layer-chunk':'Layer / chunk'}
    order=list(labels);colors=['#8ab6d6','#2166ac','#a6dba0','#1b7837']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,arm in enumerate(order):
        values=[100*v for v in methods[arm]['accuracy_by_seed'].values()]
        axes[0].scatter([i+(j-2)*.035 for j in range(len(values))],values,s=24,color=colors[i])
        axes[0].plot(i,100*methods[arm]['accuracy_mean'],'_',color='black',markersize=18,markeredgewidth=2)
        axes[2].scatter(methods[arm]['training_gpu_hours_mean'],100*methods[arm]['accuracy_mean'],color=colors[i],s=45,label=labels[arm])
    axes[0].set_xticks(range(4),[labels[a] for a in order],rotation=25,ha='right')
    axes[0].set_ylabel('Final actor accuracy (%)');axes[0].set_title('Five actor seeds; black line = mean')
    for i,rep in enumerate(('slice','layer')):
        p=summary['primary'][rep];lo,hi=p['adjusted_interval_pp']
        axes[1].hlines(i,lo,hi,color=colors[1 if rep=='slice' else 3],linewidth=3)
        axes[1].plot(p['mean_pp'],i,'o',color=colors[1 if rep=='slice' else 3])
    axes[1].axvline(0,color='gray',linestyle='--',linewidth=1)
    axes[1].set_yticks([0,1],['SLICE','Raw layer']);axes[1].set_ylim(-.6,1.6)
    axes[1].set_xlabel('Chunk minus token accuracy (pp)');axes[1].set_title('Adjusted primary intervals')
    axes[2].set_xlabel('Online training GPU-hours / run');axes[2].set_ylabel('Final actor accuracy (%)')
    axes[2].set_title('Mean accuracy versus training GPU time');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Fixed budget: 120 updates × 64 answers; paired seeds and test questions')
    out=study/'figures';out.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out/f'chunk-comparison.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    plot(p.parse_args().study)
