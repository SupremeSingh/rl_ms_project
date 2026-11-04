"""Plot saved clarification results without refitting or selecting on test scores."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'slice-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=('rms_two_thirds','transferred_slice','target_slice')
LABELS=('RMS two-thirds (1,536 features)','Transferred SLICE (768 features)','Target-trained SLICE (768 features)')
COLORS=('#415a77','#008577','#c06c2c')
DOMAINS=('gsm8k','math')
TITLES=('GSM8K (reused test questions)','MATH-500 numeric eligible subset')
NOTE='Offline prediction; frozen Qwen. Five compressor seeds share generated answers. Inspected data; exploratory results.'


def plot(summary,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    means=summary['selected_question_brier'];budgets=sorted({int(k.split('/')[1]) for k in means})
    saved=[]
    def save(fig,name):
        for ext in ('png','pdf'):
            path=out/f'{name}.{ext}';fig.savefig(path,bbox_inches='tight');saved.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),sharey=True)
    for domain,title,ax in zip(DOMAINS,TITLES,axes):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            y=[means[f'{domain}/{b}/{arm}'] for b in budgets]
            ax.plot(budgets,y,'o-',color=color,label=label,linewidth=2,markersize=5)
        ax.set_xscale('log');ax.set_xticks(budgets,labels=budgets);ax.set_xlabel('Target questions used to fit Ridge');ax.set_title(title);ax.grid(alpha=.18)
    axes[0].set_ylabel('Test question-mean raw Brier (lower is better)')
    axes[1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Feature transfer: prediction versus target head-fitting data',fontsize=13)
    fig.text(.5,-.025,NOTE+'\nPoints are seed means; differences and uncertainty are shown separately.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'learning-curves')

    fig,axes=plt.subplots(2,2,figsize=(11.5,7.6),sharex=True)
    for col,(domain,title) in enumerate(zip(DOMAINS,TITLES)):
        for row,other in enumerate(('rms_two_thirds','target_slice')):
            ax=axes[row,col];values=[];low=[];high=[]
            for b in budgets:
                record=summary['paired'][f'{domain}/{b}/transferred_slice_minus_{other}']
                value=record['mean'];lo,hi=record['interval_95'];values.append(value);low.append(value-lo);high.append(hi-value)
            ax.errorbar(budgets,values,yerr=[low,high],fmt='o-',color=COLORS[1],capsize=4)
            ax.axhline(0,color='#666',linestyle='--',linewidth=1);ax.set_xscale('log');ax.set_xticks(budgets,labels=budgets);ax.grid(alpha=.18)
            ax.set_title(title+'\nTransferred SLICE minus '+('RMS two-thirds' if row==0 else 'target SLICE'),fontsize=10)
            ax.set_ylabel('Raw Brier difference\nNegative favors transferred SLICE')
            if row==1:ax.set_xlabel('Target questions used to fit Ridge')
    fig.suptitle('Paired prediction differences with descriptive 95% intervals',fontsize=13)
    fig.text(.5,-.025,'Intervals: approximate seed/question bootstrap; 12 comparisons, unadjusted. Crossing zero is inconclusive, not equivalence.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'paired-differences')

    # Every fixed epsilon was evaluated only after validation locked all models.
    # Display the shared preregistered grid; no test-based winner is chosen here.
    metrics=summary.get('fixed_epsilon_metrics')
    if metrics:
        epsilons=(.01,1.,10.,100.,1000.,10000.)
        fig,axes=plt.subplots(len(budgets),2,figsize=(11.5,3*len(budgets)),squeeze=False)
        for row,b in enumerate(budgets):
            for col,(domain,title) in enumerate(zip(DOMAINS,TITLES)):
                ax=axes[row,col]
                for arm,label,color in zip(ARMS,LABELS,COLORS):
                    values=[]
                    for eps in epsilons:
                        records=[v for k,v in metrics.items() if k.split('/')[:3]==[domain,str(b),arm] and float(k.split('/')[-1])==eps]
                        if len(records)!=5:raise ValueError(f'Expected five seeds for {domain}/{b}/{arm}/{eps}')
                        values.append(np.mean([v['question_brier'] for v in records]))
                    ax.plot(epsilons,values,'o-',label=label,color=color,markersize=4)
                ax.set_xscale('log');ax.set_xlabel('Ridge epsilon (raw-sum system)');ax.set_ylabel('Test question-mean raw Brier');ax.set_title(f'{title}; {b} training questions',fontsize=10);ax.grid(alpha=.18)
        axes[0,1].legend(fontsize=7)
        fig.suptitle('Fixed-epsilon test diagnostics (not a selection rule)',fontsize=13)
        fig.text(.5,-.01,'Models/epsilon were selected on validation first. Do not select a new recipe from these test curves.',ha='center',fontsize=8)
        fig.tight_layout();save(fig,'epsilon-sensitivity')
    return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();summary=json.loads((a.study/'summary.json').read_text())
    for path in plot(summary,a.out or a.study/'figures'):print(path)
    if not summary.get('fixed_epsilon_metrics'):print('Fixed-epsilon records absent: generated learning curves and paired intervals only.')
if __name__=='__main__':main()
