"""Plot predeclared transferred-PPO learning curves; no new evaluation or fitting."""
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'slice-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.study/'summary.json').read_text());out=a.study/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),sharey=True)
    for arm,label,color in [('slice','Transferred SLICE + Ridge','#008577'),('rms_two_thirds','RMS two-thirds + Ridge','#415a77')]:
        runs=summary['methods'][arm]['runs'];curves=[r['curve'] for r in runs]
        y=np.mean([[c['accuracy']*100 for c in curve] for curve in curves],axis=0)
        xs=[np.mean([[c['step']*64 for c in curve] for curve in curves],axis=0),np.mean([[c['response_tokens'] for c in curve] for curve in curves],axis=0),
            np.mean([[(c['online_seconds']+r['result']['initialization_seconds']+r['result']['initial_test']['seconds'])*2/3600 for c in r['curve']] for r in runs],axis=0)]
        for ax,x in zip(axes,xs):ax.plot(x,y,'o-',color=color,label=label,markersize=4);ax.grid(alpha=.2)
    for ax,label in zip(axes,['Fresh training answers','Fresh training response tokens','Accounted GPUh through checkpoint']):ax.set_xlabel(label)
    axes[0].set_ylabel('Greedy test answer accuracy (%)');axes[-1].legend(fontsize=8)
    fig.suptitle('Transferred-feature PPO: five-seed mean learning curves')
    fig.text(.5,-.04,'MATH numeric subset; inspected questions. GPUh includes setup, base evaluation and monitoring through each checkpoint. Offline source training/pilots excluded.\nCurves are descriptive means, not a test-selected checkpoint. Primary paired uncertainty is reported separately.',ha='center',fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'learning-efficiency.{ext}',bbox_inches='tight')
    plt.close(fig);print('Saved:',out)
if __name__=='__main__':main()
