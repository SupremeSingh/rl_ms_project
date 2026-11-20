"""Compare controlled update time with saved update-120 Ridge conditioning."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    data = json.loads((ROOT / 'docs/results/controlled-cost-summary.json').read_text())['metrics']
    conditioning = json.loads((ROOT / 'docs/results/multiseed-ridge-conditioning.json').read_text())['metrics']
    arms = ['slice-token', 'slice-chunk', 'rms-layer-token', 'rms-layer-chunk', 'ppo', 'grpo']
    labels = ['SLICE\ntoken', 'SLICE\nchunk', 'RMS layer\ntoken',
              'RMS layer\nchunk', 'PPO', 'GRPO']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': True, 'axes.spines.right': True,
                         'axes.spines.left': True, 'axes.spines.bottom': True,
                         'axes.edgecolor': '#222222', 'axes.linewidth': 1.,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(arms))
    for regime, color, marker, offset, label in [
        ('initial', '#2878B5', 'o', -.10, 'Initial'),
        ('accumulated', '#E88A24', 's', .10, 'Accumulated'),
    ]:
        rows = [data[f'{a}/{regime}'] for a in arms]
        axes[0].errorbar(x + offset, [r['median_seconds'] for r in rows],
                         yerr=[r['sd_seconds'] for r in rows], fmt=marker,
                         color=color, markersize=6, capsize=3, elinewidth=1.4,
                         label=label, zorder=3)
    for i, arm in enumerate(arms[:4]):
        values=np.array(conditioning[arm])
        center=float(np.mean(values))
        axes[1].scatter(i, center, color='#2878B5', s=45, zorder=3)
        axes[1].annotate(f'{center:.2e}', (i, center), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=10)
    for ax, title, ylabel, count in zip(axes, ['Compute', 'Conditioning'],
                                  ['Update time (s)', r'Condition number $\kappa_2(A + \epsilon I)$'], [6, 4]):
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(count), labels[:count])
        ax.set_xlim(-.5, count - .5)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.8)
        ax.tick_params(axis='x', labelsize=11, pad=8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 190)
    axes[0].set_yticks([0, 40, 80, 120, 160])
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e4, 3e6)
    axes[0].legend(frameon=True, loc='upper left', fontsize=10)
    fig.suptitle('Compute: 3 controlled rounds · conditioning: 3 actor seeds', fontsize=12, y=.98)
    fig.text(.5, .025, r'Time: median ± round SD. Conditioning: mean of 3 seeds at update 120; $\epsilon=0.01$. Separate measurements.',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.22, wspace=.25)
    target = ROOT / 'docs/figures/compute-conditioning'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(target.with_suffix('.' + suffix), dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
