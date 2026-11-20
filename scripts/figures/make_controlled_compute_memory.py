"""Render the completed fixed-workload benchmark without rescaling memory differences."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    data = json.loads((ROOT / 'docs/results/controlled-cost-summary.json').read_text())['metrics']
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
        axes[1].scatter(x + offset, [r['peak_memory_gib'] for r in rows],
                        color=color, marker=marker, s=40, label=label, zorder=3)
    for ax, title, ylabel in zip(axes, ['Compute', 'Memory'],
                                  ['Update time (s)', 'Peak allocated memory (GiB)']):
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.set_xlim(-.5, len(arms) - .5)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.8)
        ax.tick_params(axis='x', labelsize=11, pad=8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 190)
    axes[0].set_yticks([0, 40, 80, 120, 160])
    axes[1].set_ylim(25.4, 26.0)
    axes[1].set_yticks(np.arange(25.4, 26.01, .1))
    axes[0].legend(frameon=True, loc='upper left', fontsize=10)
    for i, arm in enumerate(arms):
        value=data[f'{arm}/initial']['peak_memory_gib']
        axes[1].text(i, value + .025, f'{value:.2f}', ha='center', fontsize=10)
    fig.suptitle('3 rounds · same allocation · identical workloads', fontsize=12, y=.98)
    fig.text(.5, .025, 'Error bars: between-round SD. Memory: allocator peak, not total GPU reservation.',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.22, wspace=.25)
    target = ROOT / 'docs/figures/controlled-update-cost'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(target.with_suffix('.' + suffix), dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
