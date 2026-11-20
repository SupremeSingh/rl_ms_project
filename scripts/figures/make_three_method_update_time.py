"""Wide, time-only view of the controlled PPO-update benchmark."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    metrics = json.loads((ROOT / 'docs/results/controlled-cost-summary.json').read_text())['metrics']
    arms = ['slice-token', 'slice-chunk', 'ppo']
    labels = ['SLICE/token', 'SLICE/chunk', 'Full transformer critic']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9, 3.9))
    x = np.arange(3)
    for regime, color, marker, offset, label in [
        ('initial', '#2878B5', 'o', -.065, 'Initial'),
        ('accumulated', '#E88A24', 's', .065, 'Accumulated'),
    ]:
        rows = [metrics[f'{a}/{regime}'] for a in arms]
        ax.errorbar(x + offset, [r['median_seconds'] for r in rows],
                    yerr=[r['sd_seconds'] for r in rows], fmt=marker,
                    color=color, markersize=6, capsize=4, elinewidth=1.5, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlim(-.45, 2.45)
    ax.set_ylim(0, 190)
    ax.set_yticks([0, 40, 80, 120, 160])
    ax.set_ylabel('Update time (s)')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E5E5', linewidth=.8)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_title('Controlled PPO update time', fontsize=15, pad=12)
    fig.text(.53, .025, '3 rounds · identical workloads · generation excluded · error bars: round SD',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.10, right=.98, bottom=.19, top=.86)
    target = ROOT / 'docs/figures/controlled-three-method-update-time'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(target.with_suffix('.' + suffix), dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
