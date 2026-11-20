"""Reported mean fitting workloads from the three-seed 120-update value study."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = {
        'source': 'experiments/22-multiseed-value-audit.MD; rounded reported three-seed means',
        'scope': '120-update online training runs, seeds 1101–1103. Fitting component only, not offline end-to-end costs.',
        'slice-token': {'fitted_states_mean': 5761239, 'ridge_fit_wall_minutes_mean': 1.86},
        'slice-chunk': {'fitted_states_mean': 324878, 'ridge_fit_wall_minutes_mean': .40},
    }
    (ROOT / 'docs/results/chunk-fitting-workload.json').write_text(json.dumps(source, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ['#356A9B', '#188B86']
    x = np.arange(2)
    panels = [
        ('Fitted states', 'States (millions)', [5.761239, .324878], ['5.76M', '0.325M'], 7.4, '≈18× fewer states'),
        ('Ridge fitting', 'Wall time (min)', [1.86, .40], ['1.86', '0.40'], 2.4, '≈79% less fitting time'),
    ]
    for ax, (title, ylabel, values, labels, upper, annotation) in zip(axes, panels):
        ax.bar(x, values, color=colors, width=.48, zorder=3)
        ax.set_xticks(x, ['SLICE/token', 'SLICE/chunk'])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_ylim(0, upper)
        ax.set_xlim(-.6, 1.6)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.8)
        for i, (v, label) in enumerate(zip(values, labels)):
            ax.text(i, v + upper*.025, label, ha='center', va='bottom', fontsize=12)
        ax.text(.97, .94, annotation, transform=ax.transAxes, ha='right',
                va='top', fontsize=11, fontweight='bold', color='#166D69')
    fig.suptitle('Token vs chunk fitting workload', fontsize=16, y=.98)
    fig.text(.5, .035, '3 seeds · 120 updates/run · mean workload\nRidge fitting only; excludes feature extraction and actor updates.',
             ha='center', va='center', fontsize=10, color='#555555', linespacing=1.5)
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.23, wspace=.32)
    target = ROOT / 'docs/figures/token-vs-chunk-fitting-workload'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(target.with_suffix('.'+extension), dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
