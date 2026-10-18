"""Draw two poster figures from saved predictions/labels or an evidence summary.

No model generation, feature extraction or fitting is performed.
Summary-only plotting uses Matplotlib and the Python standard library.
Prediction-data mode also uses the existing NumPy/PyTorch evidence exporter.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

NAVY = '#102E5C'
BLUE = '#477DA4'
TEAL = '#087F73'
GRAY = '#707B85'
ARMS = (
    ('final/ridge', 'FINAL LAYER', 1536, NAVY),
    ('two_thirds/ridge', 'TWO-THIRDS LAYER', 1536, BLUE),
    ('normalized_concat/ridge', 'FIVE LAYERS', 7680, BLUE),
    ('plain_768/ridge', 'PLAIN 768', 768, GRAY),
    ('sparse_768/ridge', 'SLICE: SPARSE 768', 768, TEAL),
)

# Fixed poster subset; chosen for presentation, not selected by test score.
BEFORE_ARMS = tuple(arm for arm in ARMS if arm[0] in (
    'final/ridge', 'normalized_concat/ridge', 'sparse_768/ridge'))


def checked_metric(result, name):
    value = result['mean'][name]
    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid {name}: {value}')
    return float(value)


def load_evidence(path):
    summary = json.loads(path.read_text())
    # These figures are deliberately tied to the reported five-block/768 study.
    # Refuse to relabel a different-dimensional experiment with fixed labels.
    d = summary['dimensions']
    if d['input'] != 7680 or d['bottleneck'] != 768:
        raise ValueError('Expected the five-block 7,680 -> 768 poster study')
    question_only = summary['by_prefix']['question_only']
    for group in (summary['results'], question_only['results']):
        sizes = set()
        for key in ('constant_train_prior', *(arm[0] for arm in ARMS)):
            item = group[key]
            for metric in ('accuracy', 'auroc'):
                checked_metric(item, metric)
            if item['prefixes'] < 1 or item['heads'] < 1:
                raise ValueError('Positive state and head counts required')
            sizes.add(item['prefixes'])
        if len(sizes) != 1:
            raise ValueError('Representations were evaluated on different state counts')
    return summary


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'text.color': NAVY, 'axes.labelcolor': NAVY,
        'xtick.color': NAVY, 'ytick.color': NAVY,
        'axes.edgecolor': '#BDC7D1', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold',
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def save_figure(fig, directory, stem, dpi):
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = directory / f'{stem}.{extension}'
        fig.savefig(path, dpi=dpi, facecolor='white')
        paths.append(path.name)
    plt.close(fig)
    return paths


def quality_vs_size(summary, directory, dpi):
    """Minimal poster panel; method names and two axes carry the comparison."""
    results = summary['results']
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.17, top=.94)
    # A linear feature-count axis avoids needing a log-scale explanation.
    ax.set_xlim(0, 9000)
    ax.set_ylim(65.5, 75)
    ax.xaxis.set_major_locator(FixedLocator([768, 1536, 7680]))
    ax.set_xticklabels(['768', '1,536', '7,680'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([66, 68, 70, 72, 74])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel('Prediction accuracy', labelpad=10, fontsize=13)
    ax.set_xlabel('Number of features', labelpad=12, fontsize=13)
    ax.grid(axis='y', color='#E5EAF0', linewidth=.7)
    ax.set_axisbelow(True)
    for key, _, size, color in ARMS:
        value = 100 * checked_metric(results[key], 'accuracy')
        if key == 'plain_768/ridge':
            ax.scatter(size, value, s=180, facecolors='none', edgecolors=color,
                       linewidths=1.7, zorder=4)
        elif key == 'sparse_768/ridge':
            ax.scatter(size, value, s=65, marker='D', color=color, zorder=5)
        else:
            ax.scatter(size, value, s=75, color=color, zorder=4)
    for key, label, _, color in ARMS[:2]:
        value = 100 * checked_metric(results[key], 'accuracy')
        ax.annotate(label.title(), (1536, value), xytext=(12, 0),
                    textcoords='offset points', va='center', fontsize=11, color=color)
    concat = 100 * checked_metric(results['normalized_concat/ridge'], 'accuracy')
    ax.annotate('Five layers', (7680, concat), xytext=(6800, 73.25),
                ha='center', va='center', fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle='-', color=BLUE, linewidth=.9))
    sparse = 100 * checked_metric(results['sparse_768/ridge'], 'accuracy')
    # Both markers remain at their true values; no jitter or invented separation.
    ax.annotate('SLICE / Plain', (768, sparse), xytext=(1850, 74),
                ha='left', va='center', fontsize=12, color=TEAL, weight='bold',
                arrowprops=dict(arrowstyle='-', color=TEAL, linewidth=.9))
    return save_figure(fig, directory, 'prediction-quality-vs-size', dpi)


def before_generation(summary, directory, dpi):
    group = summary['by_prefix']['question_only']
    results = group['results']
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    fig.subplots_adjust(left=.35, right=.83, bottom=.30, top=.78)
    fig.text(.05, .95, 'PREDICTION BEFORE GENERATION', fontsize=15, weight='bold')
    fig.text(.05, .895, 'Question only. No answer tokens. The language model stays frozen.', fontsize=10.5)
    ax.set_xlim(0, 82)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(-.6, len(BEFORE_ARMS) - .4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(BEFORE_ARMS)), labels=[arm[1] for arm in BEFORE_ARMS])
    ax.tick_params(axis='y', length=0, pad=10)
    ax.set_xlabel('PREDICTION ACCURACY (%)', labelpad=12)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#E5EAF0', linewidth=.7)
    ax.set_axisbelow(True)
    for i, (key, _, _, color) in enumerate(BEFORE_ARMS):
        item = results[key]
        value = 100 * checked_metric(item, 'accuracy')
        ax.barh(i, value, height=.53, color=color, alpha=.95)
        ax.text(value + 1.1, i, f'{value:.2f}%', va='center', fontsize=10, color=color)
        ax.text(1.21, i, f'{checked_metric(item, "auroc"):.3f}',
                transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=11)
    ax.text(1.21, 1.06, 'AUROC', transform=ax.transAxes, ha='center',
            fontsize=10, weight='bold')
    baseline = 100 * checked_metric(results['constant_train_prior'], 'accuracy')
    ax.axvline(baseline, linestyle=(0, (4, 3)), color=GRAY, linewidth=1.1)
    fig.text(.05, .15, f'CONSTANT BASELINE: {baseline:.2f}% ACCURACY / 0.500 AUROC',
             fontsize=10, color=GRAY)
    fig.text(.05, .105, f'{group["questions"]:,} questions; '
             f'{results["sparse_768/ridge"]["prefixes"]:,} sampled answer outcomes.', fontsize=9)
    fig.text(.05, .065, 'Repeated question states, not independent questions; all readouts use Ridge.',
             fontsize=9, color=GRAY)
    fig.text(.05, .025, 'Inspected test questions; verifiable answers only. Descriptive scores.',
             fontsize=9, color=GRAY)
    return save_figure(fig, directory, 'prediction-before-generation', dpi)


def export(summary_path, out, dpi=300):
    if dpi < 72:
        raise ValueError('Use at least 72 DPI; 300 or higher is recommended for printing')
    summary_path = Path(summary_path).resolve()
    summary = load_evidence(summary_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Never let a chart-side export overwrite its input evidence.
    if summary_path == out / 'plotted-values.csv' or summary_path == out / 'figure-manifest.json':
        raise ValueError('Output would overwrite source evidence')
    style()
    files = quality_vs_size(summary, out, dpi) + before_generation(summary, out, dpi)
    with (out / 'plotted-values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'figure', 'representation', 'features', 'accuracy', 'auroc', 'prefixes', 'heads'])
        writer.writeheader()
        for figure, items, arms in [('quality-vs-size', summary['results'], ARMS),
                                    ('before-generation', summary['by_prefix']['question_only']['results'], BEFORE_ARMS)]:
            for key, _, size, _ in arms:
                item = items[key]
                writer.writerow(dict(figure=figure, representation=key, features=size,
                    accuracy=checked_metric(item, 'accuracy'), auroc=checked_metric(item, 'auroc'),
                    prefixes=item['prefixes'], heads=item['heads']))
    manifest = dict(source=str(summary_path), source_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        figures=files, dpi=dpi, matplotlib=matplotlib.__version__,
        scope='Descriptive scores on inspected questions, conditional on verifiable answers. '
              'Bottleneck metrics average three feature-training heads sharing data, not ensemble predictions. '
              'No error bars: marginal confidence intervals are not available in this metric export. '
              'All plotted predictors use Ridge. Feature width excludes compressor parameters and encoder cost.')
    (out / 'figure-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path in files:
        print(out / path)


def from_prediction_data(source, bottlenecks, out, dpi=300):
    """Recompute scores from saved arrays and trajectories before drawing figures."""
    if dpi < 72:
        raise ValueError('Use at least 72 DPI')
    source, bottlenecks, out = (Path(p).resolve() for p in (source, bottlenecks, out))
    for study in (source, bottlenecks):
        if out == study or study in out.parents:
            raise ValueError('Output must be separate from both input studies')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory for prediction-data mode')
    # Imported only for this mode. The exporter verifies saved array/label alignment,
    # trajectory identities and checksums, then derives metrics and position masks.
    # No published accuracy or AUROC is supplied to the plotting code as a constant.
    from export_probe_poster import export as recompute_evidence
    import torch
    torch.set_num_threads(8)
    evidence = out / 'evidence'
    recompute_evidence(source, bottlenecks, evidence, run_benchmark=False)
    export(evidence / 'summary.json', out, dpi)
    manifest_path = out / 'figure-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['input_mode'] = 'recomputed-from-saved-predictions-and-labels'
    manifest['study'] = str(source)
    manifest['bottlenecks'] = str(bottlenecks)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path,
                        help='Original large study: saved trajectories and packed prefix labels')
    inputs.add_argument('--summary', type=Path,
                        help='Optional shortcut: plot a previously computed evidence summary')
    parser.add_argument('--bottlenecks', type=Path,
                        help='Matching bottleneck study containing predictions.npz; required with --source')
    parser.add_argument('--out', type=Path, default=Path('outputs/probe-poster-figures'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.source is not None:
        if args.bottlenecks is None:
            parser.error('--source requires --bottlenecks')
        from_prediction_data(args.source, args.bottlenecks, args.out, args.dpi)
    else:
        if args.bottlenecks is not None:
            parser.error('--bottlenecks is only valid with --source')
        export(args.summary, args.out, args.dpi)


if __name__ == '__main__':
    main()
