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

ARMS = (
    ('final/ridge', 'Final layer', 1536, '#4C72B0'),
    ('two_thirds/ridge', 'Two-thirds layer', 1536, '#DD8452'),
    ('normalized_concat/ridge', 'Five layers', 7680, '#55A868'),
    ('plain_768/ridge', 'Plain', 768, '#8C8C8C'),
    ('sparse_768/ridge', 'SLICE (sparse)', 768, '#C44E52'),
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
    """Ordinary Matplotlib styling, matching the supplied simple bar-chart examples."""
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.edgecolor': 'black', 'axes.titleweight': 'bold',
        'axes.titlesize': 16, 'axes.labelsize': 14, 'axes.labelweight': 'bold',
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
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
    """Simple vertical bars; each category explicitly names its feature count."""
    results = summary['results']
    values = [100 * checked_metric(results[key], 'accuracy') for key, *_ in ARMS]
    labels = [f'{label}\n{size:,}' for _, label, size, _ in ARMS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=.11, right=.98, bottom=.22, top=.86)
    bars = ax.bar(range(len(ARMS)), values, width=.68,
                  color=[arm[3] for arm in ARMS], edgecolor='black', linewidth=.8)
    ax.set_xticks(range(len(ARMS)), labels=labels)
    ax.set_ylim(0, 82)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel('Prediction accuracy (%)')
    ax.set_xlabel('Representation (number of features)', labelpad=10)
    ax.set_title('Prediction accuracy vs. representation size', pad=14)
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5,
                 fontsize=11, fontweight='bold')
    ax.grid(False)
    return save_figure(fig, directory, 'prediction-quality-vs-size', dpi)


def before_generation(summary, directory, dpi):
    """Three plain horizontal bars; values printed at their ends, no side tables."""
    results = summary['by_prefix']['question_only']['results']
    values = [100 * checked_metric(results[key], 'accuracy') for key, *_ in BEFORE_ARMS]
    labels = ['Final layer', 'Five layers', 'SLICE: Sparse 768']
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(left=.27, right=.97, bottom=.22, top=.83)
    bars = ax.barh(range(len(BEFORE_ARMS)), values, height=.65, color='#1f77b4')
    ax.set_yticks(range(len(BEFORE_ARMS)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xticks(range(0, 81, 10))
    ax.set_xlabel('Prediction accuracy (%)', labelpad=7)
    ax.set_title('Prediction before generation', pad=12)
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#DDDDDD', linestyle='--', linewidth=.7)
    ax.set_axisbelow(True)
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
