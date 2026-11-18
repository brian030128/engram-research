"""Static charts for the report. Palette: dataviz reference slots 1-3, validated order."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/work/u4320956/engram-research/'
FIG = R + 'report/figures/'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'text.color': INK,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def clean(ax, axis='x'):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# Paper v2 Table 1: (MoE-27B, Engram-27B)
T1 = {'MMLU': (57.4, 60.4), 'MMLU-Redux': (60.6, 64.0), 'MMLU-Pro': (28.3, 30.1), 'CMMLU': (57.9, 61.9),
      'C-Eval': (58.0, 62.7), 'AGIEval': (38.6, 41.8), 'ARC-Easy': (86.5, 89.0), 'ARC-Challenge': (70.1, 73.8),
      'TriviaQA': (48.8, 50.7), 'TriviaQA-ZH': (74.8, 76.3), 'PopQA': (19.2, 19.4), 'CCPM': (79.6, 87.1),
      'BBH': (50.9, 55.9), 'HellaSwag': (71.8, 72.7), 'PIQA': (71.9, 73.5), 'WinoGrande': (67.6, 67.8),
      'DROP': (55.7, 59.0), 'RACE-Middle': (80.9, 82.8), 'RACE-High': (75.4, 78.2), 'C3': (60.1, 63.6),
      'HumanEval': (37.8, 40.8), 'MBPP': (46.6, 48.2), 'CruxEval-i': (30.7, 32.2), 'CruxEval-o': (34.1, 35.0),
      'GSM8K': (58.4, 60.6), 'MGSM': (46.8, 49.4), 'MATH': (28.3, 30.7)}
FACT = {'TriviaQA', 'TriviaQA-ZH', 'PopQA'}
items = sorted(((k, round(v[1] - v[0], 1)) for k, v in T1.items()), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(8, 7.2))
y = np.arange(len(items))
cols = [S2 if k in FACT else S1 for k, _ in items]
ax.barh(y, [d for _, d in items], color=cols, height=0.68)
ax.set_yticks(y, [k for k, _ in items])
for yy, (k, d) in zip(y, items):
    ax.text(d + 0.08, yy, f'+{d:.1f}', va='center', fontsize=9, color=MUTED)
clean(ax)
ax.set_xlabel('Engram-27B minus MoE-27B (benchmark points)')
ax.set_title('All three factual-recall benchmarks gain less than the median', loc='left', fontsize=13, weight='bold', pad=26)
ax.text(0, 1.012, 'Paper v2 Table 1, single reported values, no error bars. Median gain is +2.5.', transform=ax.transAxes, fontsize=9.5, color=MUTED)
ax.barh([0], [0], color=S2, label='Factual-recall group in the paper\'s Figure 6')
ax.barh([0], [0], color=S1, label='All other benchmarks')
ax.legend(loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig(FIG + 'table1_gains.png', dpi=150)

path = R + 'results/expA/analysis.json'
if os.path.exists(path):
    A = json.load(open(path))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rows = []
    for c in ('wiki_en', 'support_en', 'wiki_zh_tw'):
        for n in ('2', '3', '4'):
            rows += A[c]['orders'][n]['collisions']
    rows.sort(key=lambda r: r['load'])
    L = np.array([r['load'] for r in rows])
    grid = np.logspace(np.log10(max(L.min(), 1e-3)), np.log10(L.max()), 200)
    ax.plot(grid, 1 - np.exp(-grid), color=S1, lw=2, label='Random-hash theory, one head')
    ax.plot(grid, (1 - np.exp(-grid)) ** 8, color=S2, lw=2, label='Random-hash theory, all 8 heads')
    ax.scatter(L, [r['single_head_type'] for r in rows], s=42, color=S1, edgecolor=SURF, linewidth=1.5, zorder=3, label='Measured, one head')
    ax.scatter(L, [r['all_heads_type'] for r in rows], s=42, color=S2, marker='D', edgecolor=SURF, linewidth=1.5, zorder=3, label='Measured, all 8 heads')
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    clean(ax, 'both')
    ax.set_xlabel('Load factor: distinct n-grams per row in one head')
    ax.set_ylabel('Share of distinct n-grams colliding')
    ax.set_title('The hash behaves as random-hash theory predicts', loc='left', fontsize=13, weight='bold', pad=10)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG + 'expA_collisions.png', dpi=150)

    # Band chart at a heavy-load operating point
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    target = next(r for r in A['wiki_en']['orders']['3']['collisions'] if r['M'] == 65536)
    names = list(target['bands'])
    v = [target['bands'][b]['owned_none'] for b in names]
    ax.bar(names, v, color=S1, width=0.55)
    for i, x in enumerate(v):
        ax.text(i, x + 0.015, f'{x:.1%}', ha='center', fontsize=10, color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    clean(ax, 'y')
    ax.tick_params(axis='x', colors=INK)
    ax.set_xlabel('How often the 3-gram occurs in the corpus')
    ax.set_ylabel('Share with no head of its own')
    ax.set_title(f"Under heavy load, rare n-grams lose every row (load {target['load']:.0f})", loc='left', fontsize=13, weight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(FIG + 'expA_rare_ngrams.png', dpi=150)
print('ok')
