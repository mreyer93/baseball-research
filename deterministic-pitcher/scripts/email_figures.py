#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:40:00 2026

@author: reyer
"""

import pandas as pd
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Two figures for a short email about the deterministic pitcher, redrawn from the Phase 1 results so they stand on their own (no project shorthand, no project figure numbers)
# Data: Retrosheet game logs 2022 to 2025. The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

project_dir = pathlib.Path(__file__).resolve().parent.parent
results_dir = project_dir / 'results'
out_dir = project_dir / 'figures' / 'email'
out_dir.mkdir(exist_ok=True)
surface = '#fcfcfb'
ink = '#0b0b0b'
ink_secondary = '#52514e'
ink_muted = '#898781'
diverging_cmap = LinearSegmentedColormap.from_list('d_minus_alt', ['#e34948', '#f0efec', '#2a78d6'])
plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface, 'axes.edgecolor': ink_muted, 'axes.labelcolor': ink_secondary, 'xtick.color': ink_muted, 'ytick.color': ink_muted, 'text.color': ink, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
source_note = 'Data: Retrosheet game logs, 2022 to 2025 (information obtained free of charge from and copyrighted by Retrosheet, www.retrosheet.org).'


# Step 1 Where he cannot lose: the middle-offense panel of the Phase 1 grid
grid = pd.read_csv(results_dir / 'phase1_grid.csv')
table = 100 * grid[grid['tier'] == 'middle'].pivot(index='lead', columns='innings_left', values='d_minus_mop_up').sort_index(ascending=False)
limit = np.abs(table.values).max()
fig, ax = plt.subplots(figsize=(7.2, 7.4))
image = ax.imshow(table.values, cmap=diverging_cmap, vmin=-limit, vmax=limit, aspect='auto')
for r in range(table.shape[0]):
    for c in range(table.shape[1]):
        value = table.values[r, c]
        if abs(value) >= 0.05:
            ax.text(c, r, f'{value:.1f}', ha='center', va='center', fontsize=7, color='white' if abs(value) > 0.6 * limit else ink)
ax.set_xticks(range(table.shape[1]), table.columns)
ax.set_yticks(range(table.shape[0]), [f'{l:+d}' if l else '0' for l in table.index])
ax.plot(np.arange(table.shape[1]) - 0.5, [list(table.index).index(min(n + 1, 8)) + 0.5 if n + 1 <= 8 else -0.5 for n in table.columns], drawstyle='steps-post', color=ink, lw=1.6)
ax.set_xlabel('innings he would pitch (the opponent\'s remaining turns at bat)')
ax.set_ylabel('our lead when he comes in')
colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
colorbar.set_label('win probability, his minus a mop-up reliever\'s (points)', color=ink_secondary)
ax.set_title('Home run then three strikeouts, every inning: against a mop-up reliever', loc='left', fontsize=10, color=ink, pad=26)
ax.text(0, 1.012, 'Above the black line (lead larger than the innings left) he cannot lose. Average offense.', transform=ax.transAxes, fontsize=8, style='italic', color=ink_secondary, va='bottom')
fig.text(0.01, 0.005, source_note, fontsize=6.5, color=ink_muted)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(out_dir / 'deterministic_pitcher_where_he_cannot_lose.png', dpi=200)
plt.close(fig)


# Step 2 What he is worth: games real bullpens lost after leading by more than the innings left
locks = pd.read_csv(results_dir / 'phase1_lock_counts.csv')
counts = locks['blown_leads'].value_counts().reindex(range(0, int(locks['blown_leads'].max()) + 1), fill_value=0)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(counts.index, counts.values, width=0.8, color='#2a78d6', edgecolor=surface, linewidth=2)
mean = locks['blown_leads'].mean()
ax.axvline(mean, color=ink, lw=1.2, ls='--')
ax.set_ylim(0, counts.max() * 1.15)
ax.annotate(f'mean {mean:.1f} a season', (mean, counts.max() * 1.1), xytext=(6, 0), textcoords='offset points', fontsize=8, color=ink_secondary, va='center', bbox=dict(facecolor=surface, edgecolor='none', pad=1))
ax.set_xticks(counts.index)
ax.set_xlabel('games lost after leading by more than the innings left, per team-season')
ax.set_ylabel('team-seasons')
ax.grid(axis='y', color='#e6e5e1', linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title('Leads real bullpens lost that he could not have lost', loc='left', fontsize=10, color=ink, pad=22)
ax.text(0, 1.02, f'All 30 teams, 2022 to 2025 ({len(locks)} team-seasons). Each of these games is a win for him.', transform=ax.transAxes, fontsize=8, style='italic', color=ink_secondary, va='bottom')
fig.text(0.01, 0.01, source_note, fontsize=6.5, color=ink_muted)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out_dir / 'deterministic_pitcher_leads_lost.png', dpi=200)
plt.close(fig)
print('team-seasons', len(locks), 'mean lost leads', round(mean, 2), 'lock games per season', round(locks['eligible_games'].mean(), 1))


# Step 3 Where his value comes from, and why the rest adds so little
# Locked wins from the real games (Phase 1); the value of rest from the Phase 4 season simulation (central case and the range across its sensitivity checks); the share of close-late half-innings started by a top reliever from the Phase 4 diagnostics table, read from its run log
sensitivity = pd.read_csv(results_dir / 'phase4_sensitivity.csv')
rest_central = sensitivity.loc[sensitivity['scenario'] == 'central', 'rest_value'].iloc[0]
rest_low, rest_high = sensitivity['rest_value'].min(), sensitivity['rest_value'].max()
locked = locks['blown_leads'].mean()
diagnostic_line = [l for l in open(results_dir / 'phase4_run_log.txt').read().splitlines() if l.startswith('share of close-late half-innings started by a top reliever')][0]
no_d_fatigue, no_d_fresh, d_fatigue, d_fresh = [float(x) for x in diagnostic_line.split()[-4:]]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]
ax.barh(0, locked, color='#2a78d6', height=0.5, edgecolor=surface, linewidth=2)
ax.barh(0, rest_central, left=locked, color='#1baf7a', height=0.5, edgecolor=surface, linewidth=2)
ax.errorbar(locked + rest_central, 0, xerr=[[rest_central - rest_low], [rest_high - rest_central]], fmt='none', ecolor=ink, elinewidth=1.2, capsize=4)
ax.text(locked / 2, 0, f'locked wins\n{locked:.2f}', ha='center', va='center', color='white', fontsize=9)
ax.annotate(f'rest {rest_central:.2f}\n(range {rest_low:.2f} to {rest_high:.2f})', (locked + rest_high, 0), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color=ink_secondary)
ax.set_xlim(0, 3.6)
ax.set_ylim(-0.6, 0.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('wins a season with the lock rule')
ax.grid(axis='x', color='#e6e5e1', linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title('Where his value comes from', loc='left', fontsize=10, color=ink, pad=22)
ax.text(0, 1.03, 'Locked wins: real games. Rest: 2,000 simulated seasons. Injuries not modelled.', transform=ax.transAxes, fontsize=7.5, style='italic', color=ink_secondary, va='bottom')
ax = axes[1]
labels = ['without him', 'with him', 'if relievers never tired']
values = [100 * no_d_fatigue, 100 * d_fatigue, 100 * no_d_fresh]
ax.barh(range(3), values, color=['#86b6ef', '#2a78d6', '#d9d8d3'], height=0.55, edgecolor=surface, linewidth=2)
for i, v in enumerate(values):
    ax.text(v - 2, i, f'{v:.0f}%', ha='right', va='center', fontsize=8.5, color='white' if i < 2 else ink)
ax.set_yticks(range(3), labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('close late innings started by a top reliever')
ax.grid(axis='x', color='#e6e5e1', linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title('Why rest adds so little', loc='left', fontsize=10, color=ink, pad=22)
ax.text(0, 1.03, 'Close late: 8th inning on, 1 behind to 3 ahead, or extras.', transform=ax.transAxes, fontsize=7.5, style='italic', color=ink_secondary, va='bottom')
fig.text(0.01, 0.01, source_note.replace('game logs', 'game logs and play-by-play'), fontsize=6.5, color=ink_muted)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out_dir / 'deterministic_pitcher_value.png', dpi=200)
plt.close(fig)
print(f'locked {locked:.2f}, rest {rest_central:.3f} ({rest_low:.3f} to {rest_high:.3f}), close-late share without/with/never tired {no_d_fatigue}/{d_fatigue}/{no_d_fresh}')
