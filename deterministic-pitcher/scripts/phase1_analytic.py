#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:05:00 2026

@author: reyer
"""

import pandas as pd
import gzip
import os
import io
import shutil
import glob
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Phase 1 of the deterministic pitcher project (the plan is in ../PLAN.md)
# The pitcher (D from here on) gives up a solo home run and then strikes out the side, every inning, without exception.
# Working convention for this phase: D enters only at the start of one of our fielding half-innings and pitches every remaining regulation half-inning, so he concedes exactly one run per half-inning he pitches. He never pitches a tie or extra innings (the automatic runner on second would make him a two-run-per-inning pitcher), so a game tied after nine is scored as a coin flip.
# The question for this phase: in which game states does D give us a better chance of winning than a real pitcher, and how much does that depend on how good our offense is?
# No simulation here. Everything is computed from real per-inning run distributions, and the method is checked against how often real teams actually won from each state (Step 7).
# Data: Retrosheet game logs 2022 to 2025 (inning-by-inning line scores for every game). 2022 is the first season with the universal designated hitter and without seven-inning doubleheaders, so pitchers batting do not distort the run distributions.
# The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

project_dir = pathlib.Path(__file__).resolve().parent.parent
data_dir = project_dir / 'data' / 'retrosheet'
fig_dir = project_dir / 'figures'
results_dir = project_dir / 'results'
results_dir.mkdir(exist_ok=True)
years = [2022, 2023, 2024, 2025]
tiers = ['low', 'middle', 'high']

# Figure styling: a light chart surface, recessive grid and axes, ink-coloured text
surface = '#fcfcfb'
ink = '#0b0b0b'
ink_secondary = '#52514e'
ink_muted = '#898781'
# Offense tiers are ordered, so they get one hue stepped light to dark rather than three unrelated colours
tier_colors = {'low': '#86b6ef', 'middle': '#2a78d6', 'high': '#104281'}
# Diverging scale for "D minus the alternative": red where D is worse, blue where D is better, gray at zero
diverging_cmap = LinearSegmentedColormap.from_list('d_minus_alt', ['#e34948', '#f0efec', '#2a78d6'])
plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface, 'axes.edgecolor': ink_muted, 'axes.labelcolor': ink_secondary, 'xtick.color': ink_muted, 'ytick.color': ink_muted, 'text.color': ink, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


# Step 1 Read the game logs and turn each line score into a list of runs by inning
def parse_line_score(line_score):
    # Retrosheet writes one character per inning, an inning of 10 or more runs in parentheses, and an x for a bottom of the 9th that was never played (home team already ahead)
    runs = []
    i = 0
    while i < len(line_score):
        if line_score[i] == '(':
            close = line_score.index(')', i)
            runs.append(int(line_score[i + 1:close]))
            i = close + 1
        elif line_score[i] == 'x':
            runs.append(None)
            i += 1
        else:
            runs.append(int(line_score[i]))
            i += 1
    return runs


games_list = []
for year in years:
    gl = pd.read_csv(data_dir / f'gl{year}.txt', header=None, dtype=str, keep_default_na=False)
    # Retrosheet game log fields (1-indexed in their guide): 1 date, 4 visitor, 7 home, 10 and 11 final scores, 12 length in outs, 20 and 21 line scores
    gl = gl[[0, 3, 6, 9, 10, 11, 19, 20]]
    gl.columns = ['date', 'visitor', 'home', 'visitor_score', 'home_score', 'length_outs', 'visitor_line', 'home_line']
    gl['year'] = year
    games_list.append(gl)
games = pd.concat(games_list, ignore_index=True)
games['visitor_score'] = games['visitor_score'].astype(int)
games['home_score'] = games['home_score'].astype(int)
games['visitor_runs'] = games['visitor_line'].apply(parse_line_score)
games['home_runs'] = games['home_line'].apply(parse_line_score)

# Check the parsing: every line score should add up to the final score
games['visitor_check'] = games.apply(lambda g: sum(r for r in g['visitor_runs'] if r is not None) == g['visitor_score'], axis=1)
games['home_check'] = games.apply(lambda g: sum(r for r in g['home_runs'] if r is not None) == g['home_score'], axis=1)
print('games read:', len(games))
print('line scores that do not add up to the final score:', (~games['visitor_check']).sum(), 'visitor,', (~games['home_check']).sum(), 'home')
print('games ending in a tie:', (games['visitor_score'] == games['home_score']).sum())

# Keep complete regulation games only (the visitor batted all nine innings); rain-shortened games go
games['visitor_innings'] = games['visitor_runs'].apply(len)
print('games shortened before the visitor batted in the 9th (dropped):', (games['visitor_innings'] < 9).sum())
games = games[(games['visitor_innings'] >= 9) & games['visitor_check'] & games['home_check'] & (games['visitor_score'] != games['home_score'])].reset_index(drop=True)
games['extra_innings'] = games['visitor_innings'] > 9
print('games kept:', len(games), ' of which extra innings:', games['extra_innings'].sum())
print('home team win share in extra-inning games:', round((games.loc[games['extra_innings'], 'home_score'] > games.loc[games['extra_innings'], 'visitor_score']).mean(), 3), '(the model scores a tie after nine as 0.5)')
####
# games read: 9719
# line scores that do not add up to the final score: 0 visitor, 0 home
# games ending in a tie: 0
# games shortened before the visitor batted in the 9th (dropped): 22
# games kept: 9697  of which extra innings: 842
# home team win share in extra-inning games: 0.506 (the model scores a tie after nine as 0.5)
####
# 9,719 of a possible 9,720 games are in the files. Extra innings split almost exactly evenly (0.506 for the home team), so scoring a tie after nine as 0.5 is fair.


# Step 2 Rate each team-season's offense, and split the 120 team-seasons into thirds
team_games = pd.concat([pd.DataFrame({'year': games['year'], 'team': games['visitor'], 'runs_scored': games['visitor_score'], 'runs_allowed': games['home_score']}), pd.DataFrame({'year': games['year'], 'team': games['home'], 'runs_scored': games['home_score'], 'runs_allowed': games['visitor_score']})])
team_seasons = team_games.groupby(['year', 'team']).agg(games=('runs_scored', 'size'), runs_scored_per_game=('runs_scored', 'mean'), runs_allowed_per_game=('runs_allowed', 'mean')).reset_index()
team_seasons['run_differential_per_game'] = team_seasons['runs_scored_per_game'] - team_seasons['runs_allowed_per_game']
# Thirds by runs scored per game (tertiles: the bottom 40, middle 40 and top 40 team-seasons)
team_seasons['tier'] = pd.qcut(team_seasons['runs_scored_per_game'], 3, labels=tiers).astype(str)
tier_lookup = team_seasons.set_index(['year', 'team'])['tier']
games['visitor_tier'] = [tier_lookup[(y, t)] for y, t in zip(games['year'], games['visitor'])]
games['home_tier'] = [tier_lookup[(y, t)] for y, t in zip(games['year'], games['home'])]
print(team_seasons.groupby('tier')['runs_scored_per_game'].agg(['count', 'min', 'mean', 'max']).loc[tiers].round(2).to_string())
####
#         count   min  mean   max
# tier
# low        40  3.15  3.96  4.22
# middle     41  4.23  4.41  4.60
# high       39  4.64  4.96  5.86
####


# Step 3 Lay the games out as half-innings, with the score at the start of each one
# Each row is one half-inning, seen from the fielding team, since that is the team that would bring D in
# Top of inning i: the visitor bats, the home team fields. Bottom of inning i: the home team bats, the visitor fields (only if it was played)
half_inning_rows = []
for g in games.itertuples():
    V = g.visitor_runs
    H = g.home_runs
    visitor_won = g.visitor_score > g.home_score
    for i in range(1, 10):
        half_inning_rows.append([g.year, g.Index, i, 'top', g.visitor_tier, g.home_tier, g.home, V[i - 1], sum(H[:i - 1]) - sum(V[:i - 1]), not visitor_won])
        if H[i - 1] is not None:
            half_inning_rows.append([g.year, g.Index, i, 'bottom', g.home_tier, g.visitor_tier, g.visitor, H[i - 1], sum(V[:i]) - sum(H[:i - 1]), visitor_won])
half_innings = pd.DataFrame(half_inning_rows, columns=['year', 'game', 'inning', 'half', 'batting_tier', 'fielding_tier', 'fielding_team', 'runs', 'fielding_lead', 'fielding_team_won'])
# The bottom of the 9th stops the moment the home team takes the lead (a walk-off), so its run counts are cut short and it is only played when the home team is not ahead. It is left out of every run distribution below; the top of the 9th stands in for it.
half_innings['full_half_inning'] = ~((half_innings['inning'] == 9) & (half_innings['half'] == 'bottom'))
print('half-innings:', len(half_innings), ' full half-innings used for run distributions:', half_innings['full_half_inning'].sum())
####
# half-innings: 170241  full half-innings used for run distributions: 164849
####


# Step 4 Per-inning run distributions for each offense tier, and for the league
# A run distribution here is the share of half-innings in which a team scored 0, 1, 2, ... runs (a probability mass function, pmf)
def runs_pmf(runs):
    counts = np.bincount(np.asarray(runs, dtype=int))
    return counts / counts.sum()


full = half_innings[half_innings['full_half_inning']]
tier_inning_pmf = {tier: {i: runs_pmf(full.loc[(full['batting_tier'] == tier) & (full['inning'] == i), 'runs']) for i in range(1, 10)} for tier in tiers}
league_inning_pmf = {i: runs_pmf(full.loc[full['inning'] == i, 'runs']) for i in range(1, 10)}

inning_summary = full.groupby(['batting_tier', 'inning'])['runs'].mean().unstack().loc[tiers].round(3)
print('mean runs per half-inning, by offense tier and inning')
print(inning_summary.to_string())
print('share of scoreless half-innings, by offense tier:', full.groupby('batting_tier')['runs'].apply(lambda r: round((r == 0).mean(), 3)).loc[tiers].to_dict())
####
# mean runs per half-inning, by offense tier and inning
# inning            1      2      3      4      5      6      7      8      9
# batting_tier
# low           0.459  0.395  0.460  0.434  0.450  0.466  0.439  0.423  0.403
# middle        0.497  0.466  0.503  0.504  0.512  0.499  0.476  0.469  0.491
# high          0.591  0.512  0.578  0.570  0.569  0.550  0.541  0.553  0.544
# share of scoreless half-innings, by offense tier: {'low': 0.752, 'middle': 0.733, 'high': 0.707}
####
# The gap between the low and high tiers is about 0.1 runs per half-inning, and the 1st inning runs high (the top of the order bats). Late innings run a little lower than the middle innings, which is the bullpen.


# Step 5 Two real alternatives to D: a mop-up reliever, and an average late-inning pitcher
# Mop-up reliever: runs allowed in the 6th inning or later when the fielding team led or trailed by 5 or more at the start of the half-inning. Those innings mostly go to the last men in the bullpen (and occasionally a position player), so this is a fair stand-in for the 13th pitcher D would replace.
# The selection is not clean: an offense that has just scored a lot may be having a good day, which would make the mop-up line look worse than the pitcher really is. I check how much the threshold matters below.
# Average late-inning pitcher: every full half-inning from the 6th on, whatever the score, which is roughly what a typical bullpen allows.
late = full[full['inning'] >= 6]
mop_up_pmf = runs_pmf(late.loc[late['fielding_lead'].abs() >= 5, 'runs'])
average_late_pmf = runs_pmf(late['runs'])
for threshold in [4, 5, 6]:
    mop = late.loc[late['fielding_lead'].abs() >= threshold, 'runs']
    print(f'mop-up (score gap >= {threshold}): half-innings {len(mop)}, mean runs {mop.mean():.3f}, share scoreless {(mop == 0).mean():.3f}')
print(f'average late-inning pitcher: half-innings {len(late)}, mean runs {late["runs"].mean():.3f}, share scoreless {(late["runs"] == 0).mean():.3f}')
print('D: half-innings any, mean runs 1.000, share scoreless 0.000')
####
# mop-up (score gap >= 4): half-innings 22263, mean runs 0.505, share scoreless 0.729
# mop-up (score gap >= 5): half-innings 15112, mean runs 0.520, share scoreless 0.722
# mop-up (score gap >= 6): half-innings 10037, mean runs 0.532, share scoreless 0.720
# average late-inning pitcher: half-innings 67879, mean runs 0.489, share scoreless 0.736
# D: half-innings any, mean runs 1.000, share scoreless 0.000
####
# D gives up about 0.48 runs per half-inning more than the mop-up reliever (1.000 against 0.520), close to the 0.45 I guessed in PLAN.md. The mop-up line is only 0.03 runs worse than the average late-inning pitcher, and the 4, 5 and 6 run thresholds move it by about 0.01 runs each, so the choice of threshold does not matter much.


# Step 6 Test: can the runs from several innings be added up as if each inning were independent?
# Everything below adds per-inning distributions together (a convolution: the distribution of a sum, computed by assuming each inning knows nothing about the others). Real innings within a game are not fully independent (the same opposing pitcher, the same park, the same weather), so check it before relying on it.
# The test: visitors' total runs in innings 6 to 9 of real games, against the sum of the per-inning distributions for their tier
def add_innings(pmf_list):
    total = np.array([1.0])
    for pmf in pmf_list:
        total = np.convolve(total, pmf)
    return total


visitor_late_totals = {tier: np.array([sum(g.visitor_runs[5:9]) for g in games.itertuples() if g.visitor_tier == tier]) for tier in tiers}
independence_rows = []
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, tier in zip(axes, tiers):
    observed = runs_pmf(visitor_late_totals[tier])
    # First version added the tier distributions, which pool home and visiting teams. Home teams score a little more, so that mixed home advantage into a test meant to be about independence. The comparison has to use visitors on both sides.
    # predicted = add_innings([tier_inning_pmf[tier][i] for i in range(6, 10)])
    predicted = add_innings([runs_pmf(full.loc[(full['batting_tier'] == tier) & (full['inning'] == i) & (full['half'] == 'top'), 'runs']) for i in range(6, 10)])
    r_obs = np.arange(len(observed))
    r_pred = np.arange(len(predicted))
    independence_rows.append([tier, len(visitor_late_totals[tier]), (r_obs * observed).sum(), (r_pred * predicted).sum(), visitor_late_totals[tier].var(), ((r_pred - (r_pred * predicted).sum()) ** 2 * predicted).sum(), observed[0], predicted[0], observed[r_obs >= 4].sum(), predicted[r_pred >= 4].sum()])
    show = 9
    ax.bar(np.arange(show), observed[:show] if len(observed) >= show else np.pad(observed, (0, show - len(observed))), width=0.8, color=tier_colors[tier], alpha=0.45, label='real games')
    ax.plot(np.arange(show), predicted[:show], 'o-', color=tier_colors[tier], lw=2, ms=4, label='innings added independently')
    ax.set_title(f'{tier} offense', color=ink, fontsize=9)
    ax.set_xlabel('visitor runs, innings 6 to 9')
axes[0].set_ylabel('share of games')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Figure 1. Visitor runs in innings 6 to 9: real games against innings added independently', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure1_independence_check.png', dpi=200)
plt.close(fig)
independence = pd.DataFrame(independence_rows, columns=['tier', 'games', 'mean_real', 'mean_added', 'variance_real', 'variance_added', 'scoreless_real', 'scoreless_added', 'four_plus_real', 'four_plus_added'])
print(independence.round(3).to_string(index=False))
####
#   tier  games  mean_real  mean_added  variance_real  variance_added  scoreless_real  scoreless_added  four_plus_real  four_plus_added
#    low   3230      1.648       1.648          3.449           3.264           0.341            0.338           0.142            0.145
# middle   3312      1.885       1.885          4.085           3.935           0.315            0.303           0.181            0.181
#   high   3155      2.154       2.154          5.368           4.827           0.276            0.267           0.216            0.222
####
# The means agree by construction (the mean of a sum is the sum of the means whether or not the innings are independent), so the real test is the spread and the tails. Adding innings independently understates the variance by about 5, 4 and 10 percent (low, middle, high) and the chance of a scoreless stretch by 0.3 to 1.2 points: within a game, runs cluster a little more than independence allows (same opposing pitcher, same park, same weather).
# The consequence: our offense looks slightly more reliable than it is, which slightly flatters D in the states where he needs us to score (Step 10). It does not touch the states where his lead is bigger than the innings left, since there our offense does not matter at all.


# Step 7 Win probability for D and for the alternatives
# State: the start of one of our fielding half-innings, inning i, with our lead L. D (or the alternative) pitches the n = 10 - i half-innings the opponent has left.
# Our remaining batting: as the home team (fielding the top of i) we still bat in innings i to 9; as the visitor (fielding the bottom of i) we have already batted in inning i, so we bat in innings i+1 to 9.
# The bottom of the 9th is cut short by a walk-off, but that never changes who wins, so using a full-inning distribution for it gives the right answer.
def win_probability(lead, our_total_pmf, their_total_pmf):
    # Chance that lead + our runs - their runs ends above zero; a tie after nine counts as half a win
    margin_pmf = np.convolve(our_total_pmf, their_total_pmf[::-1])
    margin = lead + np.arange(len(margin_pmf)) - (len(their_total_pmf) - 1)
    return margin_pmf[margin > 0].sum() + 0.5 * margin_pmf[margin == 0].sum()


def our_remaining_pmf(tier, i, perspective):
    first = i if perspective == 'home' else i + 1
    return add_innings([tier_inning_pmf[tier][j] for j in range(first, 10)])


def d_total_pmf(n):
    pmf = np.zeros(n + 1)
    pmf[n] = 1.0
    return pmf


# Step 8 Test: does this machinery reproduce how often real teams won from each state?
# Real teams use real pitchers, so the comparison is against the model with a league-average opponent offense (per inning) in place of D
states = half_innings.copy()
states['perspective'] = np.where(states['half'] == 'top', 'home', 'visitor')
states['lead_clipped'] = states['fielding_lead'].clip(-10, 10)
league_remaining = {i: add_innings([league_inning_pmf[j] for j in range(i, 10)]) for i in range(1, 10)}
model_cache = {}
for key in states[['fielding_tier', 'inning', 'perspective', 'lead_clipped']].drop_duplicates().itertuples(index=False):
    model_cache[tuple(key)] = win_probability(key.lead_clipped, our_remaining_pmf(key.fielding_tier, key.inning, key.perspective), league_remaining[key.inning])
states['model_wp'] = [model_cache[k] for k in zip(states['fielding_tier'], states['inning'], states['perspective'], states['lead_clipped'])]
states['won'] = states['fielding_team_won'].astype(float)
states['bin'] = pd.cut(states['model_wp'], np.linspace(0, 1, 11), include_lowest=True)
calibration = states.groupby('bin', observed=True).agg(states=('won', 'size'), mean_model=('model_wp', 'mean'), observed=('won', 'mean'))
print(calibration.round(3).to_string())
brier_model = ((states['model_wp'] - states['won']) ** 2).mean()
# Baseline: a lookup table of real win rates by inning, half and lead (in-sample, so it is a generous baseline)
lookup = states.groupby(['inning', 'half', 'lead_clipped'])['won'].transform('mean')
brier_lookup = ((lookup - states['won']) ** 2).mean()
print(f'Brier score (mean squared error of a win probability, lower is better): model {brier_model:.4f}, in-sample lookup table {brier_lookup:.4f}, coin flip 0.2500')
by_tier = states.groupby('fielding_tier').apply(lambda s: pd.Series({'mean_model': s['model_wp'].mean(), 'observed': s['won'].mean()}), include_groups=False).loc[tiers]
print(by_tier.round(3).to_string())

fig, ax = plt.subplots(figsize=(5, 4.4))
ax.plot([0, 1], [0, 1], color=ink_muted, lw=1, ls='--')
for tier in tiers:
    s = states[states['fielding_tier'] == tier]
    c = s.groupby('bin', observed=True).agg(mean_model=('model_wp', 'mean'), observed=('won', 'mean'))
    ax.plot(c['mean_model'], c['observed'], 'o-', color=tier_colors[tier], lw=2, ms=5, label=f'{tier} offense')
ax.set_xlabel('model win probability')
ax.set_ylabel('share of games actually won')
ax.legend(frameon=False, fontsize=8)
ax.set_title('Figure 2. Model win probability against real outcomes', loc='left', x=-0.12, fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure2_calibration.png', dpi=200)
plt.close(fig)
####
#                states  mean_model  observed
# bin
# (-0.001, 0.1]   18777       0.036     0.029
# (0.1, 0.2]      11626       0.146     0.126
# (0.2, 0.3]      10910       0.250     0.234
# (0.3, 0.4]      11397       0.352     0.330
# (0.4, 0.5]      39572       0.459     0.459
# (0.5, 0.6]      17351       0.550     0.565
# (0.6, 0.7]      13575       0.649     0.666
# (0.7, 0.8]      11702       0.749     0.771
# (0.8, 0.9]      12808       0.847     0.865
# (0.9, 1.0]      22523       0.962     0.969
# Brier score (mean squared error of a win probability, lower is better): model 0.1615, in-sample lookup table 0.1620, coin flip 0.2500
#                mean_model  observed
# fielding_tier
# low                 0.456     0.446
# middle              0.509     0.513
# high                0.568     0.580
####
# Calibration is within 2.2 points in every bin, and the model (which is fitted on nothing but per-inning run distributions) scores as well as an in-sample lookup table of real win rates (Brier 0.1615 against 0.1620). The one pattern: teams that are ahead win 1.5 to 2.2 points more often than the model says, and teams behind 1.6 to 2.2 points less. My guess is two things the model does not know: managers save their best relievers for leads, and a team that is ahead is more often the better team, pitching included (the tiers rate offense only).
# So the model is good enough to compare pitchers inside it, and it slightly understates how well real bullpens hold leads. That matters for Step 9, where D is compared with a mop-up reliever, not with a real closer; Step 11 avoids the issue by counting real outcomes.


# Step 9 The grid: D against the mop-up reliever and the average late-inning pitcher, for every lead and inning
# Each cell averages the home and visitor versions of the state, except where only one can happen (a visitor fielding the bottom of the 9th cannot be behind, since the game would already be over)
grid_rows = []
for tier in tiers:
    for i in range(1, 10):
        n = 10 - i
        mop_total = add_innings([mop_up_pmf] * n)
        avg_total = add_innings([average_late_pmf] * n)
        for lead in range(-8, 9):
            perspectives = ['home', 'visitor'] if not (i == 9 and lead < 0) else ['home']
            wp_d = np.mean([win_probability(lead, our_remaining_pmf(tier, i, p), d_total_pmf(n)) for p in perspectives])
            wp_mop = np.mean([win_probability(lead, our_remaining_pmf(tier, i, p), mop_total) for p in perspectives])
            wp_avg = np.mean([win_probability(lead, our_remaining_pmf(tier, i, p), avg_total) for p in perspectives])
            grid_rows.append([tier, i, n, lead, wp_d, wp_mop, wp_avg])
grid = pd.DataFrame(grid_rows, columns=['tier', 'inning', 'innings_left', 'lead', 'wp_d', 'wp_mop_up', 'wp_average'])
grid['d_minus_mop_up'] = grid['wp_d'] - grid['wp_mop_up']
grid['d_minus_average'] = grid['wp_d'] - grid['wp_average']
grid.to_csv(results_dir / 'phase1_grid.csv', index=False)


def print_grid(tier, column):
    table = grid[grid['tier'] == tier].pivot(index='lead', columns='innings_left', values=column).sort_index(ascending=False)
    print(f'{column}, {tier} offense, percentage points (rows: our lead, columns: half-innings D would pitch)')
    print((100 * table).round(1).to_string())


for tier in tiers:
    print_grid(tier, 'd_minus_mop_up')
####
# d_minus_mop_up, low offense, percentage points (rows: our lead, columns: half-innings D would pitch)
# innings_left     1     2     3     4     5     6     7     8     9
# lead
#  8             0.1   0.3   0.6   1.1   1.7   2.4   3.2  -1.9 -11.0
#  7             0.2   0.5   1.1   1.9   2.9   4.0  -2.8 -13.8 -23.3
#  6             0.3   1.1   2.1   3.4   4.9  -4.3 -17.3 -27.2 -35.3
#  5             0.8   2.2   4.0   6.0  -6.2 -21.7 -31.4 -38.9 -44.7
#  4             1.8   4.4   7.3  -9.0 -26.9 -35.9 -42.0 -46.7 -50.1
#  3             4.0   8.6 -12.6 -33.1 -39.9 -44.4 -47.4 -49.7 -51.3
#  2             9.0 -17.4 -39.8 -42.8 -44.7 -46.2 -47.3 -48.2 -48.8
#  1           -25.2 -46.0 -42.8 -41.9 -41.9 -42.2 -42.6 -43.0 -43.2
#  0           -33.5 -29.9 -30.5 -31.7 -33.0 -34.0 -34.9 -35.5 -35.9
# -1            -6.1 -10.4 -15.9 -19.9 -22.7 -24.7 -26.3 -27.4 -28.2
# -2            -2.7  -5.1  -8.8 -12.1 -14.8 -16.9 -18.7 -20.0 -21.0
# -3            -1.3  -2.5  -4.7  -7.0  -9.2 -11.0 -12.6 -13.9 -15.0
# -4            -0.5  -1.1  -2.4  -3.9  -5.4  -6.9  -8.2  -9.3 -10.3
# -5            -0.2  -0.5  -1.2  -2.1  -3.1  -4.1  -5.1  -6.0  -6.9
# -6            -0.0  -0.2  -0.6  -1.1  -1.7  -2.4  -3.1  -3.8  -4.4
# -7            -0.0  -0.1  -0.3  -0.6  -0.9  -1.4  -1.8  -2.3  -2.8
# -8            -0.0  -0.0  -0.1  -0.3  -0.5  -0.7  -1.1  -1.4  -1.7
# d_minus_mop_up, middle offense, percentage points (rows: our lead, columns: half-innings D would pitch)
# innings_left     1     2     3     4     5     6     7     8     9
# lead
#  8             0.1   0.2   0.6   1.0   1.6   2.2   2.9  -1.3  -8.7
#  7             0.1   0.5   1.1   1.8   2.7   3.6  -2.1 -11.3 -19.4
#  6             0.3   1.0   2.0   3.2   4.5  -3.4 -14.5 -23.3 -30.5
#  5             0.8   2.1   3.8   5.7  -5.3 -18.8 -27.5 -34.4 -39.9
#  4             1.8   4.2   7.0  -8.0 -24.1 -32.1 -38.0 -42.6 -46.2
#  3             4.0   8.3 -11.6 -30.4 -36.7 -41.0 -44.2 -46.8 -48.9
#  2             8.8 -16.4 -37.1 -40.2 -42.4 -44.1 -45.6 -46.9 -48.0
#  1           -24.7 -43.7 -40.8 -40.5 -40.9 -41.6 -42.6 -43.3 -44.1
#  0           -32.9 -29.2 -30.5 -32.0 -33.5 -35.0 -36.2 -37.2 -38.0
# -1            -6.5 -11.4 -17.1 -21.2 -24.3 -26.7 -28.6 -29.9 -31.1
# -2            -3.3  -6.1 -10.0 -13.6 -16.7 -19.2 -21.3 -22.8 -24.2
# -3            -1.5  -3.0  -5.6  -8.3 -10.8 -13.1 -15.1 -16.7 -18.1
# -4            -0.7  -1.5  -3.1  -4.9  -6.7  -8.6 -10.2 -11.7 -13.0
# -5            -0.3  -0.7  -1.6  -2.7  -4.0  -5.4  -6.7  -7.9  -9.1
# -6            -0.1  -0.3  -0.8  -1.5  -2.3  -3.3  -4.3  -5.2  -6.1
# -7            -0.0  -0.1  -0.4  -0.8  -1.3  -2.0  -2.6  -3.3  -4.0
# -8            -0.0  -0.1  -0.2  -0.4  -0.7  -1.1  -1.6  -2.1  -2.6
# d_minus_mop_up, high offense, percentage points (rows: our lead, columns: half-innings D would pitch)
# innings_left     1     2     3     4     5     6     7     8     9
# lead
#  8             0.1   0.2   0.5   0.9   1.4   2.0   2.6  -0.6  -6.4
#  7             0.1   0.5   1.0   1.7   2.5   3.3  -1.3  -8.7 -15.4
#  6             0.3   1.0   1.9   3.0   4.2  -2.4 -11.7 -19.2 -25.5
#  5             0.8   2.0   3.6   5.3  -4.2 -15.8 -23.4 -29.7 -34.9
#  4             1.8   4.1   6.6  -6.7 -21.1 -28.4 -33.8 -38.3 -42.0
#  3             3.9   8.1 -10.3 -27.4 -33.4 -37.6 -40.8 -43.7 -46.0
#  2             8.8 -15.4 -34.5 -37.6 -39.9 -41.9 -43.6 -45.2 -46.7
#  1           -24.4 -42.3 -39.2 -39.1 -39.9 -40.9 -42.1 -43.2 -44.4
#  0           -32.8 -29.1 -30.4 -32.2 -34.0 -35.6 -37.1 -38.5 -39.8
# -1            -6.8 -11.8 -18.1 -22.4 -25.6 -28.3 -30.5 -32.2 -33.8
# -2            -3.3  -6.5 -11.1 -14.9 -18.3 -21.2 -23.7 -25.6 -27.4
# -3            -1.7  -3.5  -6.4  -9.5 -12.4 -15.1 -17.5 -19.5 -21.3
# -4            -0.8  -1.6  -3.6  -5.8  -8.1 -10.3 -12.5 -14.3 -16.0
# -5            -0.2  -0.8  -2.1  -3.5  -5.2  -6.9  -8.6 -10.2 -11.7
# -6            -0.2  -0.6  -1.2  -2.1  -3.2  -4.4  -5.8  -7.0  -8.3
# -7            -0.2  -0.3  -0.7  -1.2  -1.9  -2.8  -3.8  -4.7  -5.7
# -8            -0.0  -0.1  -0.4  -0.7  -1.1  -1.7  -2.4  -3.1  -3.8
####
# The black line in Figure 3 is the rule: a lead greater than the half-innings left. Above it D cannot lose, and he beats the mop-up reliever by up to 9 points (a 2-run lead with one half-inning left). On and below it he is worse in every state, for every offense (or level, in hopeless ones), and by 24 to 46 points with a tie or a one-run lead. Two things I did not expect: the gain above the line is small when the lead is very large (0.1 to about 1 point, since the mop-up reliever rarely blows those either), and the loss just below the line is enormous, because D turns the lead into a tie or a deficit with certainty.
# In blowout losses (trailing by 5 or more) he costs 0 to 12 points depending on how many half-innings he pitches, and more for a better offense, since a better offense has more of a comeback for him to give away.

fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), sharey=True)
limit = 100 * grid['d_minus_mop_up'].abs().max()
for ax, tier in zip(axes, tiers):
    table = 100 * grid[grid['tier'] == tier].pivot(index='lead', columns='innings_left', values='d_minus_mop_up').sort_index(ascending=False)
    image = ax.imshow(table.values, cmap=diverging_cmap, vmin=-limit, vmax=limit, aspect='auto')
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            value = table.values[r, c]
            if abs(value) >= 0.05:
                ax.text(c, r, f'{value:.1f}', ha='center', va='center', fontsize=6, color='white' if abs(value) > 0.6 * limit else ink)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    ax.set_title(f'{tier} offense', color=ink, fontsize=9)
    ax.set_xlabel('half-innings D pitches (n)')
    # The line above which D cannot lose: lead greater than n
    ax.plot(np.arange(table.shape[1]) - 0.5, [list(table.index).index(min(n + 1, 8)) + 0.5 if n + 1 <= 8 else -0.5 for n in table.columns], drawstyle='steps-post', color=ink, lw=1.2)
axes[0].set_ylabel('our lead when D enters')
colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
colorbar.set_label('D minus mop-up reliever, win probability (points)', color=ink_secondary)
fig.suptitle('Figure 3. Where D beats a mop-up reliever: above the black line he cannot lose', x=0.01, ha='left', fontsize=10, color=ink)
fig.savefig(fig_dir / 'figure3_d_minus_mop_up_grid.png', dpi=200, bbox_inches='tight')
plt.close(fig)


# Step 10 The offense threshold: the states where D needs our offense to score
# With a lead of L and n half-innings to pitch, D's side finishes on L + (our runs) - n. When L is at most n, we need our offense to score at least n - L + 1 to win outright (n - L ties it). This is where how good our offense is should matter.
threshold_rows = []
for tier in tiers:
    for needed in [0, 1, 2]:
        for n in range(1, 9):
            lead = n - needed
            row = grid[(grid['tier'] == tier) & (grid['innings_left'] == n) & (grid['lead'] == lead)].iloc[0]
            threshold_rows.append([tier, needed, n, lead, row['wp_d'], row['wp_mop_up'], row['d_minus_mop_up']])
threshold = pd.DataFrame(threshold_rows, columns=['tier', 'runs_short', 'innings_left', 'lead', 'wp_d', 'wp_mop_up', 'd_minus_mop_up'])
print((100 * threshold.pivot_table(index=['runs_short', 'tier'], columns='innings_left', values='d_minus_mop_up')).round(1).loc[[(k, t) for k in [0, 1, 2] for t in tiers]].to_string())
print('D win probability itself when the lead equals the innings left (runs_short 0):')
print((100 * threshold[threshold['runs_short'] == 0].pivot(index='tier', columns='innings_left', values='wp_d')).round(1).loc[tiers].to_string())
####
# innings_left          1     2     3     4     5     6     7     8
# runs_short tier
# 0          low    -25.2 -17.4 -12.6  -9.0  -6.2  -4.3  -2.8  -1.9
#            middle -24.7 -16.4 -11.6  -8.0  -5.3  -3.4  -2.1  -1.3
#            high   -24.4 -15.4 -10.3  -6.7  -4.2  -2.4  -1.3  -0.6
# 1          low    -33.5 -46.0 -39.8 -33.1 -26.9 -21.7 -17.3 -13.8
#            middle -32.9 -43.7 -37.1 -30.4 -24.1 -18.8 -14.5 -11.3
#            high   -32.8 -42.3 -34.5 -27.4 -21.1 -15.8 -11.7  -8.7
# 2          low     -6.1 -29.9 -42.8 -42.8 -39.9 -35.9 -31.4 -27.2
#            middle  -6.5 -29.2 -40.8 -40.2 -36.7 -32.1 -27.5 -23.3
#            high    -6.8 -29.1 -39.2 -37.6 -33.4 -28.4 -23.4 -19.2
# D win probability itself when the lead equals the innings left (runs_short 0):
# innings_left     1     2     3     4     5     6     7     8
# tier
# low           55.8  66.2  74.4  80.8  85.8  89.3  92.0  94.0
# middle        56.5  67.8  76.0  82.4  87.2  90.7  93.2  95.0
# high          57.0  69.2  77.9  84.2  88.9  92.2  94.5  96.1
####
# This is where the offense threshold from the interview lives. When D needs our offense to score, a better offense narrows his deficit to the mop-up reliever by up to 8 points (high against low tier, more when he has more innings to pitch), but never closes it: there is no offense level in this data at which using D with a lead of n or less is the right call. The threshold question therefore has a simpler answer than I expected. Whether to use D does not depend on the offense at all (only on whether the lead is greater than the innings left); the offense only changes how often that state comes up (Step 11).

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, needed in zip(axes, [0, 1, 2]):
    for tier in tiers:
        t = threshold[(threshold['tier'] == tier) & (threshold['runs_short'] == needed)]
        ax.plot(t['innings_left'], 100 * t['d_minus_mop_up'], 'o-', color=tier_colors[tier], lw=2, ms=4)
        ax.annotate(f'{tier}', (t['innings_left'].iloc[-1], 100 * t['d_minus_mop_up'].iloc[-1]), xytext=(4, {'low': -7, 'middle': 0, 'high': 7}[tier] if needed == 0 else 0), textcoords='offset points', va='center', fontsize=7, color=ink_secondary)
    ax.axhline(0, color=ink_muted, lw=1)
    ax.set_title(f'lead = n - {needed}' if needed else 'lead = n (D can at worst tie)', color=ink, fontsize=9)
    ax.set_xlabel('half-innings D pitches (n)')
    ax.set_xlim(0.5, 9.2)
axes[0].set_ylabel('D minus mop-up reliever (points)')
fig.suptitle('Figure 4. When D needs our offense to score he is always worse than a mop-up reliever; a better offense narrows the gap by up to 8 points', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure4_offense_threshold.png', dpi=200)
plt.close(fig)


# Step 11 First count of how often D could lock up a win (a preview of Phase 2)
# For every team-game, find the first fielding half-inning where our lead is greater than the half-innings left (L > n). From there D wins with certainty, whatever our offense does.
# Every game a team lost after reaching that state is a game D would have won. Every inning from that point on is an inning the rest of the staff did not have to pitch.
# A second, costlier use: trailing by 5 or more at the start of a fielding half-inning from the 6th on (blowout losses). Counted here, priced in Phase 2.
lock_rows = []
for (year, game, team), s in states.groupby(['year', 'game', 'fielding_team'], sort=False):
    s = s.sort_values('inning')
    eligible = s[s['fielding_lead'] > 10 - s['inning']]
    blowout_innings = ((s['inning'] >= 6) & (s['fielding_lead'] <= -5)).sum()
    if len(eligible):
        first = eligible.iloc[0]
        lock_rows.append([year, team, first['fielding_tier'], True, 10 - first['inning'], first['fielding_team_won'], blowout_innings])
    else:
        lock_rows.append([year, team, s['fielding_tier'].iloc[0], False, 0, s['fielding_team_won'].iloc[0], blowout_innings])
locks = pd.DataFrame(lock_rows, columns=['year', 'team', 'tier', 'eligible', 'd_half_innings', 'won', 'blowout_half_innings'])
locks['blown'] = locks['eligible'] & ~locks['won']
per_season = locks.groupby(['year', 'team']).agg(tier=('tier', 'first'), eligible_games=('eligible', 'sum'), d_half_innings=('d_half_innings', 'sum'), blown_leads=('blown', 'sum'), blowout_half_innings=('blowout_half_innings', 'sum')).reset_index()
per_season = per_season.merge(team_seasons[['year', 'team', 'run_differential_per_game']], on=['year', 'team'])
per_season.to_csv(results_dir / 'phase1_lock_counts.csv', index=False)
print('per team-season, mean (and range):')
for column in ['eligible_games', 'd_half_innings', 'blown_leads', 'blowout_half_innings']:
    print(f'  {column}: {per_season[column].mean():.1f} ({per_season[column].min()} to {per_season[column].max()})')
print(per_season.groupby('tier')[['eligible_games', 'd_half_innings', 'blown_leads', 'blowout_half_innings']].mean().loc[tiers].round(1).to_string())
print('real win rate once L > n was reached:', round(locks.loc[locks['eligible'], 'won'].mean(), 4))
for column in ['eligible_games', 'd_half_innings', 'blown_leads']:
    print(f'correlation with run differential per game, {column}: {per_season[column].corr(per_season["run_differential_per_game"]):.2f}')
####
# per team-season, mean (and range):
#   eligible_games: 62.6 (30 to 96)
#   d_half_innings: 165.2 (60 to 276)
#   blown_leads: 2.5 (0 to 10)
#   blowout_half_innings: 56.1 (10 to 139)
#         eligible_games  d_half_innings  blown_leads  blowout_half_innings
# tier
# low               52.3           130.0          2.6                  68.4
# middle            62.4           164.1          2.7                  57.0
# high              73.4           202.6          2.2                  42.7
# real win rate once L > n was reached: 0.9601
# correlation with run differential per game, eligible_games: 0.92
# correlation with run differential per game, d_half_innings: 0.91
# correlation with run differential per game, blown_leads: -0.17
####
# A team reaches a lead greater than the innings left in about 63 games a season (52 for the low-offense third, 73 for the high), and D could pitch about 165 innings from those points on: roughly a full starter's workload taken off the rest of the staff, at no risk to the result.
# Real teams went on to lose 4.0 percent of those games, 2.5 per team-season (0 to 10; 2025 Arizona had the 10, checked game by game). D would have won every one. So before counting any value from rest, D adds about 2.5 wins a season against the bullpens teams actually used, and a bullpen that blows more leads gains more from him. Blown leads barely track run differential (-0.17), so this is not just a good-team or bad-team effect.
# This turns my interview answer around. D does not belong only in garbage time: he is the best possible closer for any lead of 2 or more in the 9th (3 or more in the 8th, and so on), and the roster spot he takes is the 13th pitcher's. Phase 2 prices the blowout-loss use (about 56 half-innings a season trailing by 5 or more from the 6th on), the rest the 165 innings buys, and the comparison with the 13th pitcher.
# Caveat: this counts 2022 to 2025 in-sample, and assumes everything before D enters is unchanged. It also scores each blown game as a full win for D, which is exact under the deterministic rule.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for tier in tiers:
    s = per_season[per_season['tier'] == tier]
    axes[0].scatter(s['run_differential_per_game'], s['d_half_innings'], s=18, color=tier_colors[tier], edgecolor=surface, linewidth=0.8, label=f'{tier} offense')
    axes[1].scatter(s['run_differential_per_game'], s['blown_leads'], s=18, color=tier_colors[tier], edgecolor=surface, linewidth=0.8)
axes[0].set_xlabel('run differential per game')
axes[0].set_ylabel('half-innings D could pitch at no risk')
axes[1].set_xlabel('run differential per game')
axes[1].set_ylabel('losses after leading by more than n')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Figure 5. Innings D could absorb, and losses he would have prevented, per team-season', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure5_lock_counts.png', dpi=200)
plt.close(fig)


# Step 12 Save what Phase 2 needs: the run distributions and the offense tiers
import json
saved_pmfs = {'tier_inning_pmf': {tier: {str(i): list(tier_inning_pmf[tier][i]) for i in range(1, 10)} for tier in tiers}, 'league_inning_pmf': {str(i): list(league_inning_pmf[i]) for i in range(1, 10)}, 'mop_up_pmf': list(mop_up_pmf), 'average_late_pmf': list(average_late_pmf)}
with open(results_dir / 'phase1_pmfs.json', 'w') as f:
    json.dump(saved_pmfs, f)
team_seasons.to_csv(results_dir / 'phase1_team_seasons.csv', index=False)
