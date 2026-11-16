#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:53:56 2026

@author: reyer
"""

# How strong should the prior be? Empirical Bayes against a backtest, on pitcher strikeout rate
# An earlier strikeout projection of mine pulled each pitcher's K% toward the league average with
# a constant k = 200 batters faced, chosen by backtest:
#     shrunk K% = (K% x TBF + league K% x k) / (TBF + k)
# That formula is the posterior mean of a beta-binomial model whose Beta prior has mean equal
# to the league K% and strength (alpha + beta) equal to k. So k is a prior hyperparameter, and
# there are two standard ways to set it:
#   1. empirical Bayes: maximize the marginal (beta-binomial) likelihood of one season's counts
#   2. backtest: choose the k that best predicts the NEXT season, out of sample
# They answer different questions. Empirical Bayes measures how spread out true talent is within
# a season. The backtest also pays for talent changing between seasons, so it should want a
# larger k. This script measures both on the same data, one season to the next.
# Data (in data/, not included in the repository): k_2026.csv, a FanGraphs pitcher leaderboard
# export for 2021 to 2025 (columns Season, Name, PlayerId as the MLBAM id, Team, K%, TBF), and
# player_positions.csv, each player's primary position from the MLB Stats API, used to remove
# position players who pitched.
# v2 (15 November 2026): file header, data paths relative to this file, no period after step
# numbers, longer variable names, and Figure 1. The numbers are the same as v1.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# Rockies colors for the figure
PURPLE, LAVENDER, GRAY, SILVER = '#33006F', '#A88FD6', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Load pitcher seasons and recover strikeout counts
k_data = pd.read_csv(os.path.join(DATA, 'k_2026.csv'), encoding='utf-8-sig')
positions = pd.read_csv(os.path.join(DATA, 'player_positions.csv'))
k_data = k_data.merge(positions[['PlayerId', 'position_type']], on='PlayerId', how='left')
k_data = k_data[k_data['position_type'] == 'Pitcher'].copy()
k_data['K'] = np.round(k_data['K%'] * k_data['TBF']).astype(int)
k_data = k_data[k_data['TBF'] >= 1]
print('pitcher seasons', len(k_data), 'seasons', sorted(k_data['Season'].unique()))

####
# pitcher seasons 4062 seasons [np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024), np.int64(2025)]
####

# Step 2 Empirical Bayes: beta-binomial marginal likelihood, one season at a time
# Parameters: mu (prior mean) on the logit scale and k (prior strength) on the log scale.
def neg_marginal_loglik(theta, K, n):
    mu = 1 / (1 + np.exp(-theta[0]))
    k = np.exp(theta[1])
    a, b = mu * k, (1 - mu) * k
    ll = gammaln(n + 1) - gammaln(K + 1) - gammaln(n - K + 1) + betaln(K + a, n - K + b) - betaln(a, b)
    return -ll.sum()

eb = {}
for season, season_data in k_data.groupby('Season'):
    fit = minimize(neg_marginal_loglik, x0=[np.log(0.22 / 0.78), np.log(100)], args=(season_data['K'].to_numpy(), season_data['TBF'].to_numpy()), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    mu = 1 / (1 + np.exp(-fit.x[0])); k = np.exp(fit.x[1])
    eb[season] = k
    sd_talent = np.sqrt(mu * (1 - mu) / (k + 1))
    print(f'{season}: empirical Bayes prior mean {mu:.4f}, prior strength k {k:.0f} batters faced, spread of true talent (SD) {sd_talent:.4f}, pitchers {len(season_data)}')

####
# 2021: empirical Bayes prior mean 0.2264, prior strength k 68 batters faced, spread of true talent (SD) 0.0503, pitchers 849
# 2022: empirical Bayes prior mean 0.2210, prior strength k 70 batters faced, spread of true talent (SD) 0.0494, pitchers 806
# 2023: empirical Bayes prior mean 0.2238, prior strength k 76 batters faced, spread of true talent (SD) 0.0475, pitchers 803
# 2024: empirical Bayes prior mean 0.2231, prior strength k 82 batters faced, spread of true talent (SD) 0.0458, pitchers 802
# 2025: empirical Bayes prior mean 0.2209, prior strength k 88 batters faced, spread of true talent (SD) 0.0440, pitchers 802
#
# Talent is spread with an SD of about 4.4 to 5.0 points of K%, and the prior is worth
# 70 to 90 batters faced. The talent spread narrows a little each season.
####

# Step 3 Backtest: shrink season t toward season t's league K%, score against season t+1
# Score: RMSE weighted by season t+1 batters faced, over pitchers who pitched in both seasons.
def weighted_rmse(pred, actual, w):
    return float(np.sqrt(np.average((pred - actual) ** 2, weights=w)))

grid = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700]
results = []
for season in sorted(k_data['Season'].unique())[:-1]:
    now = k_data[k_data['Season'] == season][['PlayerId', 'K', 'TBF', 'K%']]
    nxt = k_data[k_data['Season'] == season + 1][['PlayerId', 'K%', 'TBF']]
    both = now.merge(nxt, on='PlayerId', suffixes=('', '_next'))
    league = now['K'].sum() / now['TBF'].sum()
    for k in grid + [eb[season]]:
        pred = (both['K'] + league * k) / (both['TBF'] + k)
        results.append({'season': season, 'k': k, 'is_eb': k == eb[season], 'rmse': weighted_rmse(pred, both['K%_next'], both['TBF_next']), 'pitchers': len(both)})
results = pd.DataFrame(results)
for season, season_results in results.groupby('season'):
    best = season_results[~season_results['is_eb']].loc[season_results[~season_results['is_eb']]['rmse'].idxmin()]
    ebrow = season_results[season_results['is_eb']].iloc[0]
    print(f'{season} -> {season + 1}: backtest best k {best["k"]:.0f} (RMSE {best["rmse"]:.5f}); empirical Bayes k {ebrow["k"]:.0f} (RMSE {ebrow["rmse"]:.5f}); no shrinkage RMSE {season_results[season_results["k"] == 0]["rmse"].iloc[0]:.5f}; pitchers {int(best["pitchers"])}')

####
# 2021 -> 2022: backtest best k 150 (RMSE 0.04499); empirical Bayes k 68 (RMSE 0.04600); no shrinkage RMSE 0.05796; pitchers 598
# 2022 -> 2023: backtest best k 125 (RMSE 0.04145); empirical Bayes k 70 (RMSE 0.04191); no shrinkage RMSE 0.05063; pitchers 586
# 2023 -> 2024: backtest best k 200 (RMSE 0.04090); empirical Bayes k 76 (RMSE 0.04193); no shrinkage RMSE 0.05299; pitchers 593
# 2024 -> 2025: backtest best k 150 (RMSE 0.04055); empirical Bayes k 82 (RMSE 0.04119); no shrinkage RMSE 0.05245; pitchers 590
####

# Step 4 Pooled over the four season pairs
pooled = results[~results['is_eb']].groupby('k')['rmse'].mean()
print('mean RMSE over the four season pairs, by k:')
print(pooled.round(5).to_string())
eb_mean = results[results['is_eb']]['rmse'].mean()
print(f'empirical Bayes k (season by season) mean RMSE {eb_mean:.5f}; mean empirical Bayes k {np.mean(list(eb.values())):.0f}')

####
# mean RMSE over the four season pairs, by k:
# k
# 0.0      0.05351
# 25.0     0.04551
# 50.0     0.04362
# 75.0     0.04274
# 100.0    0.04229
# 125.0    0.04207
# 150.0    0.04199
# 200.0    0.04207
# 250.0    0.04231
# 300.0    0.04262
# 400.0    0.04331
# 500.0    0.04398
# 700.0    0.04515
# empirical Bayes k (season by season) mean RMSE 0.04276; mean empirical Bayes k 77
#
# Shrinkage is the big gain: no shrinkage 0.05351, best k (150) 0.04199. Empirical Bayes,
# fit only to the current season, gets 93 percent of that gain (0.04276) but picks a prior about
# half as strong as the forecast wants (77 against 125 to 200). The difference is talent change
# between seasons, which a single-season marginal likelihood cannot see. A model with talent that
# drifts from season to season (a state-space model, fit by its marginal likelihood across seasons)
# would estimate both at once; the backtest k stands in for it.
####

# Figure 1 Forecast error by prior strength, with the empirical Bayes choices marked
fig, ax = plt.subplots(figsize=(12, 5.4))
eb_values = list(eb.values())
ax.axvspan(min(eb_values), max(eb_values), color=LAVENDER, alpha=0.35, lw=0)
ax.text(np.mean(eb_values), 0.0532, f'empirical Bayes,\none season at a time\n(k = {min(eb_values):.0f} to {max(eb_values):.0f})', ha='center', va='top', fontsize=11, color='#333333')
for season, season_results in results[~results['is_eb']].groupby('season'):
    ax.plot(season_results['k'], season_results['rmse'], color=SILVER, lw=1.5, zorder=1)
ax.plot([], [], color=SILVER, lw=1.5, label='each season pair')
ax.plot(pooled.index, pooled.values, 'o-', color=PURPLE, lw=2.5, ms=7, zorder=3, label='mean over the four season pairs')
best_k = pooled.idxmin()
ax.annotate(f'best k = {best_k:.0f}\n(error {pooled.min():.5f})', xy=(best_k, pooled.min()), xytext=(best_k + 120, pooled.min() - 0.0025), fontsize=11, arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8))
ax.plot(np.mean(eb_values), eb_mean, 'D', color=GRAY, ms=9, zorder=4, label=f'empirical Bayes k, mean error {eb_mean:.5f}')
ax.annotate(f'no shrinkage\n(error {pooled.loc[0]:.5f})', xy=(0, pooled.loc[0]), xytext=(60, pooled.loc[0] + 0.0012), fontsize=11, arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8))
ax.set_xlabel('prior strength k (batters faced added at the league rate)')
ax.set_ylabel('next-season K% error (RMSE)')
ax.set_ylim(0.0385, 0.0595)
ax.legend(frameon=False, loc='upper right', fontsize=11)
ax.set_title('Pitcher strikeout rate: prior strength chosen by backtest and by empirical Bayes, 2021 to 2025', loc='left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'prior_strength_eb_vs_backtest_v2_fig1.png'))
plt.close(fig)
