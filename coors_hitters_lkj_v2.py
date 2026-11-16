#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:43:03 2026

@author: reyer
"""

# How much of a Rockies hitter's line travels? A hierarchical model with an LKJ prior
# Question: Rockies hitters strike out less and get more hits on balls in play at home. How big
# is the Coors gap, does it differ from hitter to hitter, and is it related to how good the hitter
# is on the road? For forecasting, the useful version is: given one season of home and road
# results, what should we expect from this hitter on the road next season?
#
# The model gives every Rockies hitter-season four random effects:
#   1. road strikeout talent (logit K per PA)
#   2. Coors strikeout gap (home minus road, logit)
#   3. road BABIP talent (logit hits per ball in play)
#   4. Coors BABIP gap (home minus road, logit)
# The four effects are drawn from a multivariate normal with SDs tau and correlation matrix Omega.
# Omega gets the Lewandowski-Kurowicka-Joe (LKJ) prior, density proportional to det(Omega)^(eta - 1).
# With eta = 2 it pulls every correlation gently toward 0 (the SD of one correlation under the
# prior is 1/sqrt(2 eta + K - 1) = 1/sqrt(7) = 0.378 for K = 4), and it guarantees that Omega is a
# valid (positive definite) correlation matrix, which separate priors on six correlations would not.
# The SDs and the correlations get separate priors (the separation strategy):
#     Sigma = diag(tau) Omega diag(tau)
# and the model is written non-centered, u = diag(tau) L z with Omega = L L', z ~ N(0, 1).
#
# "Home minus road" for Rockies hitters is the Coors effect plus ordinary home advantage plus any
# road hangover, so Step 2 prints the other 29 teams' home-road gap for comparison.
# Data: MLB Stats API home and road splits for every Rockies hitter, 2015 to 2026 (no 2020).
# v2 (15 November 2026): file header, no period after step numbers, longer variable names, and
# Figures 1, 3 and 4 (the old Figure 1 is now Figure 2, in Rockies colors). The numbers are the same as v1.

import json
import os
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from cmdstanpy import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'splits')
os.makedirs(DATA, exist_ok=True)
SEASONS = [s for s in range(2015, 2027) if s != 2020]
ETA = 2.0
SEED = 20261115
np.random.seed(SEED)

# Rockies colors for the figures: purple for Coors and the posterior, gray for the road and the prior
PURPLE, LAVENDER, GRAY, SILVER = '#33006F', '#A88FD6', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Download (or load cached) home and road splits
def get_json(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r:
            open(path, 'wb').write(r.read())
        time.sleep(0.5)
    return json.load(open(path))

player_rows, team_rows, pitch_rows = [], [], []
for season in SEASONS:
    splits_json = get_json(f'https://statsapi.mlb.com/api/v1/stats?stats=statSplits&group=hitting&sitCodes=h,a&season={season}&teamId=115&playerPool=ALL&limit=1000', os.path.join(DATA, f'col_hitters_{season}.json'))
    for split in splits_json['stats'][0]['splits']:
        stat = split['stat']
        player_rows.append({'season': season, 'player_id': split['player']['id'], 'name': split['player']['fullName'],
                            'pos': split.get('position', {}).get('abbreviation', ''), 'site': split['split']['code'],
                            'pa': stat['plateAppearances'], 'ab': stat['atBats'], 'h': stat['hits'], 'hr': stat['homeRuns'],
                            'so': stat['strikeOuts'], 'sf': stat['sacFlies']})
    splits_json = get_json(f'https://statsapi.mlb.com/api/v1/teams/stats?season={season}&sportIds=1&stats=statSplits&group=hitting&sitCodes=h,a&limit=100', os.path.join(DATA, f'teams_{season}.json'))
    for split in splits_json['stats'][0]['splits']:
        stat = split['stat']
        team_rows.append({'season': season, 'team_id': split['team']['id'], 'site': split['split']['code'], 'pa': stat['plateAppearances'],
                          'ab': stat['atBats'], 'h': stat['hits'], 'hr': stat['homeRuns'], 'so': stat['strikeOuts'], 'sf': stat['sacFlies']})
    splits_json = get_json(f'https://statsapi.mlb.com/api/v1/teams/stats?season={season}&sportIds=1&stats=statSplits&group=pitching&sitCodes=h,a&limit=100', os.path.join(DATA, f'teams_pitching_{season}.json'))
    for split in splits_json['stats'][0]['splits']:
        stat = split['stat']
        pitch_rows.append({'season': season, 'team_id': split['team']['id'], 'site': split['split']['code'], 'pa': stat['battersFaced'],
                           'ab': stat['atBats'], 'h': stat['hits'], 'hr': stat['homeRuns'], 'so': stat['strikeOuts'], 'sf': stat['sacFlies']})
players = pd.DataFrame(player_rows)
teams = pd.DataFrame(team_rows)
pitching = pd.DataFrame(pitch_rows)
for df in (players, teams, pitching):
    df['bip'] = df['ab'] - df['so'] - df['hr'] + df['sf']
    df['hb'] = df['h'] - df['hr']
print('player split rows', len(players), 'team split rows', len(teams), 'team pitching split rows', len(pitching))

# Check: the player splits add up to the team totals, season by season
check = players.groupby(['season', 'site'])['pa'].sum().unstack()
team_col = teams[teams['team_id'] == 115].groupby(['season', 'site'])['pa'].sum().unstack()
print('seasons where Rockies player PA differ from the team total:', int((check != team_col).any(axis=1).sum()))

####
# player split rows 806 team split rows 660 team pitching split rows 660
# seasons where Rockies player PA differ from the team total: 0
####

# Step 2 Hitter-seasons, and the home-road gap for the Rockies against the other 29 teams
# Pitchers are dropped (they batted in the National League until 2022).
hitters = players[players['pos'] != 'P']
wide = hitters.pivot_table(index=['season', 'player_id', 'name'], columns='site', values=['pa', 'so', 'bip', 'hb'],
                           aggfunc='sum', fill_value=0)
wide.columns = [f'{a}_{b}' for a, b in wide.columns]
hitter_seasons = wide.reset_index()
hitter_seasons = hitter_seasons[(hitter_seasons['pa_h'] + hitter_seasons['pa_a']) > 0].reset_index(drop=True)
hitter_seasons[['pa_h', 'pa_a', 'so_h', 'so_a', 'bip_h', 'bip_a', 'hb_h', 'hb_a']] = hitter_seasons[['pa_h', 'pa_a', 'so_h', 'so_a', 'bip_h', 'bip_a', 'hb_h', 'hb_a']].astype(int)
assert (hitter_seasons['bip_h'] >= hitter_seasons['hb_h']).all() and (hitter_seasons['bip_a'] >= hitter_seasons['hb_a']).all() and (hitter_seasons['bip_h'] >= 0).all() and (hitter_seasons['bip_a'] >= 0).all()
print('hitter-seasons', len(hitter_seasons), 'hitters', hitter_seasons['player_id'].nunique(), 'PA home', hitter_seasons['pa_h'].sum(), 'PA road', hitter_seasons['pa_a'].sum())

team_splits = teams.groupby(['season', 'team_id', 'site'])[['pa', 'so', 'bip', 'hb']].sum().unstack('site')
team_k_gap = team_splits['so']['h'] / team_splits['pa']['h'] - team_splits['so']['a'] / team_splits['pa']['a']
team_b_gap = team_splits['hb']['h'] / team_splits['bip']['h'] - team_splits['hb']['a'] / team_splits['bip']['a']
gaps = pd.DataFrame({'k_gap': team_k_gap, 'babip_gap': team_b_gap}).reset_index()
col_hitters = gaps[gaps['team_id'] == 115].set_index('season')
other_hitters = gaps[gaps['team_id'] != 115].groupby('season')[['k_gap', 'babip_gap']].mean()
print('home minus road, team level (K% and BABIP):')
print(pd.DataFrame({'COL K%': col_hitters['k_gap'], 'other 29 K%': other_hitters['k_gap'], 'COL BABIP': col_hitters['babip_gap'],
                    'other 29 BABIP': other_hitters['babip_gap']}).round(3).to_string())
print(f'mean over seasons: COL K% {col_hitters["k_gap"].mean():+.3f}, other teams {other_hitters["k_gap"].mean():+.3f}; COL BABIP {col_hitters["babip_gap"].mean():+.3f}, other teams {other_hitters["babip_gap"].mean():+.3f}')

# The same park from the other dugout: Rockies pitchers at home face visiting hitters at Coors.
# If Coors itself cut strikeouts, Rockies pitchers would also strike out fewer batters at home.
pitching_splits = pitching.groupby(['season', 'team_id', 'site'])[['pa', 'so', 'bip', 'hb']].sum().unstack('site')
pitcher_gaps = pd.DataFrame({'k_gap': pitching_splits['so']['h'] / pitching_splits['pa']['h'] - pitching_splits['so']['a'] / pitching_splits['pa']['a'],
                       'babip_gap': pitching_splits['hb']['h'] / pitching_splits['bip']['h'] - pitching_splits['hb']['a'] / pitching_splits['bip']['a']}).reset_index()
col_pitchers = pitcher_gaps[pitcher_gaps['team_id'] == 115].set_index('season')
other_pitchers = pitcher_gaps[pitcher_gaps['team_id'] != 115].groupby('season')[['k_gap', 'babip_gap']].mean()
print('home minus road for PITCHERS (K% and BABIP allowed):')
print(pd.DataFrame({'COL K%': col_pitchers['k_gap'], 'other 29 K%': other_pitchers['k_gap'], 'COL BABIP': col_pitchers['babip_gap'],
                    'other 29 BABIP': other_pitchers['babip_gap']}).round(3).to_string())
print(f'mean over seasons: COL pitchers K% {col_pitchers["k_gap"].mean():+.3f}, other teams {other_pitchers["k_gap"].mean():+.3f}; COL pitchers BABIP {col_pitchers["babip_gap"].mean():+.3f}, other teams {other_pitchers["babip_gap"].mean():+.3f}')

# Rough split of the Rockies hitters' gap, using means over seasons. The Rockies pitchers' gap is
# the park's effect on visiting hitters plus ordinary home advantage for pitchers (the other 29
# teams' pitchers), so
#     park effect = COL pitchers' gap - other pitchers' gap
#     left over   = COL hitters' gap - other hitters' gap (home advantage) - park effect
# This assumes Coors changes strikeouts and BABIP equally for visiting and Rockies hitters. What is
# left over is specific to Rockies hitters: a road hangover, home familiarity, or both.
park_k = col_pitchers['k_gap'].mean() - other_pitchers['k_gap'].mean()
park_b = col_pitchers['babip_gap'].mean() - other_pitchers['babip_gap'].mean()
print(f'park effect at Coors, from the pitching side: K% {park_k:+.3f}, BABIP {park_b:+.3f}')
print(f'left over for Rockies hitters: K% {col_hitters["k_gap"].mean() - other_hitters["k_gap"].mean() - park_k:+.3f}, BABIP {col_hitters["babip_gap"].mean() - other_hitters["babip_gap"].mean() - park_b:+.3f}')

####
# hitter-seasons 242 hitters 114 PA home 32940 PA road 32360
# home minus road, team level (K% and BABIP):
#         COL K%  other 29 K%  COL BABIP  other 29 BABIP
# season                                                
# 2015    -0.054       -0.008      0.070           0.007
# 2016    -0.053       -0.008      0.046           0.003
# 2017    -0.038       -0.006      0.040           0.009
# 2018    -0.031       -0.007      0.060           0.003
# 2019    -0.058       -0.007      0.056           0.001
# 2021    -0.057       -0.006      0.050           0.006
# 2022    -0.050       -0.008      0.040           0.002
# 2023    -0.036       -0.008      0.033           0.003
# 2024    -0.043       -0.007      0.049          -0.003
# 2025    -0.053       -0.009      0.067           0.002
# 2026    -0.022       -0.011      0.036           0.002
# mean over seasons: COL K% -0.045, other teams -0.008; COL BABIP +0.050, other teams +0.003
# home minus road for PITCHERS (K% and BABIP allowed):
#         COL K%  other 29 K%  COL BABIP  other 29 BABIP
# season                                                
# 2015    -0.011        0.010      0.042          -0.011
# 2016     0.001        0.010      0.035          -0.006
# 2017    -0.017        0.008      0.018          -0.011
# 2018     0.000        0.008      0.037          -0.006
# 2019    -0.008        0.010      0.048          -0.006
# 2021    -0.021        0.009     -0.020          -0.008
# 2022    -0.023        0.011      0.013          -0.004
# 2023    -0.007        0.010      0.041          -0.006
# 2024    -0.009        0.009      0.015           0.001
# 2025    -0.005        0.011      0.035          -0.006
# 2026     0.014        0.011      0.007          -0.004
# mean over seasons: COL pitchers K% -0.008, other teams +0.010; COL pitchers BABIP +0.025, other teams -0.006
# park effect at Coors, from the pitching side: K% -0.018, BABIP +0.031
# left over for Rockies hitters: K% -0.020, BABIP +0.016
#
# Rockies hitters strike out 4.5 percentage points less at home and gain 50 points of BABIP
# (0.050); the other 29 teams' hitters, 0.8 percentage points and 3 points. From the pitching
# side, the park itself is worth about 1.8 percentage points of strikeouts and 31 points of
# BABIP, which leaves about 2.0 percentage points of strikeouts and 16 points of BABIP that
# belong to Rockies hitters (hangover, familiarity, or both).
####

# Figure 1 The Rockies hitters' home-road gap, split into ordinary home advantage, the park, and what is left over
home_adv = (other_hitters['k_gap'].mean(), other_hitters['babip_gap'].mean())
park = (park_k, park_b)
left_over = (col_hitters['k_gap'].mean() - other_hitters['k_gap'].mean() - park_k, col_hitters['babip_gap'].mean() - other_hitters['babip_gap'].mean() - park_b)
fig, axes = plt.subplots(2, 1, figsize=(12, 5.6))
for ax, idx, scale_to, title in ((axes[0], 0, 100, 'Strikeout rate, home minus road (percentage points)'), (axes[1], 1, 1000, 'BABIP, home minus road (points, thousandths)')):
    start = 0.0
    for value, color, label, label_y in ((home_adv[idx], GRAY, 'ordinary home advantage', 1.05), (park[idx], LAVENDER, 'the park (from Rockies pitchers)', 0.42), (left_over[idx], PURPLE, 'Rockies hitters only', 1.05)):
        width = value * scale_to
        ax.barh(0, width, left=start, color=color, edgecolor='white', lw=2, height=0.6)
        ax.text(start + width / 2, label_y, f'{label} {abs(width):.1f}', ha='center', va='bottom', fontsize=11)
        start += width
    total = (col_hitters['k_gap'].mean(), col_hitters['babip_gap'].mean())[idx] * scale_to
    ax.text(start + (0.05 * abs(start) if start > 0 else -0.05 * abs(start)), 0, f'total {total:+.1f}', ha='left' if start > 0 else 'right', va='center', fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(-0.45, 1.5)
    ax.grid(axis='y', visible=False)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_title(title, loc='left')
axes[0].set_xlim(-6.5, 0.6)
axes[1].set_xlim(-5, 62)
fig.suptitle('Rockies hitters at Coors, 2015 to 2026: sources of the home-road gap (team totals, mean over seasons)', x=0.01, ha='left', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_hitters_lkj_v2_fig1.png'))
plt.close(fig)

# Step 3 The Stan model
# Flags let one program fit every variant in the backtest (Step 8):
#   use_home = 0 ignores home results; gap_re = 0 gives every hitter the same Coors gap;
#   use_corr = 0 fixes Omega at the identity (independent effects). When a flag switches a
#   piece off, its parameters are left with only their priors, which the data never touch.
STAN = '''
data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=0> pa_h;
  array[N] int<lower=0> so_h;
  array[N] int<lower=0> pa_a;
  array[N] int<lower=0> so_a;
  array[N] int<lower=0> bip_h;
  array[N] int<lower=0> hb_h;
  array[N] int<lower=0> bip_a;
  array[N] int<lower=0> hb_a;
  int<lower=0, upper=1> use_home;
  int<lower=0, upper=1> gap_re;
  int<lower=0, upper=1> use_corr;
  real<lower=0> eta;
}
parameters {
  vector[S] mu_k;                  // road strikeout level by season (logit)
  vector[S] mu_b;                  // road BABIP level by season (logit)
  real gap_k;                      // average Coors strikeout gap (logit)
  real gap_b;                      // average Coors BABIP gap (logit)
  vector<lower=0>[4] tau;          // SDs of the four hitter effects
  cholesky_factor_corr[4] L;       // Cholesky factor of Omega
  matrix[4, N] z;                  // standard normal draws (non-centered)
}
transformed parameters {
  matrix[4, N] u;                  // rows: K talent, K gap, BABIP talent, BABIP gap
  if (use_corr == 1) {
    u = diag_pre_multiply(tau, L) * z;
  } else {
    u = diag_pre_multiply(tau, identity_matrix(4)) * z;
  }
}
model {
  mu_k ~ normal(-1.27, 0.5);       // logit(0.22)
  mu_b ~ normal(-0.85, 0.5);       // logit(0.30)
  gap_k ~ normal(0, 0.5);
  gap_b ~ normal(0, 0.5);
  tau ~ normal(0, 0.5);            // half-normal, because tau > 0
  L ~ lkj_corr_cholesky(eta);
  to_vector(z) ~ std_normal();
  so_a ~ binomial_logit(pa_a, mu_k[season] + u[1]');
  hb_a ~ binomial_logit(bip_a, mu_b[season] + u[3]');
  if (use_home == 1) {
    so_h ~ binomial_logit(pa_h, mu_k[season] + gap_k + gap_re * u[2]' + u[1]');
    hb_h ~ binomial_logit(bip_h, mu_b[season] + gap_b + gap_re * u[4]' + u[3]');
  }
}
generated quantities {
  corr_matrix[4] Omega = multiply_lower_tri_self_transpose(L);
  vector[N] k_road = inv_logit(mu_k[season] + u[1]');
  vector[N] b_road = inv_logit(mu_b[season] + u[3]');
}
'''
stan_file = os.path.join(HERE, 'coors_hitters_lkj.stan')
if not os.path.exists(stan_file) or open(stan_file).read() != STAN:
    open(stan_file, 'w').write(STAN)
model = CmdStanModel(stan_file=stan_file)

def stan_data(df, use_home=1, gap_re=1, use_corr=1):
    seasons = sorted(df['season'].unique())
    idx = {s: i + 1 for i, s in enumerate(seasons)}
    return {'N': len(df), 'S': len(seasons), 'season': [idx[s] for s in df['season']],
            'pa_h': df['pa_h'].tolist(), 'so_h': df['so_h'].tolist(), 'pa_a': df['pa_a'].tolist(), 'so_a': df['so_a'].tolist(),
            'bip_h': df['bip_h'].tolist(), 'hb_h': df['hb_h'].tolist(), 'bip_a': df['bip_a'].tolist(), 'hb_a': df['hb_a'].tolist(),
            'use_home': use_home, 'gap_re': gap_re, 'use_corr': use_corr, 'eta': ETA}

# Early in warmup Stan may print 'lkj_corr_cholesky_lpdf: Random variable[2] is 0'. Those are
# rejected proposals while the step size adapts, not a problem with the fit; R-hat, effective
# sample size, and divergences after warmup are what to check.
def fit_model(data, seed=SEED):
    fit = model.sample(data=data, chains=4, iter_warmup=1000, iter_sampling=2000, adapt_delta=0.95, seed=seed, show_progress=False)
    summ = fit.summary()
    rhat = summ['R_hat'].max()
    ess = summ['ESS_bulk'].min()
    div = int(fit.method_variables()['divergent__'].sum())
    return fit, rhat, ess, div

def interval(x):
    return np.percentile(x, 5), np.percentile(x, 95)

EFFECTS = ['K talent', 'K gap', 'BABIP talent', 'BABIP gap']
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]

# Step 4 Test: can the model recover parameters it was simulated from?
# Simulate outcomes with known tau and Omega, using the real PA and ball-in-play counts, fit, and
# check that the 90% intervals cover the true values.
true_tau = np.array([0.30, 0.10, 0.10, 0.06])
true_Omega = np.array([[1.0, -0.4, 0.3, 0.0],
                       [-0.4, 1.0, 0.0, 0.2],
                       [0.3, 0.0, 1.0, -0.3],
                       [0.0, 0.2, -0.3, 1.0]])
assert np.all(np.linalg.eigvalsh(true_Omega) > 0)
sim = hitter_seasons.copy()
rng = np.random.default_rng(SEED)
u_true = rng.multivariate_normal(np.zeros(4), np.diag(true_tau) @ true_Omega @ np.diag(true_tau), size=len(sim))
expit = lambda x: 1 / (1 + np.exp(-x))
sim['so_a'] = rng.binomial(sim['pa_a'], expit(-1.30 + u_true[:, 0]))
sim['so_h'] = rng.binomial(sim['pa_h'], expit(-1.30 - 0.15 + u_true[:, 0] + u_true[:, 1]))
sim['hb_a'] = rng.binomial(sim['bip_a'], expit(-0.90 + u_true[:, 2]))
sim['hb_h'] = rng.binomial(sim['bip_h'], expit(-0.90 + 0.15 + u_true[:, 2] + u_true[:, 3]))
fit_sim, rhat, ess, div = fit_model(stan_data(sim))
print(f'recovery fit: max R-hat {rhat:.3f}, min bulk ESS {ess:.0f}, divergences {div}')
tau_d = fit_sim.stan_variable('tau'); om_d = fit_sim.stan_variable('Omega')
covered = 0
for k in range(4):
    lo, hi = interval(tau_d[:, k]); covered += lo <= true_tau[k] <= hi
    print(f'  tau {EFFECTS[k]:13s} true {true_tau[k]:.2f}  posterior mean {tau_d[:, k].mean():.3f}  90% {lo:.3f} to {hi:.3f}')
for i, j in PAIRS:
    lo, hi = interval(om_d[:, i, j]); covered += lo <= true_Omega[i, j] <= hi
    print(f'  corr {EFFECTS[i]} / {EFFECTS[j]}: true {true_Omega[i, j]:+.1f}  posterior mean {om_d[:, i, j].mean():+.2f}  90% {lo:+.2f} to {hi:+.2f}')
gap_k_sim = fit_sim.stan_variable('gap_k'); gap_b_sim = fit_sim.stan_variable('gap_b')
print(f'  gap_k true -0.15 posterior mean {gap_k_sim.mean():+.3f}; gap_b true +0.15 posterior mean {gap_b_sim.mean():+.3f}')
print(f'true values inside the 90% interval: {covered} of 10')

# Null case: the same simulation with Omega = identity. If the model manufactured a correlation
# between K talent and the K gap (both use the road strikeouts), it would show up here.
u_null = rng.multivariate_normal(np.zeros(4), np.diag(true_tau ** 2), size=len(sim))
sim['so_a'] = rng.binomial(sim['pa_a'], expit(-1.30 + u_null[:, 0]))
sim['so_h'] = rng.binomial(sim['pa_h'], expit(-1.30 - 0.15 + u_null[:, 0] + u_null[:, 1]))
sim['hb_a'] = rng.binomial(sim['bip_a'], expit(-0.90 + u_null[:, 2]))
sim['hb_h'] = rng.binomial(sim['bip_h'], expit(-0.90 + 0.15 + u_null[:, 2] + u_null[:, 3]))
fit_null, rhat, ess, div = fit_model(stan_data(sim))
om_n = fit_null.stan_variable('Omega')
print(f'null fit: max R-hat {rhat:.3f}, divergences {div}')
for i, j in PAIRS:
    lo, hi = interval(om_n[:, i, j])
    print(f'  corr {EFFECTS[i]} / {EFFECTS[j]}: true 0, posterior mean {om_n[:, i, j].mean():+.2f}, 90% {lo:+.2f} to {hi:+.2f}')

####
# recovery fit: max R-hat 1.005, min bulk ESS 763, divergences 0
#   tau K talent      true 0.30  posterior mean 0.312  90% 0.275 to 0.351
#   tau K gap         true 0.10  posterior mean 0.090  90% 0.015 to 0.155
#   tau BABIP talent  true 0.10  posterior mean 0.099  90% 0.064 to 0.135
#   tau BABIP gap     true 0.06  posterior mean 0.049  90% 0.004 to 0.115
#   corr K talent / K gap: true -0.4  posterior mean -0.33  90% -0.68 to +0.11
#   corr K talent / BABIP talent: true +0.3  posterior mean +0.25  90% -0.03 to +0.52
#   corr K talent / BABIP gap: true +0.0  posterior mean +0.08  90% -0.49 to +0.60
#   corr K gap / BABIP talent: true +0.0  posterior mean +0.10  90% -0.43 to +0.58
#   corr K gap / BABIP gap: true +0.2  posterior mean -0.06  90% -0.65 to +0.56
#   corr BABIP talent / BABIP gap: true -0.3  posterior mean -0.16  90% -0.71 to +0.49
#   gap_k true -0.15 posterior mean -0.156; gap_b true +0.15 posterior mean +0.140
# true values inside the 90% interval: 10 of 10
# null fit: max R-hat 1.005, divergences 0
#   corr K talent / K gap: true 0, posterior mean +0.09, 90% -0.39 to +0.58
#   corr K talent / BABIP talent: true 0, posterior mean +0.34, 90% +0.08 to +0.58
#   corr K talent / BABIP gap: true 0, posterior mean -0.19, 90% -0.65 to +0.35
#   corr K gap / BABIP talent: true 0, posterior mean -0.16, 90% -0.65 to +0.41
#   corr K gap / BABIP gap: true 0, posterior mean -0.01, 90% -0.61 to +0.60
#   corr BABIP talent / BABIP gap: true 0, posterior mean -0.18, 90% -0.69 to +0.45
#
# The model recovers what it was simulated from (10 of 10). The correlations involving the
# gaps are weakly identified with 242 hitter-seasons: their intervals are nearly as wide as the
# prior. In the null case the K talent / K gap correlation stays near 0 (+0.09), so the model
# does not manufacture the one correlation found in the real data. One of six null intervals
# excludes 0 (+0.34), about the one-in-ten expected of 90% intervals.
####

# Step 5 Fit the full model to every Rockies hitter-season, 2015 to 2026
fit, rhat, ess, div = fit_model(stan_data(hitter_seasons))
print(f'full fit: max R-hat {rhat:.3f}, min bulk ESS {ess:.0f}, divergences {div}, hitter-seasons {len(hitter_seasons)}')
mu_k = fit.stan_variable('mu_k').mean(axis=1); mu_b = fit.stan_variable('mu_b').mean(axis=1)
gap_k = fit.stan_variable('gap_k'); gap_b = fit.stan_variable('gap_b')
tau_d = fit.stan_variable('tau'); om_d = fit.stan_variable('Omega')
# Convert logit gaps to rate points at the average road level
gap_k_rate = expit(mu_k + gap_k) - expit(mu_k); gap_b_rate = expit(mu_b + gap_b) - expit(mu_b)
print(f'average Coors gap, K per PA: {gap_k_rate.mean():+.4f} (90% {interval(gap_k_rate)[0]:+.4f} to {interval(gap_k_rate)[1]:+.4f}) at road level {expit(mu_k).mean():.3f}')
print(f'average Coors gap, BABIP:   {gap_b_rate.mean():+.4f} (90% {interval(gap_b_rate)[0]:+.4f} to {interval(gap_b_rate)[1]:+.4f}) at road level {expit(mu_b).mean():.3f}')
road_k_level = expit(mu_k).mean(); road_b_level = expit(mu_b).mean()
scale = [road_k_level * (1 - road_k_level), road_k_level * (1 - road_k_level), road_b_level * (1 - road_b_level), road_b_level * (1 - road_b_level)]
for k in range(4):
    lo, hi = interval(tau_d[:, k])
    print(f'SD across hitter-seasons, {EFFECTS[k]:13s}: {tau_d[:, k].mean():.3f} logit (90% {lo:.3f} to {hi:.3f}), about {tau_d[:, k].mean() * scale[k]:.4f} in rate')
prior_lo, prior_hi = 2 * beta_dist.ppf([0.05, 0.95], ETA - 1 + 2, ETA - 1 + 2) - 1
print(f'LKJ(eta={ETA:.0f}) prior for any one correlation, K = 4: SD {1 / np.sqrt(2 * ETA + 4 - 1):.3f}, 90% {prior_lo:+.2f} to {prior_hi:+.2f}')
for i, j in PAIRS:
    lo, hi = interval(om_d[:, i, j])
    print(f'  corr {EFFECTS[i]} / {EFFECTS[j]}: posterior mean {om_d[:, i, j].mean():+.2f}, 90% {lo:+.2f} to {hi:+.2f}, P(> 0) {np.mean(om_d[:, i, j] > 0):.2f}')

####
# full fit: max R-hat 1.003, min bulk ESS 1308, divergences 0, hitter-seasons 242
# average Coors gap, K per PA: -0.0471 (90% -0.0543 to -0.0399) at road level 0.259
# average Coors gap, BABIP:   +0.0501 (90% +0.0426 to +0.0575) at road level 0.286
# SD across hitter-seasons, K talent     : 0.365 logit (90% 0.327 to 0.408), about 0.0701 in rate
# SD across hitter-seasons, K gap        : 0.182 logit (90% 0.129 to 0.235), about 0.0350 in rate
# SD across hitter-seasons, BABIP talent : 0.047 logit (90% 0.005 to 0.090), about 0.0096 in rate
# SD across hitter-seasons, BABIP gap    : 0.038 logit (90% 0.004 to 0.089), about 0.0078 in rate
# LKJ(eta=2) prior for any one correlation, K = 4: SD 0.378, 90% -0.62 to +0.62
#   corr K talent / K gap: posterior mean -0.26, 90% -0.48 to -0.02, P(> 0) 0.04
#   corr K talent / BABIP talent: posterior mean -0.01, 90% -0.47 to +0.43, P(> 0) 0.49
#   corr K talent / BABIP gap: posterior mean -0.06, 90% -0.59 to +0.50, P(> 0) 0.42
#   corr K gap / BABIP talent: posterior mean -0.03, 90% -0.56 to +0.50, P(> 0) 0.46
#   corr K gap / BABIP gap: posterior mean +0.05, 90% -0.54 to +0.62, P(> 0) 0.56
#   corr BABIP talent / BABIP gap: posterior mean -0.06, 90% -0.65 to +0.59, P(> 0) 0.43
#
# The strikeout gap differs from hitter to hitter (SD about 3.5 points per hitter-season) and
# is larger for hitters who strike out more on the road (correlation -0.26). The BABIP gap does
# not: its spread is near zero, so it acts like a property of the park. The other five
# correlations stay where the LKJ(2) prior put them (Figure 2).
####

# Figure 2 Prior and posterior for the six correlations
NICE = {'K talent': 'strikeout talent', 'K gap': 'strikeout gap', 'BABIP talent': 'BABIP talent', 'BABIP gap': 'BABIP gap'}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True)
r = np.linspace(-0.999, 0.999, 400)
prior_pdf = beta_dist.pdf((r + 1) / 2, ETA - 1 + 2, ETA - 1 + 2) / 2
for ax, (i, j) in zip(axes.flat, PAIRS):
    ax.hist(om_d[:, i, j], bins=60, range=(-1, 1), density=True, color=PURPLE, alpha=0.8, label='posterior')
    ax.plot(r, prior_pdf, color=GRAY, lw=2.5, label=f'LKJ({ETA:.0f}) prior')
    ax.axvline(0, color='black', lw=0.8)
    lo, hi = interval(om_d[:, i, j])
    ax.set_title(f'{NICE[EFFECTS[i]]} and {NICE[EFFECTS[j]]}\nmean {om_d[:, i, j].mean():+.2f} (90% {lo:+.2f} to {hi:+.2f})', fontsize=12, loc='left')
    ax.set_yticks([])
    ax.grid(axis='y', visible=False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=12, frameon=False, loc='upper right', ncol=2)
for ax in axes[1]:
    ax.set_xlabel('correlation')
fig.suptitle('LKJ(2) prior and posterior for the six correlations, Rockies hitter-seasons 2015 to 2026', x=0.01, ha='left', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_hitters_lkj_v2_fig2.png'))
plt.close(fig)

# Step 6 Sensitivity to eta: refit at eta = 1 (flat over correlation matrices) and eta = 4
for eta_alt in (1.0, 4.0):
    d_alt = stan_data(hitter_seasons)
    d_alt['eta'] = eta_alt
    f_alt, rh, es, dv = fit_model(d_alt)
    om_alt = f_alt.stan_variable('Omega')
    print(f'eta = {eta_alt:.0f}: max R-hat {rh:.3f}, divergences {dv}')
    for i, j in PAIRS:
        lo, hi = interval(om_alt[:, i, j])
        print(f'  corr {EFFECTS[i]} / {EFFECTS[j]}: posterior mean {om_alt[:, i, j].mean():+.2f}, 90% {lo:+.2f} to {hi:+.2f}')

####
# eta = 1: max R-hat 1.003, divergences 0
#   corr K talent / K gap: posterior mean -0.27, 90% -0.49 to -0.02
#   corr K talent / BABIP talent: posterior mean +0.00, 90% -0.55 to +0.54
#   corr K talent / BABIP gap: posterior mean -0.07, 90% -0.69 to +0.60
#   corr K gap / BABIP talent: posterior mean -0.06, 90% -0.67 to +0.58
#   corr K gap / BABIP gap: posterior mean +0.07, 90% -0.65 to +0.74
#   corr BABIP talent / BABIP gap: posterior mean -0.09, 90% -0.78 to +0.67
# eta = 4: max R-hat 1.005, divergences 0
#   corr K talent / K gap: posterior mean -0.25, 90% -0.45 to -0.01
#   corr K talent / BABIP talent: posterior mean -0.01, 90% -0.39 to +0.37
#   corr K talent / BABIP gap: posterior mean -0.04, 90% -0.48 to +0.42
#   corr K gap / BABIP talent: posterior mean -0.02, 90% -0.45 to +0.41
#   corr K gap / BABIP gap: posterior mean +0.03, 90% -0.45 to +0.50
#   corr BABIP talent / BABIP gap: posterior mean -0.05, 90% -0.53 to +0.45
#
# The one correlation the data support does not depend on eta (-0.27, -0.26, -0.25). The
# others widen at eta = 1 and narrow at eta = 4, because for them the prior is doing the work.
####

# Step 7 What the model does to individual hitters: raw and shrunk Coors gaps, 2026
# The model's gap for a hitter is gap_k + u[2] (logit); shown here in rate points at his road level.
u_d = fit.stan_variable('u')
rows_2026 = hitter_seasons.index[hitter_seasons['season'] == 2026]
season_2026 = sorted(hitter_seasons['season'].unique()).index(2026)
rows = []
for n in rows_2026:
    road_k = expit(fit.stan_variable('mu_k')[:, season_2026] + u_d[:, 0, n]); home_k = expit(fit.stan_variable('mu_k')[:, season_2026] + gap_k + u_d[:, 0, n] + u_d[:, 1, n])
    road_b = expit(fit.stan_variable('mu_b')[:, season_2026] + u_d[:, 2, n]); home_b = expit(fit.stan_variable('mu_b')[:, season_2026] + gap_b + u_d[:, 2, n] + u_d[:, 3, n])
    h = hitter_seasons.loc[n]
    rows.append({'hitter': h['name'], 'PA home': h['pa_h'], 'PA road': h['pa_a'],
                 'raw K gap': h['so_h'] / max(h['pa_h'], 1) - h['so_a'] / max(h['pa_a'], 1),
                 'model K gap': (home_k - road_k).mean(),
                 'raw BABIP gap': h['hb_h'] / max(h['bip_h'], 1) - h['hb_a'] / max(h['bip_a'], 1),
                 'model BABIP gap': (home_b - road_b).mean()})
hitters_2026 = pd.DataFrame(rows).sort_values('PA home', ascending=False).head(10)
print(hitters_2026.round(3).to_string(index=False))

####
#         hitter  PA home  PA road  raw K gap  model K gap  raw BABIP gap  model BABIP gap
#    TJ Rumfield      325      306     -0.008       -0.023         -0.004            0.050
#  Jake McCarthy      305      291     -0.031       -0.034          0.034            0.051
# Hunter Goodman      272      302      0.014       -0.022          0.001            0.047
#  Mickey Moniak      241      195     -0.152       -0.092         -0.005            0.045
# Ezequiel Tovar      241      265      0.012       -0.023         -0.042            0.045
#  Troy Johnston      237      212      0.011       -0.022          0.074            0.053
#    Kyle Karros      234      241     -0.053       -0.047          0.049            0.050
#   Willi Castro      222      255     -0.059       -0.054          0.103            0.052
#   Cole Carrigg      190      182     -0.059       -0.051          0.104            0.052
# Edouard Julien      158      118      0.005       -0.034          0.047            0.049
#
# Every 2026 hitter gets about the same BABIP gap (+0.045 to +0.053), whatever his raw split.
# Strikeout gaps are shrunk but keep their order (Moniak -0.152 raw, -0.092 model).
####

# Figure 3 Raw and model Coors gaps for the ten 2026 hitters with the most home plate appearances
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
order = hitters_2026.iloc[::-1].reset_index(drop=True)
for ax, raw_col, model_col, scale_to, avg, title in ((axes[0], 'raw K gap', 'model K gap', 100, gap_k_rate.mean(), 'Strikeout gap (percentage points)'), (axes[1], 'raw BABIP gap', 'model BABIP gap', 1000, gap_b_rate.mean(), 'BABIP gap (points)')):
    ax.axvline(avg * scale_to, color=LAVENDER, lw=2, ls='--', zorder=0)
    ax.axvline(0, color='black', lw=0.8, zorder=0)
    for y, row in order.iterrows():
        ax.annotate('', xy=(row[model_col] * scale_to, y), xytext=(row[raw_col] * scale_to, y), arrowprops=dict(arrowstyle='->', color=SILVER, lw=2))
        ax.plot(row[raw_col] * scale_to, y, 'o', color=GRAY, ms=9)
        ax.plot(row[model_col] * scale_to, y, 'o', color=PURPLE, ms=9)
    ax.set_title(title, loc='left', fontsize=13)
    ax.grid(axis='y', visible=False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([f"{row['hitter']} ({row['PA home']} home, {row['PA road']} road PA)" for _, row in order.iterrows()], fontsize=11)
axes[0].plot([], [], 'o', color=GRAY, ms=9, label='raw split')
axes[0].plot([], [], 'o', color=PURPLE, ms=9, label='model')
axes[0].plot([], [], color=LAVENDER, lw=2, ls='--', label='model average gap')
axes[0].legend(frameon=False, fontsize=11, loc='lower left')
fig.suptitle('2026 Rockies hitters: raw home-road splits and the model estimates', x=0.01, ha='left', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_hitters_lkj_v2_fig3.png'))
plt.close(fig)

# Step 8 Backtest: does each piece of the model help forecast next season's road line?
# Fit on every season up to t, predict each hitter's road K% and road BABIP in t + 1 from his
# season-t estimate, and score with RMSE weighted by t + 1 road PA (K%) or road balls in play (BABIP).
# The ladder adds one piece at a time:
#   raw     season-t road rate, no model
#   M1      road results only, shrunk (home results ignored)
#   M2      home and road, one Coors gap for everyone (a park factor)
#   M3      home and road, a Coors gap for each hitter, effects independent (Omega = identity)
#   M4      M3 plus the LKJ correlation between all four effects
def weighted_rmse(pred, actual, w):
    return float(np.sqrt(np.average((pred - actual) ** 2, weights=w)))

VARIANTS = {'M1': dict(use_home=0, gap_re=0, use_corr=0), 'M2': dict(use_home=1, gap_re=0, use_corr=0),
            'M3': dict(use_home=1, gap_re=1, use_corr=0), 'M4': dict(use_home=1, gap_re=1, use_corr=1)}
results = []
prior_check = []
for t_season in [2021, 2022, 2023, 2024, 2025]:
    train = hitter_seasons[hitter_seasons['season'] <= t_season].reset_index(drop=True)
    now = train[(train['season'] == t_season) & (train['pa_a'] > 0) & (train['bip_a'] > 0)]
    nxt = hitter_seasons[hitter_seasons['season'] == t_season + 1][['player_id', 'pa_a', 'so_a', 'bip_a', 'hb_a']]
    both = now.reset_index().merge(nxt, on='player_id', suffixes=('', '_next'))
    both = both[(both['pa_a_next'] > 0) & (both['bip_a_next'] > 0)]
    k_next = both['so_a_next'] / both['pa_a_next']; b_next = both['hb_a_next'] / both['bip_a_next']
    preds = {'raw': (both['so_a'] / both['pa_a'], both['hb_a'] / both['bip_a'])}
    for name, flags in VARIANTS.items():
        fit_variant, rh, es, dv = fit_model(stan_data(train, **flags))
        preds[name] = (fit_variant.stan_variable('k_road').mean(axis=0)[both['index']], fit_variant.stan_variable('b_road').mean(axis=0)[both['index']])
        if name == 'M3':
            prior_check.append(fit_variant.stan_variable('Omega')[:, 0, 1])   # Omega is prior-only when use_corr = 0
        print(f'  {t_season}: {name} max R-hat {rh:.3f}, divergences {dv}')
    for name, (pred_k, pred_b) in preds.items():
        results.append({'season': f'{t_season}->{t_season + 1}', 'model': name, 'hitters': len(both),
                        'K% RMSE': weighted_rmse(np.asarray(pred_k), k_next, both['pa_a_next']),
                        'BABIP RMSE': weighted_rmse(np.asarray(pred_b), b_next, both['bip_a_next'])})
results = pd.DataFrame(results)
print(results.pivot(index='season', columns='model', values='K% RMSE').round(5).to_string())
print(results.pivot(index='season', columns='model', values='BABIP RMSE').round(5).to_string())
print('mean over the five season pairs:')
print(results.groupby('model')[['K% RMSE', 'BABIP RMSE']].mean().round(5).to_string())
print('hitters scored per pair:', results.groupby('season')['hitters'].first().to_dict())
prior_corr_draws = np.concatenate(prior_check)
print(f'check on the LKJ formula: SD of one correlation drawn from the LKJ({ETA:.0f}) prior, K = 4: sampled {prior_corr_draws.std():.3f}, formula {1 / np.sqrt(2 * ETA + 4 - 1):.3f}')

####
#   2021: M1 max R-hat 1.004, divergences 0
#   2021: M2 max R-hat 1.004, divergences 0
#   2021: M3 max R-hat 1.004, divergences 0
#   2021: M4 max R-hat 1.003, divergences 0
#   2022: M1 max R-hat 1.004, divergences 0
#   2022: M2 max R-hat 1.005, divergences 0
#   2022: M3 max R-hat 1.008, divergences 0
#   2022: M4 max R-hat 1.008, divergences 0
#   2023: M1 max R-hat 1.004, divergences 0
#   2023: M2 max R-hat 1.003, divergences 0
#   2023: M3 max R-hat 1.006, divergences 0
#   2023: M4 max R-hat 1.004, divergences 0
#   2024: M1 max R-hat 1.003, divergences 0
#   2024: M2 max R-hat 1.004, divergences 0
#   2024: M3 max R-hat 1.007, divergences 0
#   2024: M4 max R-hat 1.003, divergences 0
#   2025: M1 max R-hat 1.005, divergences 0
#   2025: M2 max R-hat 1.005, divergences 0
#   2025: M3 max R-hat 1.003, divergences 0
#   2025: M4 max R-hat 1.004, divergences 0
# model            M1       M2       M3       M4      raw
# season                                                 
# 2021->2022  0.05743  0.04995  0.05145  0.05143  0.06109
# 2022->2023  0.06164  0.05756  0.05811  0.05824  0.06556
# 2023->2024  0.05757  0.06114  0.05755  0.05873  0.06565
# 2024->2025  0.07492  0.05616  0.06122  0.06344  0.08508
# 2025->2026  0.04670  0.04695  0.04679  0.04609  0.05396
# model            M1       M2       M3       M4      raw
# season                                                 
# 2021->2022  0.03780  0.03616  0.03629  0.03659  0.04501
# 2022->2023  0.03653  0.03770  0.03762  0.03733  0.06589
# 2023->2024  0.04434  0.04065  0.04091  0.04115  0.05936
# 2024->2025  0.05883  0.05965  0.05947  0.05912  0.07616
# 2025->2026  0.02824  0.02188  0.02211  0.02244  0.06142
# mean over the five season pairs:
#        K% RMSE  BABIP RMSE
# model                     
# M1     0.05965     0.04115
# M2     0.05435     0.03921
# M3     0.05503     0.03928
# M4     0.05558     0.03933
# raw    0.06627     0.06157
# hitters scored per pair: {'2021->2022': 11, '2022->2023': 14, '2023->2024': 13, '2024->2025': 13, '2025->2026': 11}
# check on the LKJ formula: SD of one correlation drawn from the LKJ(2) prior, K = 4: sampled 0.377, formula 0.378
#
# Shrinkage does most of the work (raw to M1), and home results with one park gap help more
# (M1 to M2). Hitter-specific gaps (M3) and the LKJ correlations (M4) do not improve next
# season's road forecast. Only 62 hitters are scored, so the M2 to M4 differences are noise-
# sized. The richer model describes Coors better; the simpler one forecasts as well.
####

# Figure 4 Backtest error for each rung of the ladder, by season pair and on average
LADDER = {'raw': 'raw\nroad rate', 'M1': 'M1\nroad only,\nshrunk', 'M2': 'M2\n+ one\npark gap', 'M3': 'M3\n+ a gap for\neach hitter', 'M4': 'M4\n+ LKJ\ncorrelations'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, metric, title in ((axes[0], 'K% RMSE', 'Next-season road strikeout rate'), (axes[1], 'BABIP RMSE', 'Next-season road BABIP')):
    for x_pos, model_name in enumerate(LADDER):
        per_pair = results[results['model'] == model_name][metric].to_numpy()
        ax.plot(np.full(len(per_pair), x_pos), per_pair, 'o', color=SILVER, ms=7, zorder=1)
        mean_value = per_pair.mean()
        ax.plot([x_pos - 0.28, x_pos + 0.28], [mean_value, mean_value], color=PURPLE if model_name != 'raw' else GRAY, lw=4, zorder=2)
        ax.text(x_pos, mean_value, f'{mean_value:.5f}\n', ha='center', va='bottom', fontsize=10.5)
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels(LADDER.values(), fontsize=11)
    ax.set_xlim(-0.5, len(LADDER) - 0.5)
    ax.set_title(title, loc='left')
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('error (RMSE, weighted)')
axes[1].plot([], [], 'o', color=SILVER, ms=7, label='one season pair (2021 to 2026)')
axes[1].plot([], [], color=PURPLE, lw=4, label='mean of the five pairs')
axes[1].legend(frameon=False, fontsize=11, loc='upper right')
fig.suptitle(f'Backtest of the model ladder, {results["season"].nunique()} season pairs, {int(results.groupby("season")["hitters"].first().sum())} hitters scored', x=0.01, ha='left', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_hitters_lkj_v2_fig4.png'))
plt.close(fig)
