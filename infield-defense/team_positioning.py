#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:20:00 2026

@author: reyer
"""

# How much of an infielder's credit belongs to his team's positioning?
# Question: the expected-out model in infield_expected_outs.py does not know where the fielders stood,
# so when a team puts its shortstop where the ball is hit, the out counts as the shortstop's. Statcast's
# Outs Above Average (OAA) starts from where each fielder stood, so the same out earns him little. The
# gap between the two (this model's outs above expected minus OAA) should then hold the value of the
# team's positioning. If it does, the gap should behave like a team trait and not a player trait: the
# second baseman, third baseman and shortstop of the same team should share it, it should persist from
# season to season when the players change, and it should not come from the home park.
# Data: data/mlb_fielder_seasons.csv and data/mlb_gb_scored.csv.gz from infield_expected_outs.py (run
# it first), and the Savant OAA leaderboards it downloads, 2023 to 2026.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SEASONS = [2023, 2024, 2025, 2026]
POSITIONS = [4, 5, 6]
rng = np.random.default_rng(31)

RED, BLUE, GRAY, SILVER = '#E81828', '#6CACE4', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Load the fielder seasons, OAA and the scored ground balls
# OAA's team column reads '---' for a player who played for more than one team in a season; his OAA
# cannot be split by team, so those fielder seasons are left out of the team analyses. The fielding
# team on each ball is the home team in the top of an inning and the away team in the bottom.
fielders = pd.read_csv(os.path.join(DATA, 'mlb_fielder_seasons.csv'))
oaa = pd.concat([pd.read_csv(os.path.join(DATA, f'oaa_{s}_{pos}.csv')).assign(game_year=s, pos=pos) for s in SEASONS for pos in POSITIONS])
oaa = oaa.rename(columns={'outs_above_average': 'oaa', 'display_team_name': 'team', 'last_name, first_name': 'name'})[['player_id', 'name', 'game_year', 'pos', 'team', 'oaa']]
balls = pd.read_csv(os.path.join(DATA, 'mlb_gb_scored.csv.gz'))
balls['fielding_team'] = np.where(balls['inning_topbot'] == 'Top', balls['home_team'], balls['away_team'])
balls['at_home'] = (balls['inning_topbot'] == 'Top').astype(int)
print('fielder seasons', len(fielders), 'OAA rows', len(oaa), 'balls', len(balls))

####
# fielder seasons 2483 OAA rows 1863 balls 206205
####

# Step 2 The gap between this model and OAA, for each fielder season
# Fielder seasons at second base, third base and shortstop with at least 100 opportunities, on one
# team. Both measures are relative to the average fielder at the position, so the gap averages zero.
gap = fielders[fielders['pos'].isin(POSITIONS) & (fielders['opportunities'] >= 100)].merge(oaa, on=['player_id', 'game_year', 'pos'], how='inner')
print('fielder seasons with OAA', len(gap), 'on more than one team (left out)', int((gap['team'] == '---').sum()))
gap = gap[gap['team'] != '---'].copy()
gap['gap'] = gap['oae'] - gap['oaa']
gap['gap_per_100'] = 100 * gap['gap'] / gap['opportunities']
# Savant's OAA file names clubs by nickname and the pitch data by three-letter code; the players link them.
lookup = pd.concat([gap[gap['pos'] == pos][['player_id', 'game_year', 'team']].merge(balls[[col, 'game_year', 'fielding_team']].rename(columns={col: 'player_id'}), on=['player_id', 'game_year']) for pos, col in [(4, 'fielder_4'), (5, 'fielder_5'), (6, 'fielder_6')]])
lookup = lookup.groupby(['team', 'game_year'])['fielding_team'].agg(lambda s: s.value_counts().index[0]).rename('code').reset_index()
print('team seasons linked to a code', len(lookup), 'codes per team name', lookup.groupby('team')['code'].nunique().max())
for pos in POSITIONS:
    g = gap[gap['pos'] == pos]
    print('position', pos, 'fielder seasons', len(g), 'SD this model', round(g['oae'].std(), 2), 'SD OAA', round(g['oaa'].std(), 2), 'SD gap', round(g['gap'].std(), 2), 'mean gap', round(g['gap'].mean(), 2))

####
# fielder seasons with OAA 506 on more than one team (left out) 35
# team seasons linked to a code 120 codes per team name 1
# position 4 fielder seasons 171 SD this model 7.55 SD OAA 6.81 SD gap 6.94 mean gap -0.65
# position 5 fielder seasons 138 SD this model 7.21 SD OAA 5.92 SD gap 5.88 mean gap 0.03
# position 6 fielder seasons 162 SD this model 8.34 SD OAA 7.9 SD gap 8.85 mean gap -1.17
####
# The gap is as large as either measure (SD 8.85 outs at shortstop against 7.9 for OAA), so the two models
# disagree about shortstops about as much as shortstops differ from each other.

# Step 3 Do the three positions on a team share the gap?
# For each team season, the gap per 100 opportunities at one position against the pooled gap of the
# other two positions (leave one position out), across team seasons. The null distribution comes from
# shuffling team labels within each season and position 1,000 times, which keeps every number and
# breaks only the link between teammates.
def team_rates(g):
    sums = g.groupby(['team', 'game_year'])[['gap', 'opportunities']].sum()
    return 100 * sums['gap'] / sums['opportunities']

def leave_one_out_r(g):
    rs = []
    for pos in POSITIONS:
        both = pd.concat([team_rates(g[g['pos'] == pos]), team_rates(g[g['pos'] != pos])], axis=1, keys=['own', 'others']).dropna()
        rs.append(np.corrcoef(both['own'], both['others'])[0, 1])
    return rs

observed = leave_one_out_r(gap)
null = []
for _ in range(1000):
    shuffled = gap.copy()
    shuffled['team'] = shuffled.groupby(['game_year', 'pos'])['team'].transform(lambda t: rng.permutation(t.to_numpy()))
    null.append(np.mean(leave_one_out_r(shuffled)))
null = np.array(null)
keys = gap[['team', 'game_year']].drop_duplicates().reset_index(drop=True)
boot = []
for _ in range(1000):
    pick = keys.iloc[rng.integers(0, len(keys), len(keys))].reset_index(drop=True).reset_index().rename(columns={'index': 'draw'})
    resampled = pick.merge(gap, on=['team', 'game_year'])
    resampled['team'] = resampled['team'] + '_' + resampled['draw'].astype(str)
    boot.append(np.mean(leave_one_out_r(resampled)))
print('leave-one-position-out r (2B, 3B, SS):', np.round(observed, 3), 'mean', round(np.mean(observed), 3), '90% bootstrap over team seasons', round(np.percentile(boot, 5), 3), 'to', round(np.percentile(boot, 95), 3))
print('shuffled teams: mean', round(null.mean(), 3), '95th percentile', round(np.percentile(null, 95), 3), 'share of shuffles at or above the observed mean', round(np.mean(null >= np.mean(observed)), 4))

####
# leave-one-position-out r (2B, 3B, SS): [0.295 0.274 0.256] mean 0.275 90% bootstrap over team seasons 0.111 to 0.416
# shuffled teams: mean 0.0 95th percentile 0.127 share of shuffles at or above the observed mean 0.0
####
# Every position agrees with its teammates (0.256 to 0.295); no shuffle of the team labels reached the
# observed mean (95th percentile of the shuffles 0.127).

# Step 4 Does a team's gap persist, and does it pass to new players?
# The team score is the pooled gap per 100 opportunities at second, third and shortstop. Season t
# against season t + 1 for all players; then season t + 1 scored with only the players who had not
# played that position for that team at all in season t (in any game, from the pitch data) (new arrivals, including those promoted from the
# minors). If positioning drives the gap, the new arrivals take on the team's score. For comparison,
# players who stayed at the same position on the same team, scored against themselves.
team = team_rates(gap).rename('score').reset_index()
presence = pd.concat([balls[[col, 'game_year', 'fielding_team']].drop_duplicates().rename(columns={col: 'player_id'}).assign(pos=pos) for pos, col in [(4, 'fielder_4'), (5, 'fielder_5'), (6, 'fielder_6')]])
presence = presence.merge(lookup, left_on=['fielding_team', 'game_year'], right_on=['code', 'game_year'])
previous = presence[['team', 'game_year', 'pos', 'player_id']].drop_duplicates().assign(game_year=presence['game_year'] + 1, was_here=1)
gap = gap.merge(previous, on=['team', 'game_year', 'pos', 'player_id'], how='left')
gap['was_here'] = gap['was_here'].fillna(0).astype(int)
new_arrivals = team_rates(gap[gap['was_here'] == 0]).rename('score_new').reset_index()
stayers = gap[gap['was_here'] == 1][['team', 'game_year', 'pos', 'player_id', 'gap_per_100']]
stayers = stayers.merge(gap[['team', 'game_year', 'pos', 'player_id', 'gap_per_100']].assign(game_year=gap['game_year'] + 1), on=['team', 'game_year', 'pos', 'player_id'], suffixes=('', '_before'))
pairs = team.merge(team.assign(game_year=team['game_year'] + 1), on=['team', 'game_year'], suffixes=('', '_before'))
pairs_new = new_arrivals.merge(team.assign(game_year=team['game_year'] + 1), on=['team', 'game_year']).rename(columns={'score': 'score_before'})

def r_with_interval(x, y, reps=2000):
    x, y = np.asarray(x), np.asarray(y)
    boot = []
    for _ in range(reps):
        i = rng.integers(0, len(x), len(x))
        boot.append(np.corrcoef(x[i], y[i])[0, 1])
    return np.corrcoef(x, y)[0, 1], np.percentile(boot, 5), np.percentile(boot, 95), len(x)

tests = []
for label, x, y in [('same team, season t to t + 1', pairs['score_before'], pairs['score']),
                    ('team in t, its new arrivals in t + 1', pairs_new['score_before'], pairs_new['score_new']),
                    ('same player, same team, t to t + 1', stayers['gap_per_100_before'], stayers['gap_per_100'])]:
    r, lo, hi, n = r_with_interval(x, y)
    tests.append((label, r, lo, hi, n))
    print(f'{label}: r = {r:.3f} (90% {lo:.3f} to {hi:.3f}), n = {n}')

####
# same team, season t to t + 1: r = 0.407 (90% 0.260 to 0.530), n = 90
# team in t, its new arrivals in t + 1: r = 0.082 (90% -0.139 to 0.306), n = 63
# same player, same team, t to t + 1: r = 0.346 (90% 0.226 to 0.457), n = 172
####
# A team's average persists (0.407), but its new arrivals do not take it on (0.082) while the same players
# on the same team keep theirs (0.346). The persistence travels with the players, not the team.

# Step 5 Is it the home park?
# A park that makes ground balls harder to field (a fast surface, a hard infield) lowers this model's
# credit for every fielder who plays there, which would look like a team trait for the home team. The
# visiting infielders measure the park without the home team's positioning: their outs above expected
# per 100 chances at second, third and shortstop in that park and season, against the home team's
# score. Visiting infielders' outs above expected are against the season average for their position.
league_rate = {}
for pos, col in [(4, 'fielder_4'), (5, 'fielder_5'), (6, 'fielder_6')]:
    f = fielders[fielders['pos'] == pos]
    league_rate[pos] = (f.groupby('game_year')['oae_raw'].sum() / f.groupby('game_year')['opportunities'].sum()).to_dict()
visitors = balls[balls['at_home'] == 0].copy()
visitors['ss_rate'] = visitors['game_year'].map(league_rate[6])
park = visitors[visitors['share_ss'] > 0].groupby(['home_team', 'game_year']).apply(lambda g: 100 * (g['share_ss'] * (g['out'] - g['p_out'] - g['ss_rate'])).sum() / g['share_ss'].sum()).rename('visiting_ss_oae_per_100').reset_index()
park = park.merge(lookup, left_on=['home_team', 'game_year'], right_on=['code', 'game_year']).merge(team, on=['team', 'game_year'])
r, lo, hi, n = r_with_interval(park['visiting_ss_oae_per_100'], park['score'])
tests.append(('visiting shortstops in the park, home team', r, lo, hi, n))
print(f'visiting shortstops at the park against the home team score: r = {r:.3f} (90% {lo:.3f} to {hi:.3f}), n = {n}')
print('SD of visiting shortstops per 100 by park season', round(park['visiting_ss_oae_per_100'].std(), 2))

####
# visiting shortstops at the park against the home team score: r = 0.167 (90% 0.039 to 0.287), n = 120
# SD of visiting shortstops per 100 by park season 2.58
####
# Visiting shortstops in a park share some of the home team's gap (0.167), so the park carries a part of it.

# Figure 1 The five tests of the team-positioning reading, as correlations with 90 percent intervals
summary = pd.DataFrame([('teammates, same season', np.mean(observed), np.percentile(boot, 5), np.percentile(boot, 95), len(keys))] + tests, columns=['test', 'r', 'lo', 'hi', 'n'])
fig, ax = plt.subplots(figsize=(11, 5.5))
ypos = np.arange(len(summary))[::-1]
ax.errorbar(summary['r'], ypos, xerr=[summary['r'] - summary['lo'], summary['hi'] - summary['r']], fmt='o', color=RED, ecolor=GRAY, capsize=4, ms=8)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels([f'{t} (n = {n})' for t, n in zip(summary['test'], summary['n'])], fontsize=11)
ax.set_xlabel('Correlation, with 90% interval')
fig.suptitle('Is the gap (this model minus OAA) a team trait? 2B, 3B and SS, 2023 to 2026', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'team_positioning_fig1.png'))
plt.close(fig)

# Step 6 An upper bound on what a team component could be worth
# The SD of the team score across team seasons bounds the spread of anything team-level in the gap
# (positioning, park, staff) from above, because it also holds noise and player traits. Scaling by the
# season-to-season r keeps only the persistent part (persistent SD = SD x square root of r), which the
# new-arrival test in Step 4 shows is mostly carried by the same players, so this is a ceiling, not an
# estimate. Opportunities per team season at second, third and shortstop convert it to outs, and
# Statcast's run expectancy change on ground balls converts outs to runs: the average delta_run_exp of
# a ground ball that was not an out minus that of one that was.
sd_score = team['score'].std()
r_persist = np.corrcoef(pairs['score_before'], pairs['score'])[0, 1]
opp_per_team = gap.groupby(['team', 'game_year'])['opportunities'].sum().mean()
run_value = balls.loc[balls['out'] == 0, 'delta_run_exp'].mean() - balls.loc[balls['out'] == 1, 'delta_run_exp'].mean()
persistent_sd = sd_score * np.sqrt(max(r_persist, 0))
print('team score SD per 100', round(sd_score, 2), 'persistent part', round(persistent_sd, 2), 'opportunities per team season (2B, 3B, SS, single-team players)', round(opp_per_team, 0))
print('runs per ground-ball out (not-out minus out, delta_run_exp)', round(run_value, 3))
print('persistent team SD in outs per season', round(persistent_sd * opp_per_team / 100, 1), 'in runs', round(persistent_sd * opp_per_team / 100 * run_value, 1))

####
# team score SD per 100 1.79 persistent part 1.15 opportunities per team season (2B, 3B, SS, single-team players) 978.0
# runs per ground-ball out (not-out minus out, delta_run_exp) 0.717
# persistent team SD in outs per season 11.2 in runs 8.0
####
# A ceiling of about 11 outs (8 runs) per team season at the three positions, not an estimate of positioning.
