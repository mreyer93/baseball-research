#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:30:00 2026

@author: reyer
"""

# Does a shortstop's arm show up where the throw says it should?
# Question: a stronger arm saves time only on the throw, and the time it saves grows with the
# length of the throw. If arm strength is part of what makes a shortstop convert ground balls, it
# should matter most on balls to his right (toward the third-base line, the longest throws) and
# least on balls toward second base, and more against fast runners, who leave less time to spare.
# Do the public numbers show that, and do teams already play strong-armed shortstops deeper?
# Data: the scored ground balls from infield_expected_outs.py (run it first); Baseball Savant arm
# strength, sprint speed, Outs Above Average (OAA) with directional splits, and fielder positioning,
# 2023 to 2026.

import os
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SEASONS = [2023, 2024, 2025, 2026]
rng = np.random.default_rng(23)

RED, BLUE, GRAY, SILVER = '#E81828', '#6CACE4', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Load the scored ground balls and download the shortstop leaderboards
# Arm strength is Savant's average of a fielder's hardest 10 percent of throws at the position
# (arm_ss, mph). Positioning is each fielder's average starting spot with nobody on base and no
# shading, by batter side: depth from home plate in feet and angle in degrees (negative toward the
# third-base line, the same convention as the spray angle).
def get_file(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r:
            open(path, 'wb').write(r.read())
        time.sleep(2)
    return path

ARM = 'https://baseballsavant.mlb.com/leaderboard/arm-strength?type=player&year={season}&minThrows=1&pos=&team=&csv=true'
POSITION = 'https://baseballsavant.mlb.com/visuals/position_data?type=player&teamId=&firstBase=0&shift=0&batSide={side}&season={season}&position=6&attempts=25'
balls = pd.read_csv(os.path.join(DATA, 'mlb_gb_scored.csv.gz'))
arm, depth = [], []
for season in SEASONS:
    a = pd.read_csv(get_file(ARM.format(season=season), os.path.join(DATA, f'arm_{season}.csv')))
    arm.append(a.assign(game_year=season)[['player_id', 'game_year', 'arm_ss', 'total_throws_ss']])
    for side in ('R', 'L'):
        pos = json.load(open(get_file(POSITION.format(side=side, season=season), os.path.join(DATA, f'position_ss_{season}_{side}.json'))))
        d = pd.DataFrame(pos['positionData'])
        d = d.assign(game_year=season, stand=side, depth=d['avg_norm_start_distance'].astype(float), angle=d['avg_norm_start_angle'].astype(float), n_pa=d['n'].astype(int))
        depth.append(d[['fielder_id', 'game_year', 'stand', 'depth', 'angle', 'n_pa']].rename(columns={'fielder_id': 'player_id'}))
arm = pd.concat(arm).dropna(subset=['arm_ss'])
print('arm rows', len(arm), 'duplicate player seasons', int(arm.duplicated(['player_id', 'game_year']).sum()))
# A shortstop who played for two teams has a row for each; combine them, weighted by plate appearances.
# Savant's league average pools both batter sides, so the league depth and angle for each side are
# the plate-appearance-weighted means of the player rows instead.
depth = pd.concat(depth)
print('positioning rows', len(depth), 'player seasons on two teams', int(depth.duplicated(['player_id', 'game_year', 'stand']).sum()))
depth = depth.assign(depth_n=depth['depth'] * depth['n_pa'], angle_n=depth['angle'] * depth['n_pa']).groupby(['player_id', 'game_year', 'stand'], as_index=False)[['depth_n', 'angle_n', 'n_pa']].sum()
depth = depth.assign(depth=depth['depth_n'] / depth['n_pa'], angle=depth['angle_n'] / depth['n_pa'])[['player_id', 'game_year', 'stand', 'depth', 'angle', 'n_pa']]
league_depth = {(s, side): (np.average(g['depth'], weights=g['n_pa']), np.average(g['angle'], weights=g['n_pa'])) for (s, side), g in depth.groupby(['game_year', 'stand'])}
sprint = pd.concat([pd.read_csv(os.path.join(DATA, f'sprint_{s}.csv')).assign(game_year=s) for s in SEASONS])
sprint = sprint.rename(columns={'sprint_speed': 'fielder_sprint_speed'})[['player_id', 'game_year', 'fielder_sprint_speed']]
print('shortstop arm seasons', len(arm), 'positioning player seasons and sides', len(depth))
print('league shortstop depth and angle by season and batter side', {k: (round(v[0], 1), round(v[1], 1)) for k, v in league_depth.items()})
print(arm['arm_ss'].describe().round(2))

####
# arm rows 252 duplicate player seasons 0
# positioning rows 745 player seasons on two teams 20
# shortstop arm seasons 252 positioning player seasons and sides 725
# league shortstop depth and angle by season and batter side {(2023, 'L'): (np.float64(152.4), np.float64(-6.5)), (2023, 'R'): (np.float64(148.3), np.float64(-18.2)), (2024, 'L'): (np.float64(152.7), np.float64(-6.3)), (2024, 'R'): (np.float64(148.4), np.float64(-18.4)), (2025, 'L'): (np.float64(152.3), np.float64(-6.0)), (2025, 'R'): (np.float64(148.3), np.float64(-18.1)), (2026, 'L'): (np.float64(152.5), np.float64(-6.1)), (2026, 'R'): (np.float64(148.1), np.float64(-18.4))}
# count    252.00
# mean      84.13
# std        3.66
# min       75.80
# 25%       81.20
# 50%       83.65
# 75%       86.45
# max       95.30
# Name: arm_ss, dtype: float64
####

# Step 2 How much time does arm strength buy on the throw?
# Arithmetic only. A shortstop at league depth (2025, right-handed batters) fields the ball along its spray angle and throws to
# first base, which sits at (63.6, 63.6) feet in the positioning coordinates (x toward first, y
# toward center). Throw time is distance over speed, ignoring drag and the arc. Near first base a runner
# is close to his top speed, which Savant measures as sprint speed (feet per second in his fastest
# one-second window), so the average batter's sprint speed turns the time saved into feet.
league_r, league_angle = league_depth[(2025, 'R')]
runner_speed = balls['batter_sprint_speed'].mean()
print('average batter sprint speed on ground balls', round(runner_speed, 2), 'ft/s')
for angle in [league_angle - 8, league_angle, league_angle + 8, 0.0]:
    x, y = league_r * np.sin(np.radians(angle)), league_r * np.cos(np.radians(angle))
    throw = np.hypot(63.64 - x, 63.64 - y)
    t_avg, t_strong = throw / (84 * 5280 / 3600), throw / (88 * 5280 / 3600)
    print(f'spray angle {angle:6.1f}: throw {throw:5.1f} ft, 84 mph {t_avg:.3f} s, 88 mph {t_strong:.3f} s, saved {t_avg - t_strong:.3f} s, {(t_avg - t_strong) * runner_speed:.2f} ft of running')

####
# average batter sprint speed on ground balls 27.35 ft/s
# spray angle  -26.1: throw 146.4 ft, 84 mph 1.189 s, 88 mph 1.135 s, saved 0.054 s, 1.48 ft of running
# spray angle  -18.1: throw 134.2 ft, 84 mph 1.089 s, 88 mph 1.040 s, saved 0.050 s, 1.35 ft of running
# spray angle  -10.1: throw 121.7 ft, 84 mph 0.988 s, 88 mph 0.943 s, saved 0.045 s, 1.23 ft of running
# spray angle    0.0: throw 105.9 ft, 84 mph 0.860 s, 88 mph 0.821 s, saved 0.039 s, 1.07 ft of running
####
# Four mph of arm saves 0.039 seconds on the 106-foot throw from behind second base and 0.054 on the
# 146-foot throw from deep toward third, 1.4 times as much: 1.1 to 1.5 feet of the batter's run to first.

# Step 3 The shortstop's share of each ground ball, and the direction relative to where he stood
# share_ss is the shortstop's responsibility from Step 4 of infield_expected_outs.py: 1 for a ball
# he fielded, 0 for a ball another infielder fielded, and his share for a ball that reached the
# outfield. Direction is the ball's spray angle minus the shortstop's own average starting angle
# against that batter side (the league average for that side when Savant lists no positioning for
# him, which it does below 25 plate appearances): more than 4 degrees toward third base is "to his right", more than 4 degrees
# toward second is "to his left", and the rest is "at him". The throw distance uses his own depth.
ss = balls[balls['share_ss'] > 0].rename(columns={'fielder_6': 'player_id'}).copy()
ss = ss.merge(arm, on=['player_id', 'game_year'], how='inner')
ss = ss.merge(depth, on=['player_id', 'game_year', 'stand'], how='left')
ss['depth'] = ss['depth'].fillna(ss.apply(lambda r: league_depth[(r['game_year'], r['stand'])][0], axis=1))
ss['angle'] = ss['angle'].fillna(ss.apply(lambda r: league_depth[(r['game_year'], r['stand'])][1], axis=1))
ss = ss.merge(sprint, on=['player_id', 'game_year'], how='left')
ss['relative_angle'] = ss['spray_angle'] - ss['angle']
ss['direction'] = np.select([ss['relative_angle'] < -4, ss['relative_angle'] > 4], ['to his right', 'to his left'], 'at him')
fx, fy = ss['depth'] * np.sin(np.radians(ss['spray_angle'])), ss['depth'] * np.cos(np.radians(ss['spray_angle']))
ss['throw_ft'] = np.hypot(63.64 - fx, 63.64 - fy)
ss['oae'] = ss['share_ss'] * (ss['out'] - ss['p_out'] - ss['ss_league_rate'])  # against the average shortstop that season, as in Step 5 of infield_expected_outs.py
print('shortstop balls with an arm strength', len(ss), 'shortstop seasons', ss.groupby(['player_id', 'game_year']).ngroups, 'weighted opportunities', round(ss['share_ss'].sum(), 1))
print(ss.groupby('direction').apply(lambda g: pd.Series({'opportunities': g['share_ss'].sum(), 'out_rate': np.average(g['out'], weights=g['share_ss']), 'expected': np.average(g['p_out'], weights=g['share_ss']), 'throw_ft': np.average(g['throw_ft'], weights=g['share_ss'])})).round(3))

####
# shortstop balls with an arm strength 65474 shortstop seasons 252 weighted opportunities 53861.6
#               opportunities  out_rate  expected  throw_ft
# direction
# at him            19977.544     0.883     0.855   131.750
# to his left       17564.770     0.590     0.618   113.846
# to his right      16319.241     0.651     0.677   142.490
####
# Balls hit at the shortstop's usual spot are converted 2.8 points above the fielder-blind expectation (88.3
# percent against 85.5); balls to either side fall short. That is positioning showing through.

# Step 4 Strong arms against weak arms, by the length of the throw
# Shortstop seasons are split into thirds by arm strength (within season). For each band of throw
# distance, the difference in outs above expected per 100 chances between the strongest and the
# weakest third, with a 90 percent bootstrap interval that resamples shortstop seasons. If the arm
# works through the throw, the difference should grow with the distance. The direction bins of Step 3
# are not used for this, because the expected-out model does not know where the shortstop
# stood: balls hit at him are converted well above expectation by every arm (Step 3 output), which
# is the value of positioning (team_positioning.py) rather than of the arm.
print(ss.groupby('direction').apply(lambda g: 100 * g['oae'].sum() / g['share_ss'].sum()).round(2).to_dict())
ss['arm_third'] = ss.groupby('game_year')['arm_ss'].transform(lambda a: pd.qcut(a.rank(method='first'), 3, labels=['weakest third', 'middle third', 'strongest third']))
BANDS = [0, 115, 130, 145, 400]
BAND_LABELS = ['under 115 ft', '115 to 130 ft', '130 to 145 ft', '145 ft and over']
ss['throw_band'] = pd.cut(ss['throw_ft'], BANDS, labels=BAND_LABELS, right=False)
season_band = ss.groupby(['player_id', 'game_year', 'arm_third', 'throw_band'], observed=True).agg(oae=('oae', 'sum'), opp=('share_ss', 'sum')).reset_index()
def per_100(g):
    return 100 * g['oae'].sum() / g['opp'].sum()
table = []
for band in BAND_LABELS:
    strong = season_band[(season_band['arm_third'] == 'strongest third') & (season_band['throw_band'] == band)]
    weak = season_band[(season_band['arm_third'] == 'weakest third') & (season_band['throw_band'] == band)]
    boot = [per_100(strong.iloc[rng.integers(0, len(strong), len(strong))]) - per_100(weak.iloc[rng.integers(0, len(weak), len(weak))]) for _ in range(2000)]
    table.append((band, round(strong['opp'].sum() + weak['opp'].sum(), 1), per_100(strong), per_100(weak), per_100(strong) - per_100(weak), np.percentile(boot, 5), np.percentile(boot, 95)))
table = pd.DataFrame(table, columns=['throw', 'opportunities', 'strongest', 'weakest', 'difference', 'lo90', 'hi90'])
print(table.round(2).to_string())

####
# {'at him': 3.58, 'to his left': -2.06, 'to his right': -1.86}
#              throw  opportunities  strongest  weakest  difference  lo90  hi90
# 0     under 115 ft         6015.5      -4.79    -1.98       -2.81 -4.80 -0.88
# 1    115 to 130 ft        10498.6       1.36     3.24       -1.88 -3.44 -0.36
# 2    130 to 145 ft        15082.8       1.47     1.56       -0.09 -1.17  1.13
# 3  145 ft and over         4172.4      -3.64    -6.63        3.00  0.21  5.65
####
# The difference rises with throw length, from -2.81 under 115 feet to +3.00 at 145 feet and over (90 percent
# interval 0.21 to 5.65).

# Figure 1 Strongest third of arms minus weakest third, by throw distance
fig, ax = plt.subplots(figsize=(9, 5.5))
xpos = np.arange(len(table))
ax.bar(xpos, table['difference'], width=0.6, color=RED)
ax.errorbar(xpos, table['difference'], yerr=[table['difference'] - table['lo90'], table['hi90'] - table['difference']], fmt='none', ecolor='black', lw=1, capsize=4)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(table['throw'])
ax.set_xlabel('Distance of the throw to first base')
ax.set_ylabel('Outs above expected per 100 chances,\nstrongest third minus weakest third')
ax.set_title('Shortstops, 2023 to 2026: what the arm adds, by throw length')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'shortstop_arm_tests_fig1.png'))
plt.close(fig)

# Step 5 One play-level model for all three predictions
# A logistic regression on every shortstop ball, weighted by his share, with the expected-out
# model's log odds as an offset (so the coefficients measure departures from an average shortstop).
# Arm strength and the shortstop's own sprint speed are z-scored within season. The predictions
# are a positive arm main effect, a positive arm x throw-distance interaction and a positive
# arm x runner-speed interaction. The runner-speed and throw-distance main effects sit in the model
# to absorb any miscalibration of the offset. Intervals are 90 percent bootstrap intervals that
# resample shortstop seasons, because plays by the same shortstop are not independent.
ss = ss.dropna(subset=['batter_sprint_speed', 'fielder_sprint_speed']).copy()
ss['arm_z'] = ss.groupby('game_year')['arm_ss'].transform(lambda a: (a - a.mean()) / a.std())
ss['fielder_speed_z'] = ss.groupby('game_year')['fielder_sprint_speed'].transform(lambda a: (a - a.mean()) / a.std())
ss['runner_z'] = (ss['batter_sprint_speed'] - ss['batter_sprint_speed'].mean()) / ss['batter_sprint_speed'].std()
ss['throw_z'] = (ss['throw_ft'] - ss['throw_ft'].mean()) / ss['throw_ft'].std()
ss['offset'] = np.log(np.clip(ss['p_out'], 1e-4, 1 - 1e-4) / (1 - np.clip(ss['p_out'], 1e-4, 1 - 1e-4)))
TERMS = ['arm_z', 'fielder_speed_z', 'runner_z', 'throw_z']
ss['arm_x_throw'] = ss['arm_z'] * ss['throw_z']
ss['arm_x_runner'] = ss['arm_z'] * ss['runner_z']
TERMS = TERMS + ['arm_x_throw', 'arm_x_runner']

def fit_glm(d):
    X = sm.add_constant(d[TERMS])
    return sm.GLM(d['out'], X, family=sm.families.Binomial(), offset=d['offset'], freq_weights=d['share_ss']).fit().params

point = fit_glm(ss)
groups = ss.groupby(['player_id', 'game_year']).indices
keys = list(groups.keys())
boot = []
for b in range(300):
    pick = rng.integers(0, len(keys), len(keys))
    idx = np.concatenate([groups[keys[k]] for k in pick])
    boot.append(fit_glm(ss.iloc[idx]))
boot = pd.DataFrame(boot)
result = pd.DataFrame({'coef': point, 'lo90': boot.quantile(0.05), 'hi90': boot.quantile(0.95)})
print('plays', len(ss), 'shortstop seasons', len(keys))
print(result.round(4))

####
# plays 65392 shortstop seasons 252
#                    coef    lo90    hi90
# const           -0.0480 -0.0730 -0.0231
# arm_z           -0.0259 -0.0511  0.0002
# fielder_speed_z  0.0041 -0.0200  0.0316
# runner_z        -0.0173 -0.0338  0.0011
# throw_z          0.0511  0.0221  0.0809
# arm_x_throw      0.0446  0.0140  0.0765
# arm_x_runner     0.0118 -0.0059  0.0286
####
# The arm x throw interaction is positive (0.0446, interval 0.0140 to 0.0765) and the arm main effect slightly
# negative; arm x runner speed is positive (0.0118) but its interval includes zero.

# Step 5.1 What one standard deviation of arm is worth, in outs per 100 chances
# The fitted model's average change in out probability for +1 SD of arm, on the balls each
# direction actually gets, against the expected-out probabilities.
for direction in ['to his right', 'at him', 'to his left']:
    d = ss[ss['direction'] == direction]
    base = d['offset'] + point['const'] + point['runner_z'] * d['runner_z'] + point['throw_z'] * d['throw_z']
    plus = base + point['arm_z'] + point['arm_x_throw'] * d['throw_z'] + point['arm_x_runner'] * d['runner_z']
    gain = np.average(1 / (1 + np.exp(-plus)) - 1 / (1 + np.exp(-base)), weights=d['share_ss'])
    print(f'{direction}: +1 SD of arm is worth {100 * gain:.2f} outs per 100 chances (mean throw {np.average(d["throw_ft"], weights=d["share_ss"]):.0f} ft)')

####
# to his right: +1 SD of arm is worth 0.12 outs per 100 chances (mean throw 142 ft)
# at him: +1 SD of arm is worth -0.20 outs per 100 chances (mean throw 132 ft)
# to his left: +1 SD of arm is worth -0.90 outs per 100 chances (mean throw 114 ft)
####
# On the balls each direction actually gets, +1 SD of arm (3.7 mph) is worth +0.12 outs per 100 chances to
# his right and -0.90 to his left.

# Step 6 The same question in Statcast's own directional OAA
# Savant splits each shortstop's OAA by the direction he had to move: toward the third-base line,
# toward the first-base line, in front, and behind. Per 100 of this model's opportunities, the
# correlation of arm strength with each direction across shortstop seasons (at least 150
# opportunities), with a 90 percent bootstrap interval.
oaa = pd.concat([pd.read_csv(os.path.join(DATA, f'oaa_{s}_6.csv')).assign(game_year=s) for s in SEASONS])
fielders = pd.read_csv(os.path.join(DATA, 'mlb_fielder_seasons.csv'))
fielders = fielders[(fielders['pos'] == 6) & (fielders['opportunities'] >= 150)]
seasons_ss = fielders.merge(oaa, on=['player_id', 'game_year']).merge(arm, on=['player_id', 'game_year'])
directions = {'toward third-base line': 'outs_above_average_lateral_toward3bline', 'toward first-base line': 'outs_above_average_lateral_toward1bline', 'in front': 'outs_above_average_infront', 'behind': 'outs_above_average_behind', 'all OAA': 'outs_above_average'}
direction_r = []
for label, col in directions.items():
    rate = 100 * seasons_ss[col] / seasons_ss['opportunities']
    r = np.corrcoef(seasons_ss['arm_ss'], rate)[0, 1]
    boot_r = []
    for _ in range(2000):
        i = rng.integers(0, len(seasons_ss), len(seasons_ss))
        boot_r.append(np.corrcoef(seasons_ss['arm_ss'].iloc[i], rate.iloc[i])[0, 1])
    direction_r.append((label, r, np.percentile(boot_r, 5), np.percentile(boot_r, 95)))
    print(f'{label}: r = {r:.3f} (90% {np.percentile(boot_r, 5):.3f} to {np.percentile(boot_r, 95):.3f}), n = {len(seasons_ss)}')
direction_r = pd.DataFrame(direction_r, columns=['direction', 'r', 'lo90', 'hi90'])

####
# toward third-base line: r = 0.044 (90% -0.092 to 0.182), n = 133
# toward first-base line: r = -0.125 (90% -0.265 to 0.033), n = 133
# in front: r = 0.187 (90% 0.051 to 0.312), n = 133
# behind: r = 0.203 (90% 0.058 to 0.341), n = 133
# all OAA: r = 0.103 (90% -0.024 to 0.233), n = 133
####
# Arm strength goes with plays in front of him (0.187) and behind him (0.203), where time is short or the
# throw long, and hardly with lateral plays toward third (0.044).

# Figure 2 Correlation of arm strength with each part of Statcast's OAA
fig, ax = plt.subplots(figsize=(9, 5))
ypos = np.arange(len(direction_r))[::-1]
ax.errorbar(direction_r['r'], ypos, xerr=[direction_r['r'] - direction_r['lo90'], direction_r['hi90'] - direction_r['r']], fmt='o', color=RED, ecolor=GRAY, capsize=4, ms=8)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(direction_r['direction'])
ax.set_xlabel('Correlation with arm strength, across shortstop seasons (90% interval)')
ax.set_title(f'Shortstop OAA per 100 chances by direction, 2023 to 2026 (n = {len(seasons_ss)})')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'shortstop_arm_tests_fig2.png'))
plt.close(fig)

# Step 7 Do teams already play strong-armed shortstops deeper?
# Depth against right-handed batters (most shortstop chances) for shortstop seasons with at least
# 100 plate appearances of positioning data, against arm strength and sprint speed.
pos_r = depth[(depth['stand'] == 'R') & (depth['n_pa'] >= 100)].merge(arm[arm['total_throws_ss'] >= 50], on=['player_id', 'game_year']).merge(sprint, on=['player_id', 'game_year'], how='left')
slope, intercept = np.polyfit(pos_r['arm_ss'], pos_r['depth'], 1)
print('shortstop seasons', len(pos_r), 'r(depth, arm)', round(np.corrcoef(pos_r['depth'], pos_r['arm_ss'])[0, 1], 3), 'slope', round(slope, 3), 'ft per mph')
ok = pos_r.dropna(subset=['fielder_sprint_speed'])
print('r(depth, sprint speed)', round(np.corrcoef(ok['depth'], ok['fielder_sprint_speed'])[0, 1], 3), 'n', len(ok))
print('r(arm, sprint speed)', round(np.corrcoef(ok['arm_ss'], ok['fielder_sprint_speed'])[0, 1], 3))
print('r(angle, arm)', round(np.corrcoef(pos_r['angle'], pos_r['arm_ss'])[0, 1], 3), '(negative angle is toward third base)')
print('depth: mean', round(pos_r['depth'].mean(), 1), 'SD', round(pos_r['depth'].std(), 1), 'ft; arm: SD', round(pos_r['arm_ss'].std(), 2), 'mph')

####
# shortstop seasons 250 r(depth, arm) -0.064 slope -0.022 ft per mph
# r(depth, sprint speed) 0.008 n 250
# r(arm, sprint speed) 0.19
# r(angle, arm) -0.101 (negative angle is toward third base)
# depth: mean 148.3 SD 1.3 ft; arm: SD 3.66 mph
####
# Depth barely varies (SD 1.3 ft; Savant reports it in whole feet) and does not follow arm strength (r = -0.064).

# Figure 3 Shortstop depth against arm strength
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pos_r['arm_ss'], pos_r['depth'], s=22, color=RED, alpha=0.7, edgecolor='none')
xs = np.linspace(pos_r['arm_ss'].min(), pos_r['arm_ss'].max(), 10)
ax.plot(xs, intercept + slope * xs, color=GRAY, lw=2)
ax.set_xlabel('Arm strength at shortstop (mph, Savant)')
ax.set_ylabel('Average depth against right-handed batters (ft)')
ax.set_title(f'Shortstop seasons, 2023 to 2026 (n = {len(pos_r)}, {slope:.2f} ft per mph)')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'shortstop_arm_tests_fig3.png'))
plt.close(fig)
