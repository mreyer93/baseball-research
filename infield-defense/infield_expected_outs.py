#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:12:00 2026

@author: reyer
"""

# Ground-ball expected outs for infielders from public Statcast data
# Question: can an expected-out model built only from what Baseball Savant publishes for each
# ground ball (exit velocity, launch angle, spray angle, the batter's sprint speed, the base-out
# state and the infield alignment) credit infielders the way Statcast's own Outs Above Average
# (OAA) does, and is the credit as stable from one season to the next?
# Data: Baseball Savant pitch-level search, every regular-season ground ball from 2023 to 2026
# (the four seasons since the 2023 rule that requires two infielders on each side of second base
# with all four on the infield dirt); Savant sprint speed and OAA leaderboards.

import os
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
os.makedirs(DATA, exist_ok=True)
SEASONS = [2023, 2024, 2025, 2026]

# Red and light blue for the figures, gray for everything else
RED, BLUE, GRAY, SILVER = '#E81828', '#6CACE4', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Download (or load cached) every regular-season ground ball, 2023 to 2026
# Savant returns at most 25,000 rows per search, so each season is split into six date windows
# (about 10,000 ground balls each). The sprint speed leaderboard gives each batter's speed in that
# season (feet per second in his fastest one-second window on competitive runs); the OAA
# leaderboard is what the model is checked against, never trained on.
def get_file(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r:
            open(path, 'wb').write(r.read())
        time.sleep(2)
    return path

SAVANT = 'https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={season}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start}&game_date_lt={end}&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=ground%5C.%5C.ball%7C&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details'
SPRINT = 'https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season={season}&max_season={season}&position=&team=&min=0&csv=true'
OAA = 'https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder&startYear={season}&endYear={season}&split=yes&team=&range=year&min=1&pos={pos}&roles=&viz=hide&csv=true'
WINDOWS = [('03-01', '04-30'), ('05-01', '05-31'), ('06-01', '06-30'), ('07-01', '07-31'), ('08-01', '08-31'), ('09-01', '10-31')]
frames = []
for season in SEASONS:
    for start, end in WINDOWS:
        path = get_file(SAVANT.format(season=season, start=f'{season}-{start}', end=f'{season}-{end}'), os.path.join(DATA, f'mlb_gb_{season}_{start}.csv'))
        chunk = pd.read_csv(path, low_memory=False)
        assert len(chunk) < 25000, path
        frames.append(chunk)
    get_file(SPRINT.format(season=season), os.path.join(DATA, f'sprint_{season}.csv'))
    for pos in (3, 4, 5, 6):
        get_file(OAA.format(season=season, pos=pos), os.path.join(DATA, f'oaa_{season}_{pos}.csv'))
balls = pd.concat(frames, ignore_index=True)
balls = balls[(balls['game_type'] == 'R') & (balls['bb_type'] == 'ground_ball')].copy()
print('ground balls', len(balls))
print(balls.groupby('game_year').size())

####
# ground balls 212297
# game_year
# 2023    53500
# 2024    53085
# 2025    52916
# 2026    52796
# dtype: int64
####

# Step 2 Clean the ground balls and define the outcome
# Bunts come out (the play is a different one). A play counts as an out when the defense retired
# any runner: the batter (field_out), a forced runner (force_out, fielders_choice_out) or two
# (grounded_into_double_play, double_play). Errors and fielder's choices with nobody out are not
# outs, as in OAA. Spray angle is the direction of the ball from home plate in degrees, from the
# hit coordinates (Savant's home plate sits at hc_x = 125.42, hc_y = 198.27): negative toward the
# third-base line, positive toward first. The hit coordinates mark where the ball was fielded, so
# only their direction is used; their distance from home would give the answer away (a ball fielded
# in the outfield is a hit).
balls = balls[~balls['des'].str.contains('bunt', case=False, na=False)]
balls = balls.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y']).copy()
OUT_EVENTS = ['field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'fielders_choice_out', 'triple_play']
balls['out'] = balls['events'].isin(OUT_EVENTS).astype(int)
balls['spray_angle'] = np.degrees(np.arctan2(balls['hc_x'] - 125.42, 198.27 - balls['hc_y']))
balls['bats_left'] = (balls['stand'] == 'L').astype(int)
balls['alignment'] = balls['if_fielding_alignment'].map({'Standard': 0, 'Infield shade': 1, 'Strategic': 2})
balls['runner_1b'] = balls['on_1b'].notna().astype(int)
balls['runner_2b'] = balls['on_2b'].notna().astype(int)
balls['runner_3b'] = balls['on_3b'].notna().astype(int)
sprint = pd.concat([pd.read_csv(os.path.join(DATA, f'sprint_{s}.csv')).assign(game_year=s) for s in SEASONS])
sprint = sprint.rename(columns={'player_id': 'batter', 'sprint_speed': 'batter_sprint_speed'})[['batter', 'game_year', 'batter_sprint_speed']]
balls = balls.merge(sprint, on=['batter', 'game_year'], how='left')
print('ground balls after cleaning', len(balls), 'out rate', round(balls['out'].mean(), 4))
print('share with a batter sprint speed', round(balls['batter_sprint_speed'].notna().mean(), 4))
print(balls['if_fielding_alignment'].value_counts(normalize=True, dropna=False).round(4).to_dict())
print(balls['events'].value_counts().to_dict())

####
# ground balls after cleaning 206205 out rate 0.7297
# share with a batter sprint speed 0.9984
# {'Standard': 0.6765, 'Infield shade': 0.2282, 'Strategic': 0.0927, nan: 0.0027}
# {'field_out': 122882, 'single': 45753, 'force_out': 13345, 'grounded_into_double_play': 12860, 'double': 4456, 'field_error': 3729, 'fielders_choice': 1533, 'fielders_choice_out': 1232, 'triple': 269, 'double_play': 142, 'triple_play': 3, 'home_run': 1}
####
# Bunts and the 0.7 percent of ground balls with no exit velocity come out, leaving 206,205, of which
# 73 percent are outs. Savant labels 68 percent standard alignment, 23 percent shaded, 9 percent strategic.

# Step 3 Expected-out model, and a backtest of every input
# Gradient-boosted trees (scikit-learn's HistGradientBoostingClassifier) give the probability that
# a ground ball becomes an out. Each season is predicted by a model trained on the other three, so
# every probability used later is out of sample. Inputs are added one group at a time and each
# group is kept only if it lowers the held-out Brier score (the mean squared error of the
# probability) and log loss.
FEATURE_SETS = {'league rate': [],
                'exit velocity, launch angle': ['launch_speed', 'launch_angle'],
                '+ spray angle, batter side': ['launch_speed', 'launch_angle', 'spray_angle', 'bats_left'],
                '+ batter sprint speed': ['launch_speed', 'launch_angle', 'spray_angle', 'bats_left', 'batter_sprint_speed'],
                '+ runners, outs, alignment': ['launch_speed', 'launch_angle', 'spray_angle', 'bats_left', 'batter_sprint_speed', 'runner_1b', 'runner_2b', 'runner_3b', 'outs_when_up', 'alignment']}
FEATURES = FEATURE_SETS['+ runners, outs, alignment']

def fit_model(train, features):
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    return model.fit(train[features], train['out'])

rows = []
for name, features in FEATURE_SETS.items():
    for season in SEASONS:
        train, test = balls[balls['game_year'] != season], balls[balls['game_year'] == season]
        if features:
            p = fit_model(train, features).predict_proba(test[features])[:, 1]
        else:
            p = np.full(len(test), train['out'].mean())
        p = np.clip(p, 1e-6, 1 - 1e-6)
        brier = np.mean((p - test['out']) ** 2)
        logloss = -np.mean(test['out'] * np.log(p) + (1 - test['out']) * np.log(1 - p))
        rows.append((name, season, len(test), brier, logloss))
        if name == '+ runners, outs, alignment':
            balls.loc[test.index, 'p_out'] = p
backtest = pd.DataFrame(rows, columns=['inputs', 'held_out_season', 'balls', 'brier', 'log_loss'])
summary = backtest.groupby('inputs', sort=False).apply(lambda g: pd.Series({'brier': np.average(g['brier'], weights=g['balls']), 'log_loss': np.average(g['log_loss'], weights=g['balls'])}))
print(summary.round(5))

####
#                                brier  log_loss
# inputs
# league rate                  0.19725   0.58358
# exit velocity, launch angle  0.16806   0.51015
# + spray angle, batter side   0.14288   0.44715
# + batter sprint speed        0.14263   0.44610
# + runners, outs, alignment   0.14002   0.43869
####
# Every group of inputs lowered both scores. Spray angle and batter side gave the largest step after exit
# velocity and launch angle (Brier 0.16806 to 0.14288); sprint speed gave the smallest (0.14288 to 0.14263).

# Step 3.1 Calibration of the held-out probabilities
# If the model is calibrated, balls given a 30 percent chance become outs 30 percent of the time.
balls['p_decile'] = pd.qcut(balls['p_out'], 10, labels=False)
calibration = balls.groupby('p_decile').agg(predicted=('p_out', 'mean'), observed=('out', 'mean'), balls=('out', 'size'))
print(calibration.round(4))
print('outs minus expected outs by season', balls.groupby('game_year').apply(lambda g: round(g['out'].sum() - g['p_out'].sum(), 1)).to_dict())

####
#           predicted  observed  balls
# p_decile
# 0            0.2165    0.1949  20621
# 1            0.4397    0.4431  20620
# 2            0.6005    0.6003  20621
# 3            0.7161    0.7158  20620
# 4            0.7956    0.7916  20621
# 5            0.8482    0.8493  20620
# 6            0.8838    0.8907  20620
# 7            0.9099    0.9171  20621
# 8            0.9317    0.9334  20620
# 9            0.9573    0.9606  20621
# outs minus expected outs by season {2023: -83.3, 2024: -176.8, 2025: 245.1, 2026: -36.5}
####
# Within 0.7 points in nine of the ten tenths; the lowest tenth is overpredicted by 2.2 points (21.65 percent
# against 19.49). In every season the outs and the expected outs differ by fewer than 250 (of about 38,000 outs).

# Figure 1 Out rate by direction and exit velocity
# Observed out rate in 2-degree spray-angle bins for right-handed batters in the standard
# alignment (dots), with the model's held-out probability for the same balls (lines). The dips are
# the holes: between third base and shortstop near -30 degrees, up the middle at 0, and between
# first and second base near +30.
fig, ax = plt.subplots(figsize=(11, 6.5))
rhh = balls[(balls['bats_left'] == 0) & (balls['alignment'] == 0) & balls['spray_angle'].between(-46, 46)].copy()
rhh['spray_bin'] = np.floor(rhh['spray_angle'] / 2) * 2 + 1
for (lo, hi), color, label in zip([(0, 85), (85, 100), (100, 125)], [BLUE, GRAY, RED], ['exit velocity under 85 mph', '85 to 100 mph', '100 mph and up']):
    band = rhh[(rhh['launch_speed'] >= lo) & (rhh['launch_speed'] < hi)].groupby('spray_bin').agg(observed=('out', 'mean'), model=('p_out', 'mean'), n=('out', 'size'))
    band = band[band['n'] >= 50]
    ax.plot(band.index, band['observed'], 'o', color=color, ms=4)
    ax.plot(band.index, band['model'], '-', color=color, lw=2, label=label)
ax.set_xlabel('Spray angle (degrees; negative is toward the third-base line)')
ax.set_ylabel('Share of ground balls turned into an out')
ax.set_title('Right-handed batters, standard alignment, 2023 to 2026')
ax.plot([], [], 'o', color=GRAY, ms=4, label='dots: observed')
ax.plot([], [], '-', color=GRAY, lw=2, label='lines: model, held-out season')
ax.legend(frameon=False, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'infield_expected_outs_fig1.png'))
plt.close(fig)

# Step 4 Who was responsible for each ground ball
# An out, an infield hit or an error belongs to the fielder who fielded the ball (hit_location,
# the scorer's position number). A ball that reached the outfield has no infielder on record, so
# it is shared out among the positions in proportion to who fields balls hit the same way. Two
# versions of the sharing rule:
#   first try: shares among everyone who fields ground balls (pitcher, catcher and the four
#     infielders), by 2-degree spray-angle bin, batter side and alignment.
#   final: shares among the four infielders only, by the same bins and also by exit velocity
#     (under 85, 85 to 100, 100 mph and up), falling back to the bins without exit velocity when
#     a bin is empty. Balls fielded by the pitcher or catcher are left out of the infielders'
#     credit, as in Statcast's OAA, which does not rate pitchers or catchers on ground balls.
# The first try gave the pitcher about 31 percent of every ball up the middle that reached the
# outfield (Step 4 output), because pitchers field so many soft comebackers there; the middle
# infielders then escaped most of the blame for hard balls through the middle.
balls['spray_bin'] = np.clip(np.floor(balls['spray_angle'] / 2) * 2, -50, 50)
balls['alignment_group'] = balls['alignment'].fillna(0)
balls['ev_band'] = np.select([balls['launch_speed'] < 85, balls['launch_speed'] < 100], [0, 1], 2)
through_rows = ~balls['hit_location'].isin([1, 2, 3, 4, 5, 6])

def responsibility(fielded_positions, keys):
    fielded = balls['hit_location'].isin(fielded_positions)
    share = balls[fielded].groupby(keys)['hit_location'].value_counts(normalize=True).unstack(fill_value=0.0)
    share.columns = [int(c) for c in share.columns]
    share = share.reindex(columns=fielded_positions, fill_value=0.0)
    through = balls.loc[through_rows, keys].merge(share, left_on=keys, right_index=True, how='left')
    weights = pd.DataFrame(0.0, index=balls.index, columns=[3, 4, 5, 6])
    for pos in [3, 4, 5, 6]:
        weights.loc[fielded, pos] = (balls.loc[fielded, 'hit_location'] == pos).astype(float)
        weights.loc[through_rows, pos] = through[pos].to_numpy()
    return weights, share

BASE_KEYS = ['bats_left', 'alignment_group', 'spray_bin']
weights_first, share_first = responsibility([1, 2, 3, 4, 5, 6], BASE_KEYS)
weights_final, share_final = responsibility([3, 4, 5, 6], BASE_KEYS + ['ev_band'])
weights_fallback, _ = responsibility([3, 4, 5, 6], BASE_KEYS)
empty = weights_final[3].isna()
weights_final.loc[empty] = weights_fallback.loc[empty]
print('balls fielded by an infielder', int(balls['hit_location'].isin([3, 4, 5, 6]).sum()), 'by the pitcher or catcher', int(balls['hit_location'].isin([1, 2]).sum()), 'reached the outfield', int(through_rows.sum()))
print('through-balls whose exit-velocity bin was empty (fallback used)', int(empty.sum()), 'still empty', int(weights_final[3].isna().sum()))
print('first try: shares for a right-handed batter, standard alignment (1 = P, 2 = C, 3 = 1B, 4 = 2B, 5 = 3B, 6 = SS)')
print(share_first.loc[(0, 0.0)].loc[[-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]].round(3))
print('final: shares for a right-handed batter, standard alignment, 100 mph and up')
print(share_final.xs((0, 0.0, 2), level=['bats_left', 'alignment_group', 'ev_band']).loc[[-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]].round(3))

####
# balls fielded by an infielder 155924 by the pitcher or catcher 14957 reached the outfield 35324
# through-balls whose exit-velocity bin was empty (fallback used) 51 still empty 0
# first try: shares for a right-handed batter, standard alignment (1 = P, 2 = C, 3 = 1B, 4 = 2B, 5 = 3B, 6 = SS)
#                1      2      3      4      5      6
# spray_bin
# -40.0      0.012  0.001  0.000  0.000  0.983  0.005
# -30.0      0.021  0.001  0.000  0.000  0.675  0.303
# -20.0      0.021  0.000  0.000  0.001  0.042  0.936
# -10.0      0.088  0.001  0.000  0.039  0.004  0.867
#  0.0       0.312  0.001  0.001  0.597  0.005  0.084
#  10.0      0.309  0.001  0.001  0.683  0.002  0.004
#  20.0      0.118  0.001  0.025  0.855  0.000  0.001
#  30.0      0.092  0.004  0.369  0.535  0.000  0.000
# final: shares for a right-handed batter, standard alignment, 100 mph and up
#                3      4      5      6
# spray_bin
# -40.0      0.000  0.000  0.993  0.007
# -30.0      0.000  0.000  0.518  0.482
# -20.0      0.000  0.000  0.013  0.987
# -10.0      0.000  0.051  0.004  0.945
#  0.0       0.000  0.921  0.008  0.071
#  10.0      0.000  0.986  0.000  0.014
#  20.0      0.019  0.981  0.000  0.000
#  30.0      0.425  0.575  0.000  0.000
####
# The first try gives the pitcher 31 percent of balls in the 0 and 10 degree bins, because pitchers field
# so many soft balls hit back up the middle. In the final shares, balls hit 100 mph and up in those bins
# belong to the second baseman (0.921 and 0.986).

# Step 5 Credit every fielder with outs above expected
# For each ball, p is the held-out probability of an out. The fielder responsible for an out gets
# 1 - p; the fielder responsible for a ball that was not an out gets -p. A through-ball splits its
# -p among the positions by the shares in Step 4, and the opportunity splits the same way. Summed
# over a season, that is outs made minus the outs an average defense would expect on his balls.
# The expected outs are the same whoever fielded the ball, so each position's league total need
# not be zero (Step 5 prints them); the last line puts every fielder against the average fielder at
# his position that season, as OAA does: outs above expected (OAE) = (outs - expected outs) minus
# his opportunities times the position's league rate of (outs - expected outs) per opportunity.
PLAYER_COLUMN = {3: 'fielder_3', 4: 'fielder_4', 5: 'fielder_5', 6: 'fielder_6'}

def fielder_seasons(weights):
    rows = []
    for pos in [3, 4, 5, 6]:
        w = weights[pos].fillna(0.0)
        rows.append(pd.DataFrame({'game_year': balls['game_year'], 'pos': pos, 'player_id': balls[PLAYER_COLUMN[pos]], 'opportunities': w, 'outs': w * balls['out'], 'expected_outs': w * balls['p_out']}))
    plays = pd.concat(rows, ignore_index=True)
    plays = plays[plays['opportunities'] > 0]
    f = plays.groupby(['game_year', 'pos', 'player_id'], as_index=False)[['opportunities', 'outs', 'expected_outs']].sum()
    f['oae_raw'] = f['outs'] - f['expected_outs']
    league = f.groupby(['game_year', 'pos'])[['oae_raw', 'opportunities']].transform('sum')
    f['league_rate'] = league['oae_raw'] / league['opportunities']
    f['oae'] = f['oae_raw'] - f['league_rate'] * f['opportunities']
    f['oae_per_100'] = 100 * f['oae'] / f['opportunities']
    return f

fielders_first = fielder_seasons(weights_first)
fielders = fielder_seasons(weights_final)
print('league total of outs minus expected outs by position, first try', fielders_first.groupby('pos')['oae_raw'].sum().round(1).to_dict())
print('league total of outs minus expected outs by position, final', fielders.groupby('pos')['oae_raw'].sum().round(1).to_dict())
print('fielder seasons', len(fielders))
print(fielders[(fielders['pos'] == 6) & (fielders['opportunities'] >= 150)].groupby('game_year')['oae'].describe().round(2))

####
# league total of outs minus expected outs by position, first try {3: -332.5, 4: 895.3, 5: -9.0, 6: 301.5}
# league total of outs minus expected outs by position, final {3: -331.7, 4: -34.3, 5: 99.3, 6: -433.7}
# fielder seasons 2483
#            count  mean    std    min   25%   50%   75%    max
# game_year
# 2023        33.0  1.54  10.20 -15.29 -4.91  1.05  4.50  28.04
# 2024        36.0  0.50   7.85 -13.12 -6.45 -0.85  6.22  19.33
# 2025        32.0  0.82   9.18 -18.36 -3.64  1.10  6.11  22.47
# 2026        32.0  0.95   9.40 -23.52 -4.68  1.68  6.63  25.11
####
# With the first try, second basemen as a group made 895 more outs than expected and first basemen 333 fewer;
# the final rule brings second base to -34. Each fielder is measured against his own position either way.

# Step 6 Check the credit against Statcast's OAA
# Statcast's OAA for infielders uses each fielder's starting position, the distance and time he has
# to reach the ball and the runner's speed. This model knows none of the fielder's position, so a
# good positioning call is credited to the fielder here and to the team's positioning in OAA. The
# two should still agree closely if the model has the right expected outs. Fielder seasons with at
# least 150 opportunities at the position, both versions of Step 4.
oaa = pd.concat([pd.read_csv(os.path.join(DATA, f'oaa_{s}_{pos}.csv')).assign(game_year=s, pos=pos) for s in SEASONS for pos in (3, 4, 5, 6)])
oaa = oaa.rename(columns={'outs_above_average': 'oaa'})[['player_id', 'game_year', 'pos', 'oaa', 'outs_above_average_lateral_toward3bline', 'outs_above_average_lateral_toward1bline', 'outs_above_average_infront', 'outs_above_average_behind']]
for label, f in [('first try', fielders_first), ('final', fielders)]:
    c = f.merge(oaa, on=['player_id', 'game_year', 'pos'], how='inner')
    c = c[c['opportunities'] >= 150]
    for pos in [3, 4, 5, 6]:
        g = c[c['pos'] == pos]
        print(label, 'position', pos, 'fielder seasons', len(g), 'r', round(np.corrcoef(g['oae'], g['oaa'])[0, 1], 3), 'slope of OAE on OAA', round(np.polyfit(g['oaa'], g['oae'], 1)[0], 3), 'SD OAE', round(g['oae'].std(), 2), 'SD OAA', round(g['oaa'].std(), 2))
check = fielders.merge(oaa, on=['player_id', 'game_year', 'pos'], how='inner')
check = check[check['opportunities'] >= 150]

####
# first try position 3 fielder seasons 78 r 0.71 slope of OAE on OAA 0.707 SD OAE 6.09 SD OAA 6.11
# first try position 4 fielder seasons 121 r 0.554 slope of OAE on OAA 0.558 SD OAE 7.92 SD OAA 7.85
# first try position 5 fielder seasons 108 r 0.642 slope of OAE on OAA 0.754 SD OAE 7.74 SD OAA 6.59
# first try position 6 fielder seasons 131 r 0.412 slope of OAE on OAA 0.41 SD OAE 8.75 SD OAA 8.79
# final position 3 fielder seasons 79 r 0.707 slope of OAE on OAA 0.704 SD OAE 6.05 SD OAA 6.08
# final position 4 fielder seasons 131 r 0.534 slope of OAE on OAA 0.582 SD OAE 8.28 SD OAA 7.6
# final position 5 fielder seasons 106 r 0.64 slope of OAE on OAA 0.748 SD OAE 7.75 SD OAA 6.63
# final position 6 fielder seasons 133 r 0.403 slope of OAE on OAA 0.419 SD OAE 9.07 SD OAA 8.73
####
# The final rule did not improve the agreement with OAA (shortstops 0.412 with the first try, 0.403 final).
# Agreement is best at first base (0.707) and worst at shortstop (0.403).

# Figure 2 Shortstops: this model's outs above expected against Statcast's OAA
fig, ax = plt.subplots(figsize=(7, 7))
g = check[check['pos'] == 6]
ax.scatter(g['oaa'], g['oae'], s=22, color=RED, alpha=0.7, edgecolor='none')
lim = [min(g['oaa'].min(), g['oae'].min()) - 2, max(g['oaa'].max(), g['oae'].max()) + 2]
ax.plot(lim, lim, color=GRAY, lw=1, ls='--')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('Statcast Outs Above Average (Savant leaderboard)')
ax.set_ylabel('Outs above expected (this model)')
ax.set_title(f'Shortstop seasons, 2023 to 2026 (n = {len(g)}, r = {np.corrcoef(g["oae"], g["oaa"])[0, 1]:.2f})')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'infield_expected_outs_fig2.png'))
plt.close(fig)

# Step 7 Is the credit a stable skill? Season-to-season correlation of each measure
# Same shortstops in consecutive seasons, at least 150 opportunities in both. Both measures are put
# on a per-opportunity scale (per 100 of this model's opportunities) so playing time does not drive
# the correlation.
check['oaa_per_100'] = 100 * check['oaa'] / check['opportunities']
pairs = check.merge(check.assign(game_year=check['game_year'] - 1), on=['player_id', 'pos', 'game_year'], suffixes=('', '_next'))
for pos in [4, 5, 6]:
    g = pairs[pairs['pos'] == pos]
    print('position', pos, 'pairs', len(g), 'year-to-year r, this model', round(np.corrcoef(g['oae_per_100'], g['oae_per_100_next'])[0, 1], 3), 'OAA', round(np.corrcoef(g['oaa_per_100'], g['oaa_per_100_next'])[0, 1], 3))

####
# position 4 pairs 53 year-to-year r, this model 0.501 OAA 0.43
# position 5 pairs 46 year-to-year r, this model 0.177 OAA 0.518
# position 6 pairs 68 year-to-year r, this model 0.298 OAA 0.362
####
# Second-base credit is a little more stable than OAA (0.501 against 0.43), shortstop a little less (0.298
# against 0.362), and third base much less (0.177 against 0.518).

# Step 8 Save the scored ground balls for shortstop_arm_tests.py
keep = ['game_pk', 'game_date', 'game_year', 'at_bat_number', 'home_team', 'away_team', 'inning_topbot', 'delta_run_exp', 'batter', 'stand', 'bats_left', 'events', 'hit_location', 'launch_speed', 'launch_angle', 'spray_angle', 'spray_bin', 'alignment', 'runner_1b', 'runner_2b', 'runner_3b', 'outs_when_up', 'batter_sprint_speed', 'out', 'p_out', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6']
scored = balls[keep].copy()
scored['share_ss'] = weights_final[6].fillna(0.0).to_numpy()
ss_rate = fielders[fielders['pos'] == 6].groupby('game_year').apply(lambda g: g['oae_raw'].sum() / g['opportunities'].sum())
scored['ss_league_rate'] = scored['game_year'].map(ss_rate)
print('shortstop league rate of outs minus expected per opportunity, by season', ss_rate.round(4).to_dict())
scored.to_csv(os.path.join(DATA, 'mlb_gb_scored.csv.gz'), index=False)
fielders.to_csv(os.path.join(DATA, 'mlb_fielder_seasons.csv'), index=False)
print('saved', len(scored), 'balls and', len(fielders), 'fielder seasons')
