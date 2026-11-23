#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:40:00 2026

@author: reyer
"""

# Florida State League shortstops: how much one season of ground balls says, and who stayed at shortstop
# Question: the Florida State League (FSL, Single-A) has had Statcast tracking in every park
# since 2021, and Baseball Savant publishes it. Using the same kind of expected-out model as
# infield_expected_outs.py, how reliable is one FSL season of a shortstop's ground-ball
# conversion, and did it predict which young FSL shortstops were still playing shortstop three
# to five years later?
# Data: Baseball Savant minors search (FSL only), every regular-season ground ball from 2021 to
# 2026; MLB Stats API birth dates and fielding games by position at every level.

import os
import json
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
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

RED, BLUE, GRAY, SILVER = '#E81828', '#6CACE4', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Download (or load cached) every FSL regular-season ground ball, 2021 to 2026
# The minors search takes the same parameters as the major league one plus hfLevel=A| (Single-A,
# which Savant tracks only in the FSL) and minors=true. Two windows per season keep each request
# well under the 25,000-row cap.
def get_file(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r:
            open(path, 'wb').write(r.read())
        time.sleep(2)
    return path

MINORS = 'https://baseballsavant.mlb.com/statcast-search-minors/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={season}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start}&game_date_lt={end}&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=ground%5C.%5C.ball%7C&hfFlag=&metric_1=&hfLevel=A%7C&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details&minors=true'
WINDOWS = [('03-01', '06-30'), ('07-01', '10-31')]
frames = []
for season in SEASONS:
    for start, end in WINDOWS:
        path = get_file(MINORS.format(season=season, start=f'{season}-{start}', end=f'{season}-{end}'), os.path.join(DATA, f'fsl_gb_{season}_{start}.csv'))
        chunk = pd.read_csv(path, low_memory=False)
        assert len(chunk) < 25000, path
        frames.append(chunk)
balls = pd.concat(frames, ignore_index=True)
balls = balls[(balls['game_type'] == 'R') & (balls['bb_type'] == 'ground_ball')].copy()
print('FSL ground balls', len(balls))
print(balls.groupby('game_year').size())
print(balls['home_team'].value_counts().sort_index().to_dict())

####
# FSL ground balls 79079
# game_year
# 2021    11722
# 2022    12966
# 2023    13645
# 2024    13489
# 2025    13598
# 2026    13659
# dtype: int64
# {'BRD': 8379, 'CLR': 7795, 'DBT': 7692, 'DUN': 7821, 'FTM': 7105, 'JUP': 8435, 'LAK': 7819, 'PMB': 8232, 'SLU': 7910, 'TAM': 7891}
####
# All ten FSL parks in every season, about 13,000 ground balls a season.

# Step 2 Clean the ground balls and define the outcome, as in infield_expected_outs.py
# One difference: exit velocity and launch angle are missing on about 12 percent of FSL ground
# balls, against under 1 percent in the majors, so those balls stay in and the model treats a
# missing value as information of its own (the trees in Step 3 send missing values down their own
# branch). Batter sprint speed is not published for the minors and is left out.
balls = balls[~balls['des'].str.contains('bunt', case=False, na=False)]
balls = balls.dropna(subset=['hc_x', 'hc_y']).copy()
OUT_EVENTS = ['field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'fielders_choice_out', 'triple_play']
balls['out'] = balls['events'].isin(OUT_EVENTS).astype(int)
balls['spray_angle'] = np.degrees(np.arctan2(balls['hc_x'] - 125.42, 198.27 - balls['hc_y']))
balls['bats_left'] = (balls['stand'] == 'L').astype(int)
balls['alignment'] = balls['if_fielding_alignment'].map({'Standard': 0, 'Infield shade': 1, 'Strategic': 2, 'Infield shift': 3})
balls['runner_1b'] = balls['on_1b'].notna().astype(int)
balls['runner_2b'] = balls['on_2b'].notna().astype(int)
balls['runner_3b'] = balls['on_3b'].notna().astype(int)
balls = balls.reset_index(drop=True)
print('ground balls after cleaning', len(balls), 'out rate', round(balls['out'].mean(), 4))
print('missing exit velocity', round(balls['launch_speed'].isna().mean(), 4), 'out rate when missing', round(balls.loc[balls['launch_speed'].isna(), 'out'].mean(), 4), 'when present', round(balls.loc[balls['launch_speed'].notna(), 'out'].mean(), 4))
print(balls.groupby('game_year')['if_fielding_alignment'].value_counts(normalize=True).round(3).unstack().to_dict('index'))

####
# ground balls after cleaning 77524 out rate 0.6954
# missing exit velocity 0.1182 out rate when missing 0.7329 when present 0.6904
# {2021: {'Infield shade': 0.059, 'Infield shift': 0.089, 'Standard': 0.772, 'Strategic': 0.08}, 2022: {'Infield shade': 0.07, 'Infield shift': 0.009, 'Standard': 0.716, 'Strategic': 0.205}, 2023: {'Infield shade': 0.06, 'Infield shift': nan, 'Standard': 0.855, 'Strategic': 0.085}, 2024: {'Infield shade': 0.1, 'Infield shift': nan, 'Standard': 0.817, 'Strategic': 0.083}, 2025: {'Infield shade': 0.079, 'Infield shift': nan, 'Standard': 0.845, 'Strategic': 0.076}, 2026: {'Infield shade': 0.112, 'Infield shift': nan, 'Standard': 0.772, 'Strategic': 0.116}}
####
# The out rate is higher when exit velocity is missing (73.3 percent against 69.0), so those balls are not a
# random sample, and the model keeps them as a value of their own. Savant labels 8.9 percent of 2021 ground
# balls as hit into a shift, 0.9 percent in 2022, and none after.

# Step 3 Expected-out model with a leave-one-season-out backtest
# Same model class and settings as the major league model; each FSL season is predicted by a model
# trained on the other five.
FEATURE_SETS = {'league rate': [],
                'exit velocity, launch angle': ['launch_speed', 'launch_angle'],
                '+ spray angle, batter side': ['launch_speed', 'launch_angle', 'spray_angle', 'bats_left'],
                '+ runners, outs, alignment': ['launch_speed', 'launch_angle', 'spray_angle', 'bats_left', 'runner_1b', 'runner_2b', 'runner_3b', 'outs_when_up', 'alignment']}

def fit_model(train, features):
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    return model.fit(train[features], train['out'])

rows = []
for name, features in FEATURE_SETS.items():
    for season in SEASONS:
        train, test = balls[balls['game_year'] != season], balls[balls['game_year'] == season]
        p = fit_model(train, features).predict_proba(test[features])[:, 1] if features else np.full(len(test), train['out'].mean())
        p = np.clip(p, 1e-6, 1 - 1e-6)
        rows.append((name, season, len(test), np.mean((p - test['out']) ** 2), -np.mean(test['out'] * np.log(p) + (1 - test['out']) * np.log(1 - p))))
        if name == '+ runners, outs, alignment':
            balls.loc[test.index, 'p_out'] = p
backtest = pd.DataFrame(rows, columns=['inputs', 'held_out_season', 'balls', 'brier', 'log_loss'])
print(backtest.groupby('inputs', sort=False).apply(lambda g: pd.Series({'brier': np.average(g['brier'], weights=g['balls']), 'log_loss': np.average(g['log_loss'], weights=g['balls'])})).round(5))
balls['p_decile'] = pd.qcut(balls['p_out'], 10, labels=False)
print(balls.groupby('p_decile').agg(predicted=('p_out', 'mean'), observed=('out', 'mean')).round(4).T)

####
#                                brier  log_loss
# inputs
# league rate                  0.21184   0.61477
# exit velocity, launch angle  0.19111   0.56705
# + spray angle, batter side   0.17055   0.51834
# + runners, outs, alignment   0.16884   0.51396
# p_decile        0       1       2       3       4       5       6       7       8       9
# predicted  0.2309  0.4530  0.5888  0.6837  0.7458  0.7895  0.8232  0.8493  0.8736  0.9107
# observed   0.2194  0.4638  0.5974  0.6886  0.7465  0.7855  0.8239  0.8515  0.8742  0.9037
####
# Spray angle and batter side again give the largest step after exit velocity (Brier 0.19111 to 0.17055).
# The model is within 1.2 points of the observed out rate in every tenth.

# Step 4 Shortstop responsibility and credit, as in Steps 4 and 5 of infield_expected_outs.py (final rule)
# Infielders only, shares by spray-angle bin, batter side, alignment and exit-velocity band (a fourth
# band for balls with no exit velocity), falling back to the bins without exit velocity when a bin is
# empty; balls fielded by the pitcher or catcher are left out. Each shortstop is then measured against
# the average FSL shortstop that season.
balls['spray_bin'] = np.clip(np.floor(balls['spray_angle'] / 2) * 2, -50, 50)
balls['alignment_group'] = balls['alignment'].fillna(0)
balls['ev_band'] = np.select([balls['launch_speed'].isna(), balls['launch_speed'] < 85, balls['launch_speed'] < 100], [3, 0, 1], 2)
through_rows = ~balls['hit_location'].isin([1, 2, 3, 4, 5, 6])
fielded = balls['hit_location'].isin([3, 4, 5, 6])

def ss_share(keys):
    share = balls[fielded].groupby(keys)['hit_location'].value_counts(normalize=True).unstack(fill_value=0.0)
    share.columns = [int(c) for c in share.columns]
    share = share.reindex(columns=[3, 4, 5, 6], fill_value=0.0)
    return balls.loc[through_rows, keys].merge(share, left_on=keys, right_index=True, how='left')[6].to_numpy()

BASE_KEYS = ['bats_left', 'alignment_group', 'spray_bin']
through_share = ss_share(BASE_KEYS + ['ev_band'])
fallback = ss_share(BASE_KEYS)
print('through-balls with an empty exit-velocity bin (fallback used)', int(np.isnan(through_share).sum()))
through_share = np.where(np.isnan(through_share), fallback, through_share)
balls['share_ss'] = 0.0
balls.loc[fielded, 'share_ss'] = (balls.loc[fielded, 'hit_location'] == 6).astype(float)
balls.loc[through_rows, 'share_ss'] = np.nan_to_num(through_share)
season_rate = balls.groupby('game_year').apply(lambda g: (g['share_ss'] * (g['out'] - g['p_out'])).sum() / g['share_ss'].sum())
balls['oae_ss'] = balls['share_ss'] * (balls['out'] - balls['p_out'] - balls['game_year'].map(season_rate))
ss_seasons = balls[balls['share_ss'] > 0].groupby(['fielder_6', 'game_year']).agg(opportunities=('share_ss', 'sum'), oae=('oae_ss', 'sum'), games=('game_pk', 'nunique')).reset_index().rename(columns={'fielder_6': 'player_id'})
ss_seasons['oae_per_100'] = 100 * ss_seasons['oae'] / ss_seasons['opportunities']
print('FSL shortstop rate of outs minus expected per opportunity, by season', season_rate.round(4).to_dict())
print('FSL shortstop seasons', len(ss_seasons), 'with 150+ opportunities', int((ss_seasons['opportunities'] >= 150).sum()))
print(ss_seasons[ss_seasons['opportunities'] >= 150][['opportunities', 'games', 'oae', 'oae_per_100']].describe().round(2))

####
# through-balls with an empty exit-velocity bin (fallback used) 148
# FSL shortstop rate of outs minus expected per opportunity, by season {2021: -0.0213, 2022: 0.002, 2023: -0.0182, 2024: -0.0056, 2025: -0.0099, 2026: -0.0178}
# FSL shortstop seasons 454 with 150+ opportunities 42
#        opportunities   games    oae  oae_per_100
# count          42.00   42.00  42.00        42.00
# mean          212.65   69.31   1.11         0.37
# std            43.80   15.35   9.73         4.59
# min           150.19   43.00 -18.40        -9.29
# 25%           177.30   58.25  -5.40        -2.80
# 50%           210.31   66.00   0.36         0.17
# 75%           244.29   80.00   7.70         3.58
# max           365.08  113.00  23.61         8.52
####
# Only 42 FSL shortstop seasons in six years reach 150 chances (median 210).

# Step 5 How reliable is one season? Split-half correlation, FSL and the majors side by side
# Each shortstop season's games are dealt at random into two halves; the correlation of outs above
# expected per 100 chances between the halves, stepped up to a full season with the Spearman-Brown
# formula (2r / (1 + r)), is the reliability of one season. The major league numbers come from the
# scored balls that infield_expected_outs.py saves. Shortstop seasons with at least 150
# opportunities; the result is the mean over 200 random deals, with the spread across deals.
def split_half(plays, min_opp=150, reps=200):
    plays = plays[plays['share_ss'] > 0]
    totals = plays.groupby(['fielder_6', 'game_year'])['share_ss'].sum()
    keep = totals[totals >= min_opp].index
    plays = plays.set_index(['fielder_6', 'game_year']).loc[keep].reset_index()
    games = plays[['fielder_6', 'game_year', 'game_pk']].drop_duplicates()
    out = []
    for rep in range(reps):
        games = games.assign(half=rng_split.integers(0, 2, len(games)))
        p = plays.merge(games, on=['fielder_6', 'game_year', 'game_pk'])
        sums = p.groupby(['fielder_6', 'game_year', 'half'])[['oae_ss', 'share_ss']].sum()
        h = (100 * sums['oae_ss'] / sums['share_ss']).unstack().dropna()
        r = np.corrcoef(h[0], h[1])[0, 1]
        out.append((r, 2 * r / (1 + r)))
    out = np.array(out)
    return len(keep), totals.loc[keep].median(), out[:, 0].mean(), out[:, 1].mean(), np.percentile(out[:, 1], 5), np.percentile(out[:, 1], 95)

rng_split = np.random.default_rng(7)
mlb = pd.read_csv(os.path.join(DATA, 'mlb_gb_scored.csv.gz'), usecols=['game_pk', 'game_year', 'fielder_6', 'share_ss', 'out', 'p_out', 'ss_league_rate'])
mlb['oae_ss'] = mlb['share_ss'] * (mlb['out'] - mlb['p_out'] - mlb['ss_league_rate'])
for label, plays in [('FSL 2021 to 2026', balls), ('MLB 2023 to 2026', mlb)]:
    n, median_opp, r_half, rel, lo, hi = split_half(plays)
    print(f'{label}: shortstop seasons {n}, median opportunities {median_opp:.0f}, split-half r {r_half:.3f}, one-season reliability {rel:.3f} (90% of deals {lo:.3f} to {hi:.3f})')

####
# FSL 2021 to 2026: shortstop seasons 42, median opportunities 210, split-half r 0.451, one-season reliability 0.616 (90% of deals 0.465 to 0.738)
# MLB 2023 to 2026: shortstop seasons 133, median opportunities 326, split-half r 0.281, one-season reliability 0.435 (90% of deals 0.301 to 0.542)
####
# One FSL season (median 210 chances) is more reliable than one MLB season (median 326 chances): 0.616 against
# 0.435. With fewer chances and a higher reliability, FSL shortstops must differ from each other more.

# Step 6 Ages and what happened next, from the MLB Stats API
# The cohort is every FSL shortstop season from 2021 to 2023 with at least 100 opportunities by a
# player aged 21 or younger (baseball age: age on June 30 of that season; the FSL is the first
# full-season level, and most of its shortstops are 19 to 21). For each player the Stats API gives
# fielding games by position at every level, majors included. He "stayed at shortstop" if
# shortstop was at least half his games in the field in 2025 and 2026 combined, "moved" if he played
# in the field but less than half at shortstop, and "no games" if he played no affiliated games in
# 2025 or 2026 (released, retired, injured, or playing independent ball).
def get_json(url, path):
    return json.load(open(get_file(url, path)))

os.makedirs(os.path.join(DATA, 'statsapi'), exist_ok=True)
cohort = ss_seasons[(ss_seasons['game_year'] <= 2023) & (ss_seasons['opportunities'] >= 100)].copy()
people = []
for pid in cohort['player_id'].unique():
    person = get_json(f'https://statsapi.mlb.com/api/v1/people/{pid}', os.path.join(DATA, 'statsapi', f'person_{pid}.json'))['people'][0]
    fielding = get_json(f'https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=yearByYear&group=fielding&leagueListId=mlb_milb', os.path.join(DATA, 'statsapi', f'fielding_{pid}.json'))
    games = {}
    for s in fielding.get('stats', []):
        for sp in s['splits']:
            if sp.get('sport', {}).get('id') == 21 or 'team' not in sp:
                continue
            season, position = int(sp['season']), sp['position']['abbreviation']
            games[(season, position)] = games.get((season, position), 0) + int(sp['stat'].get('games', 0))
    later = {pos: g for (season, pos), g in games.items() if season in (2025, 2026) and pos != 'DH'}
    field_games, ss_games = sum(later.values()), later.get('SS', 0)
    people.append({'player_id': pid, 'name': person['fullName'], 'birth_date': person.get('birthDate'), 'later_field_games': field_games, 'later_ss_games': ss_games,
                   'later_main_position': max(later, key=later.get) if later else None})
people = pd.DataFrame(people)
cohort = cohort.merge(people, on='player_id')
birth = pd.to_datetime(cohort['birth_date'])
cohort['age'] = cohort['game_year'] - birth.dt.year - ((birth.dt.month > 6)).astype(int)
cohort = cohort[cohort['age'] <= 21].copy()
cohort['outcome'] = np.select([cohort['later_field_games'] == 0, cohort['later_ss_games'] >= 0.5 * cohort['later_field_games']], ['no games', 'stayed at shortstop'], 'moved')
print('cohort shortstop seasons', len(cohort), 'players', cohort['player_id'].nunique())
print(cohort.groupby('age').size().to_dict())
print(pd.crosstab(cohort.drop_duplicates('player_id')['age'], cohort.drop_duplicates('player_id')['outcome']))
print(cohort.drop_duplicates('player_id')['outcome'].value_counts().to_dict())
print(cohort[cohort['outcome'] == 'moved'].drop_duplicates('player_id')['later_main_position'].value_counts().to_dict())

####
# cohort shortstop seasons 30 players 30
# {19: 17, 20: 10, 21: 3}
# outcome  moved  no games  stayed at shortstop
# age
# 19          10         3                    4
# 20           4         2                    4
# 21           2         1                    0
# {'moved': 16, 'stayed at shortstop': 8, 'no games': 6}
# {'2B': 6, '3B': 5, 'SS': 3, 'CF': 2}
####
# 30 players: 8 stayed at shortstop, 16 moved (mostly to second and third base), and 6 played no affiliated
# games in 2025 or 2026.

# Step 7 Did one FSL season of outs above expected predict who stayed at shortstop?
# One row per player: his best-sampled cohort season (most opportunities). Among players who were
# still playing, the area under the ROC curve (AUC: the chance that a randomly chosen player who
# stayed had a higher FSL rate than a randomly chosen player who moved; 0.5 is no information),
# with a 90 percent bootstrap interval. The same for age (younger is the prediction) and for the
# playing time at shortstop that season (opportunities), which carries the club's own view of him.
player = cohort.sort_values('opportunities', ascending=False).drop_duplicates('player_id').copy()
still = player[player['outcome'] != 'no games'].copy()
still['stayed'] = (still['outcome'] == 'stayed at shortstop').astype(int)

def auc(score, label):
    pos, neg = score[label == 1].to_numpy(), score[label == 0].to_numpy()
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))

rng_boot = np.random.default_rng(11)
for label, score in [('FSL outs above expected per 100', still['oae_per_100']), ('FSL outs above expected (total)', still['oae']), ('FSL opportunities at shortstop (playing time)', still['opportunities']), ('younger age', -still['age'])]:
    a = auc(score, still['stayed'])
    boot = []
    for _ in range(2000):
        i = rng_boot.integers(0, len(still), len(still))
        if still['stayed'].iloc[i].nunique() == 2:
            boot.append(auc(score.iloc[i], still['stayed'].iloc[i]))
    print(f'{label}: AUC {a:.3f} (90% {np.percentile(boot, 5):.3f} to {np.percentile(boot, 95):.3f}); stayed {int(still["stayed"].sum())}, moved {int((1 - still["stayed"]).sum())}')
print(player.groupby('outcome')[['oae_per_100', 'oae', 'opportunities', 'age']].mean().round(2))
print(player.sort_values(['outcome', 'oae_per_100'], ascending=[True, False])[['name', 'game_year', 'age', 'opportunities', 'oae_per_100', 'outcome', 'later_main_position']].round(1).to_string(index=False))

####
# FSL outs above expected per 100: AUC 0.852 (90% 0.701 to 0.978); stayed 8, moved 16
# FSL outs above expected (total): AUC 0.883 (90% 0.731 to 0.992); stayed 8, moved 16
# FSL opportunities at shortstop (playing time): AUC 0.656 (90% 0.426 to 0.872); stayed 8, moved 16
# younger age: AUC 0.469 (90% 0.283 to 0.644); stayed 8, moved 16
#                      oae_per_100    oae  opportunities    age
# outcome
# moved                      -1.51  -2.36         174.85  19.50
# no games                   -5.30  -6.53         139.82  19.67
# stayed at shortstop         5.66  11.50         219.46  19.50
#                name  game_year  age  opportunities  oae_per_100             outcome later_main_position
#         Nasim Nuñez       2021   20          128.4          9.2               moved                  2B
#       Manuel Beltre       2023   19          204.3          3.2               moved                  2B
#    Danny De Andrade       2023   19          253.9          2.0               moved                  2B
#         Erick Brito       2022   20          268.3          1.7               moved                  SS
#    Orelvis Martinez       2021   19          124.2          1.7               moved                  2B
#        Gage Workman       2021   21          124.9          0.9               moved                  SS
# Kevin Villavicencio       2023   19          106.8          0.5               moved                  2B
#      Maikol Escotto       2021   19          212.7         -0.3               moved                  SS
#    Alexander Vargas       2022   20          245.1         -2.2               moved                  3B
#    Cristian Santana       2023   19          107.9         -2.9               moved                  3B
#         Jared Serna       2023   21          126.1         -3.0               moved                  2B
#       Victor Acosta       2023   19          244.1         -3.6               moved                  3B
#   Lizandro Espinoza       2023   20          234.0         -4.2               moved                  CF
#       Kahlil Watson       2022   19          131.9         -8.3               moved                  CF
#       Junior Tilien       2022   19          142.4         -8.8               moved                  3B
#       Jett Williams       2023   19          142.6        -10.0               moved                  3B
#     Estiven Machado       2022   19          139.6          2.1            no games                 NaN
#      Manuel Sequera       2022   19          160.6          0.3            no games                 NaN
#         Luis García       2021   20          157.1         -1.6            no games                 NaN
#        Keoni Cavaco       2021   20          174.7         -9.0            no games                 NaN
#        Dariel Lopez       2021   19          105.0        -11.6            no games                 NaN
#       Mack Chambers       2021   21          102.0        -12.0            no games                 NaN
#       Anthony Volpe       2021   20          102.2         11.7 stayed at shortstop                  SS
#        Javier Rivas       2023   20          134.9          8.5 stayed at shortstop                  SS
#         Noah Miller       2022   19          276.9          8.5 stayed at shortstop                  SS
#     Tsung-Che Cheng       2022   20          265.6          7.2 stayed at shortstop                  SS
#        Bryan Rincon       2023   19          236.6          6.3 stayed at shortstop                  SS
#        Jeremy Rivas       2022   19          365.1          2.9 stayed at shortstop                  SS
#      Brenny Escanio       2023   20          194.8          0.9 stayed at shortstop                  SS
#          Masyn Winn       2021   19          179.6         -0.7 stayed at shortstop                  SS
####
# The FSL measure separates the players who stayed from those who moved (AUC 0.852) better than playing time
# does (0.656), and age carries nothing (0.469). Those who stayed averaged +5.66 per 100 chances, those who
# moved -1.51.

# Figure 1 FSL outs above expected per 100 chances, by what happened next
fig, ax = plt.subplots(figsize=(9, 5.5))
order = ['stayed at shortstop', 'moved', 'no games']
for i, (outcome, color) in enumerate(zip(order, [RED, BLUE, GRAY])):
    g = player[player['outcome'] == outcome]
    jitter = rng_boot.uniform(-0.15, 0.15, len(g))
    ax.scatter(np.full(len(g), i) + jitter, g['oae_per_100'], s=28, color=color, alpha=0.75, edgecolor='none')
    ax.plot([i - 0.25, i + 0.25], [g['oae_per_100'].mean()] * 2, color='black', lw=2)
ax.axhline(0, color=GRAY, lw=0.8, ls='--')
ax.set_xticks(range(3))
ax.set_xticklabels([f'{o}\n(n = {(player["outcome"] == o).sum()})' for o in order])
ax.set_ylabel('FSL outs above expected per 100 chances')
ax.set_title('FSL shortstops aged 21 or younger, 2021 to 2023, by 2025 to 2026 position')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fsl_shortstops_fig1.png'))
plt.close(fig)

# Step 8 How the shortstop's two measurable tools age, in the majors
# Projecting a 19-year-old means projecting his body as well as his play. Savant publishes sprint
# speed (with age) from 2016 and arm strength by position from 2020. The delta method: for the same
# player in consecutive seasons, the change from age a to a + 1, averaged over players (sprint speed:
# at least 50 competitive runs in both seasons; shortstop arm strength: at least 50 throws from
# shortstop in both). Players who keep their jobs are over-represented, which flatters the decline
# a little.
SPRINT = 'https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season={season}&max_season={season}&position=&team=&min=0&csv=true'
ARM = 'https://baseballsavant.mlb.com/leaderboard/arm-strength?type=player&year={season}&minThrows=1&pos=&team=&csv=true'
sprint = pd.concat([pd.read_csv(get_file(SPRINT.format(season=s), os.path.join(DATA, f'sprint_{s}.csv'))).assign(season=s) for s in range(2016, 2027)])
sprint['competitive_runs'] = pd.to_numeric(sprint['competitive_runs'], errors='coerce')
arm = pd.concat([pd.read_csv(get_file(ARM.format(season=s), os.path.join(DATA, f'arm_{s}.csv'))).assign(season=s) for s in range(2020, 2027)])
arm = arm.merge(sprint[['player_id', 'season', 'age']], on=['player_id', 'season'], how='inner')
sp = sprint[sprint['competitive_runs'] >= 50][['player_id', 'season', 'age', 'sprint_speed']]
sp_pairs = sp.merge(sp.assign(season=sp['season'] - 1), on=['player_id', 'season'], suffixes=('', '_next'))
sp_pairs['change'] = sp_pairs['sprint_speed_next'] - sp_pairs['sprint_speed']
ar = arm[(arm['total_throws_ss'] >= 50) & arm['arm_ss'].notna()][['player_id', 'season', 'age', 'arm_ss']]
ar_pairs = ar.merge(ar.assign(season=ar['season'] - 1), on=['player_id', 'season'], suffixes=('', '_next'))
ar_pairs['change'] = ar_pairs['arm_ss_next'] - ar_pairs['arm_ss']
aging = pd.DataFrame({'sprint_pairs': sp_pairs.groupby('age').size(), 'sprint_change_ft_s': sp_pairs.groupby('age')['change'].mean(),
                      'arm_pairs': ar_pairs.groupby('age').size(), 'arm_change_mph': ar_pairs.groupby('age')['change'].mean()}).loc[21:33]
print(aging.round(3))
print('sprint speed, mean change per season at ages 22 to 26', round(sp_pairs[sp_pairs['age'].between(22, 26)]['change'].mean(), 3), 'pairs', int(sp_pairs['age'].between(22, 26).sum()))
print('shortstop arm, mean change per season at ages 22 to 26', round(ar_pairs[ar_pairs['age'].between(22, 26)]['change'].mean(), 3), 'pairs', int(ar_pairs['age'].between(22, 26).sum()))

####
#      sprint_pairs  sprint_change_ft_s  arm_pairs  arm_change_mph
# age
# 21             37              -0.157        7.0          -0.343
# 22             90              -0.156       11.0           0.227
# 23            163              -0.124       20.0          -0.595
# 24            237              -0.219       20.0          -0.170
# 25            285              -0.119       18.0          -0.211
# 26            317              -0.184       26.0          -0.492
# 27            301              -0.146       26.0          -0.269
# 28            253              -0.145       21.0          -0.600
# 29            242              -0.157       18.0          -0.389
# 30            239              -0.162       20.0           0.210
# 31            182              -0.108       14.0           1.029
# 32            155              -0.167        9.0          -1.067
# 33            117              -0.122        4.0          -0.150
# sprint speed, mean change per season at ages 22 to 26 -0.163 pairs 1092
# shortstop arm, mean change per season at ages 22 to 26 -0.309 pairs 95
####
# Sprint speed falls steadily, about 0.16 ft/s a season at every age from 21. Arm strength at shortstop did not
# rise in the early 20s (-0.31 mph a season at 22 to 26), on only 95 player pairs.

# Figure 2 Cumulative change with age, sprint speed and shortstop arm strength
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, n_col, label, unit, color in [(axes[0], 'sprint_change_ft_s', 'sprint_pairs', 'Sprint speed', 'ft/s', BLUE), (axes[1], 'arm_change_mph', 'arm_pairs', 'Arm strength at shortstop', 'mph', RED)]:
    a = aging[aging[n_col] >= 15]
    ages = list(a.index) + [a.index.max() + 1]
    ax.plot(ages, np.concatenate([[0], a[col].cumsum()]), '-o', color=color, lw=2)
    ax.axhline(0, color=GRAY, lw=0.8, ls='--')
    ax.set_xlabel('Age')
    ax.set_ylabel(f'Change since age {ages[0]} ({unit})')
    ax.set_title(label)
fig.suptitle('Same players from one season to the next, MLB (sprint speed 2016 to 2026, arm strength 2020 to 2026)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fsl_shortstops_fig2.png'))
plt.close(fig)
