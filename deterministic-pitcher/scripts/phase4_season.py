#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:30:00 2026

@author: reyer
"""

import pandas as pd
import gzip
import os
import io
import shutil
import glob
import pathlib
import json
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import game_engine as engine

# Phase 4 of the deterministic pitcher project (the plan is in ../PLAN.md)
# Phase 2 bracketed D's value with the lock rule at 2.56 wins a season (rest valued only through back-to-back outings avoided) to 3.99 (every top-reliever appearance D frees redeployed perfectly). This phase pins the value of rest down by simulating whole seasons: a 13-pitcher staff, a manager who uses it the way real managers do, relievers who get tired and become unavailable, with and without D.
# The value of rest is separated from the locked wins by running the same comparison twice: once with fatigue (availability limits and the back-to-back penalty) and once with every pitcher always fresh. The locked wins are in both; the rest is the difference between them.
# Not modelled: injuries (no usable estimate of how rest changes injury risk), call-ups and options (real teams swap tired relievers for fresh ones from Triple-A), and anything a manager would change about the starters because D exists. So the value of rest here is what better availability within one 13-man staff buys, nothing more.
# Data: Retrosheet play-by-play and per-pitcher game lines, 2022 to 2025, regular season; the engine and its inputs from Phase 3.
# The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

project_dir = pathlib.Path(__file__).resolve().parent.parent
csv_dir = project_dir / 'data' / 'retrosheet' / 'csv'
fig_dir = project_dir / 'figures'
results_dir = project_dir / 'results'
years = [2022, 2023, 2024, 2025]
tiers = ['low', 'middle', 'high']
random.seed(20261014)

surface = '#fcfcfb'
ink = '#0b0b0b'
ink_secondary = '#52514e'
ink_muted = '#898781'
plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface, 'axes.edgecolor': ink_muted, 'axes.labelcolor': ink_secondary, 'xtick.color': ink_muted, 'ytick.color': ink_muted, 'text.color': ink, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


# Step 1 Read the engine inputs, the play-by-play and the per-pitcher game lines
inputs = engine.load_inputs(results_dir / 'phase3_engine_inputs.pkl')
appearances = pd.read_csv(results_dir / 'phase2_appearances.csv')
plays_list = []
lines_list = []
for year in years:
    p = pd.read_csv(csv_dir / str(year) / f'{year}plays.csv', usecols=['gid', 'inning', 'top_bot', 'pitcher', 'pa', 'nump', 'single', 'double', 'triple', 'hr', 'walk', 'hbp', 'k', 'roe', 'xi', 'gametype'], low_memory=False)
    plays_list.append(p[p['gametype'] == 'regular'].drop(columns='gametype'))
    pt = pd.read_csv(csv_dir / str(year) / f'{year}pitching.csv', usecols=['gid', 'id', 'team', 'p_seq', 'p_ipouts', 'p_bfp', 'p_r', 'date', 'number', 'gametype'])
    pt = pt[pt['gametype'] == 'regular'].drop(columns='gametype')
    pt['year'] = year
    lines_list.append(pt)
plays = pd.concat(plays_list, ignore_index=True)
lines = pd.concat(lines_list, ignore_index=True)
plays = plays[plays['gid'].isin(appearances['gid'].unique())]
lines = lines.merge(appearances[['gid', 'id', 'role_detail']], on=['gid', 'id'], how='inner')
pa = plays[plays['pa'] == 1].copy()
pa['outcome'] = np.select([pa['hr'] == 1, pa['triple'] == 1, pa['double'] == 1, pa['single'] == 1, pa['walk'] == 1, (pa['hbp'] == 1) | (pa['xi'] == 1), pa['k'] == 1, pa['roe'] == 1], ['HR', '3B', '2B', '1B', 'BB', 'HBP', 'K', 'ROE'], 'OUT')
pa = pa.merge(appearances[['gid', 'id', 'role_detail']].rename(columns={'id': 'pitcher'}), on=['gid', 'pitcher'], how='left')
print('plate appearances:', len(pa), ' pitcher lines:', len(lines))
####
# plate appearances: 730295  pitcher lines: 82964
####
# The same games as Phases 2 and 3.


# Step 2 Starters by time through the order, and how long real starters last
# A starter's results get worse each time the lineup comes round again. Phase 3 found the engine's scoring too flat across innings because it used one rate for every starter's batter; here starters get one rate for each time through (the 1st to 9th batter he faces, the 10th to 18th, the 19th on).
starters_pa = pa[pa['role_detail'] == 'starter'].copy()
starters_pa['batter_number'] = starters_pa.groupby(['gid', 'pitcher']).cumcount()
starters_pa['time_through'] = (starters_pa['batter_number'] // 9 + 1).clip(upper=3)
for t in [1, 2, 3]:
    inputs['pitcher_rates'][f'starter|{t}'] = starters_pa.loc[starters_pa['time_through'] == t, 'outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES).values
tto = pd.DataFrame({f'time through {t}': inputs['pitcher_rates'][f'starter|{t}'] for t in [1, 2, 3]}, index=engine.OUTCOMES).T
tto['batters'] = starters_pa['time_through'].value_counts().sort_index().values
print(tto.round(4).to_string())
starter_lines = lines[lines['role_detail'] == 'starter']
starter_pitches = pa[pa['role_detail'] == 'starter'].groupby(['gid', 'pitcher'])['nump'].sum()
real_starter = {'outs per start': starter_lines['p_ipouts'].mean(), 'batters per start': starter_lines['p_bfp'].mean(), 'pitches per start': starter_pitches.mean(), 'share of starts reaching 100 pitches': (starter_pitches >= 100).mean()}
print({k: round(float(v), 3) for k, v in real_starter.items()})
####
#                      K      BB     HBP      1B      2B      3B      HR     ROE     OUT  batters
# time through 1  0.2387  0.0782  0.0105  0.1403  0.0428  0.0034  0.0309  0.0060  0.4492   172478
# time through 2  0.2102  0.0752  0.0105  0.1442  0.0455  0.0039  0.0328  0.0058  0.4718   163559
# time through 3  0.1957  0.0775  0.0103  0.1502  0.0480  0.0040  0.0352  0.0062  0.4730    89298
# {'outs per start': 15.563, 'batters per start': 21.931, 'pitches per start': 83.53, 'share of starts reaching 100 pitches': 0.105}
####
# Each time through the order a starter strikes out fewer batters (23.9, 21.0, 19.6 percent) and allows more home runs (3.1, 3.3, 3.5 percent). Real starters average 15.56 outs and 83.5 pitches a start, and only 10.5 percent reach 100 pitches.


# Step 3 How much of the top relievers' advantage is real: the same pitchers one season later
# Phase 2 picked each team's top relievers by how high-leverage their usage was, and they allowed 0.0264 fewer runs per batter than the rest. Some of that is luck that earned them the role. The test: take the pitchers who were top relievers (or other relievers) in one season and compare them the next season, whatever their role then.
relief_lines = lines[lines['role_detail'].isin(['top reliever', 'other reliever'])]
by_season = relief_lines.groupby(['year', 'id']).agg(role=('role_detail', lambda r: 'top reliever' if (r == 'top reliever').any() else 'other reliever'), runs=('p_r', 'sum'), batters=('p_bfp', 'sum')).reset_index()
nxt = by_season[['year', 'id', 'runs', 'batters']].copy()
nxt['year'] = nxt['year'] - 1
paired = by_season.merge(nxt, on=['year', 'id'], suffixes=('', '_next'))
persistence_rows = []
for role in ['top reliever', 'other reliever']:
    g = paired[paired['role'] == role]
    persistence_rows.append([role, len(g), g['runs'].sum() / g['batters'].sum(), g['runs_next'].sum() / g['batters_next'].sum()])
persistence = pd.DataFrame(persistence_rows, columns=['role this season', 'pitchers', 'runs per batter this season', 'runs per batter next season'])
print(persistence.round(4).to_string(index=False))
gap_this = persistence.iloc[1, 2] - persistence.iloc[0, 2]
gap_next = persistence.iloc[1, 3] - persistence.iloc[0, 3]
persistence_share = gap_next / gap_this
print(f'gap this season {gap_this:.4f}, next season {gap_next:.4f}: {persistence_share:.2f} of the gap persists')
# The engine's gap with the raw rates is 0.0281 (Phase 3), against 0.0264 real. Shrink the top relievers' rates toward the others' so the engine's gap equals the persistent part of the real one.
shrink_weight = persistence_share * 0.0264 / 0.0281
inputs['pitcher_rates']['top reliever|shrunk'] = inputs['pitcher_rates']['other reliever'] + shrink_weight * (inputs['pitcher_rates']['top reliever'] - inputs['pitcher_rates']['other reliever'])
print(f'shrink weight on the top-against-other difference: {shrink_weight:.3f}')
####
# role this season  pitchers  runs per batter this season  runs per batter next season
#     top reliever       243                       0.0969                       0.1091
#   other reliever      1001                       0.1178                       0.1179
# gap this season 0.0209, next season 0.0088: 0.42 of the gap persists
# shrink weight on the top-against-other difference: 0.395
####
# Most of the top relievers' advantage is not talent. The next season, the same top relievers allow 0.1091 runs per batter against 0.0969, while the other relievers do not move at all (0.1178 to 0.1179), so only 42 percent of the gap persists. Phase 2's ceiling (1.43 wins from redeploying freed top-reliever appearances) took the whole gap as talent; this is the first of two reasons it was too high. The simulation uses the persistent part (shrink weight 0.395), with the full gap as a sensitivity check in Step 10.


# Step 4 The back-to-back penalty in the engine's terms
# Phase 2 measured that a reliever pitching for the second day running allows 0.0085 more runs per batter (95% interval 0.0039 to 0.0130). In the engine a tired pitcher has every way of reaching base multiplied by a factor; find the factor that gives each of those three penalties.
def runs_per_batter(key, n=60000, seed=11):
    random.seed(seed)
    levels = engine.game_factor_levels(inputs['game_factor_sd'])
    runs_total = 0
    batters_total = 0
    for _ in range(n):
        runs, _, batters, _ = engine.half_inning(inputs, random.choice(tiers), random.randint(0, 1), random.randint(0, 8), key, factor=random.choice(levels))
        runs_total += runs
        batters_total += batters
    return runs_total / batters_total


def tired_rates(base, multiplier):
    r = inputs['pitcher_rates'][base].copy()
    r[engine.ON_BASE] = r[engine.ON_BASE] * multiplier
    return r / r.sum()


base_rate = runs_per_batter('other reliever')
tired_rows = []
for multiplier in [1.02, 1.04, 1.06, 1.08]:
    inputs['pitcher_rates']['probe'] = tired_rates('other reliever', multiplier)
    inputs['cum_cache'].clear()
    tired_rows.append([multiplier, runs_per_batter('probe') - base_rate])
tired_curve = pd.DataFrame(tired_rows, columns=['multiplier', 'extra_runs_per_batter'])
print(f'rested other reliever: {base_rate:.4f} runs per batter (same random numbers for every row)')
print(tired_curve.round(4).to_string(index=False))
slope = np.polyfit(tired_curve['multiplier'] - 1, tired_curve['extra_runs_per_batter'], 1)[0]
tired_multiplier = {label: 1 + penalty / slope for label, penalty in [('low', 0.0039), ('central', 0.0085), ('high', 0.0130)]}
print('multiplier for each penalty:', {k: round(float(v), 4) for k, v in tired_multiplier.items()})
del inputs['pitcher_rates']['probe']
inputs['cum_cache'].clear()
####
# rested other reliever: 0.1182 runs per batter (same random numbers for every row)
#  multiplier  extra_runs_per_batter
#        1.02                 0.0030
#        1.04                 0.0057
#        1.06                 0.0082
#        1.08                 0.0109
# multiplier for each penalty: {'low': 1.0297, 'central': 1.0648, 'high': 1.0991}
####
# The response is close to linear: each 0.02 on the multiplier adds about 0.003 runs per batter. The central penalty needs a multiplier of 1.065 (1.030 and 1.099 for the ends of the interval).


# Step 5 How real bullpens are used: the yardsticks for the simulated manager
# Per team-season, 2022 to 2025. The simulated manager is tuned on one of these only (how long starters last, Step 8); the rest are checks.
relief = lines[lines['role_detail'].isin(['top reliever', 'other reliever'])].copy()
relief = relief.sort_values(['id', 'date', 'number'])
relief['day'] = pd.to_datetime(relief['date'].astype(str))
relief['days_since'] = (relief['day'] - relief.groupby('id')['day'].shift(1)).dt.days
relief['days_since_2'] = (relief['day'] - relief.groupby('id')['day'].shift(2)).dt.days
relief_pitches = pa[pa['role_detail'].isin(['top reliever', 'other reliever'])].groupby(['gid', 'pitcher'])['nump'].sum().rename('pitches').reset_index().rename(columns={'pitcher': 'id'})
relief = relief.merge(relief_pitches, on=['gid', 'id'], how='left')
relief['previous_pitches'] = relief.groupby('id')['pitches'].shift(1)
team_seasons_n = lines.groupby(['year', 'team']).ngroups
real_usage = {
    'relief appearances per team-season': len(relief) / team_seasons_n,
    'relief innings per team-season': relief['p_ipouts'].sum() / 3 / team_seasons_n,
    'top-reliever appearances per team-season': (relief['role_detail'] == 'top reliever').sum() / team_seasons_n,
    'share of relief appearances back to back': (relief['days_since'] <= 1).mean(),
    'share on a third day running': ((relief['days_since'] <= 1) & (relief['days_since_2'] <= 2)).mean(),
    'share the day after 30 or more pitches': ((relief['days_since'] <= 1) & (relief['previous_pitches'] >= 30)).mean(),
    'share of relief appearances longer than 3 outs': (relief['p_ipouts'] > 3).mean(),
    'pitches per relief appearance': relief['pitches'].mean(),
}
print({k: round(float(v), 4) for k, v in real_usage.items()})
####
# {'relief appearances per team-season': 525.8667, 'relief innings per team-season': 591.5306, 'top-reliever appearances per team-season': 157.275, 'share of relief appearances back to back': 0.165, 'share on a third day running': 0.0071, 'share the day after 30 or more pitches': 0.0019, 'share of relief appearances longer than 3 outs': 0.2416, 'pitches per relief appearance': 18.294}
####
# Third consecutive days (0.7 percent of relief appearances) and pitching the day after 30 or more pitches (0.2 percent) are rare in real bullpens, which supports the availability rule in Step 6.


# Step 6 The season simulator
# Our team: middle offense, a staff of five starters, three top relievers and five other relievers (the fifth of them is the 13th pitcher). With D, D takes the 13th pitcher's place. Opponents: a random offense tier each game, league-average pitching.
# The calendar: 162 games with an off day after every seventh game (185 days), home and away in alternating three-game series. Starters rotate every fifth game.
# The manager, at the start of each of our fielding half-innings:
#   D, if he is on the staff and our lead is greater than the half-innings left in regulation (the lock rule from Phases 1 and 2); once in, he finishes the game
#   the starter is taken out the moment he reaches the pull pitch count (calibrated in Step 7), mid-inning or not, and is not sent out for a new inning within 12 pitches of it, after facing 27 batters, or after allowing 5 runs
#   (first version: pulled only between innings, with a mid-inning limit of the pull count plus 20; starters lasted 17.6 outs against 15.6 real and relievers made 417 appearances a season against 526, because real starters so often leave mid-inning)
#   a reliever pitches one inning, except that an other reliever stays on for another inning in a game that is not close if he has thrown fewer than 20 pitches and recorded fewer than 6 outs
#   a new reliever: in a close late game (from a calibrated inning on, from 1 behind to 3 ahead, and every extra inning) an available top reliever, otherwise an available other reliever, falling back on the other kind; mid-inning replacements are chosen the same way (never D)
#   which one: the manager keeps a pecking order (T1 before T2, O1 before O2) but, with a calibrated probability, skips a man who pitched yesterday if someone rested is available
#   (first version: always the freshest man, and the close-late window from the 7th. That used top relievers 213 times a season against 157 real, and the freshest-first rule spread the work so evenly that only 10.8 percent of relief appearances were back to back against 16.5 percent real; a strict pecking order gave 28.9 percent. Real managers sit in between.)
# Fatigue, when switched on: a reliever who pitched on both of the last two days, or threw 30 or more pitches yesterday, is unavailable; one who pitched yesterday pitches with the back-to-back penalty. If nobody is available, the manager uses the most rested reliever anyway (tired), and in an endless extra-inning game a position player.
# With fatigue switched off every reliever is always available and never tired; only the rule that a pitcher cannot come back into a game he left still applies.
level_cache = engine.game_factor_levels(inputs['game_factor_sd'])


def set_rates(top_key, multiplier):
    # The top relievers' rates (shrunk or raw) and the tired versions of both reliever groups for this scenario
    inputs['pitcher_rates']['top'] = inputs['pitcher_rates'][top_key]
    inputs['pitcher_rates']['other'] = inputs['pitcher_rates']['other reliever']
    for base in ['top', 'other']:
        r = inputs['pitcher_rates'][base].copy()
        r[engine.ON_BASE] = r[engine.ON_BASE] * multiplier
        inputs['pitcher_rates'][base + '|tired'] = r / r.sum()
    inputs['cum_cache'].clear()


def play_season(seed, with_d, fatigue, pull_pitches, blowout_rule=None, our_tier='middle', close_late_from=8, pick='mixed', avoid_yesterday=0.5):
    schedule_rng = random.Random(seed)
    random.seed(seed + 1)
    staff = [{'name': f'S{i}', 'role': 'starter', 'days': {}} for i in range(1, 6)]
    staff += [{'name': f'T{i}', 'role': 'top', 'days': {}} for i in range(1, 4)]
    staff += [{'name': f'O{i}', 'role': 'other', 'days': {}} for i in range(1, 5 if with_d else 6)]
    relievers = [p for p in staff if p['role'] in ('top', 'other')]
    d_member = {'name': 'D', 'role': 'D', 'days': {}}
    totals = {'wins': 0, 'runs_us': 0, 'runs_them': 0, 'extra_innings': 0, 'relief_appearances': 0, 'relief_outs': 0, 'relief_pitches': 0, 'top_appearances': 0, 'back_to_back': 0, 'third_day': 0, 'long_relief': 0, 'starter_outs': 0, 'starter_pitches': 0, 'tired_appearances': 0, 'emergency': 0, 'lock_games': 0, 'blown_locks': 0, 'd_innings': 0, 'close_late_halves': 0, 'close_late_by_top': 0, 'close_late_by_tired': 0}
    day = 0
    for game in range(162):
        if game > 0 and game % 7 == 0:
            day += 1
        day += 1
        we_are_home = (game // 3) % 2 == 0
        opponent_tier = schedule_rng.choice(tiers)
        factor_us = schedule_rng.choice(level_cache)
        factor_them = schedule_rng.choice(level_cache)
        used_today = set()
        # the manager's own coin flips come from a separate stream, so they do not shift the random numbers the game itself uses
        manager_rng = random.Random(seed * 1000 + game + 7)

        def available(member):
            if member['name'] in used_today:
                return False
            if not fatigue:
                return True
            if (day - 1) in member['days'] and (day - 2) in member['days']:
                return False
            return member['days'].get(day - 1, 0) < 30

        def appearance(member):
            used_today.add(member['name'])
            tired = fatigue and (day - 1) in member['days'] and member['role'] != 'D'
            cap = pull_pitches if member['role'] == 'starter' else (10 ** 6 if member['role'] == 'D' else 35)
            return {'name': member['name'], 'role': member['role'], 'member': member, 'pitches': 0, 'batters': 0, 'runs': 0, 'outs': 0, 'cap': cap, 'tired': tired, 'emergency': False}

        def choose(inning, lead):
            close_late = (inning >= close_late_from and -1 <= lead <= 3) or inning >= 10
            preferred = ['top', 'other'] if close_late else ['other', 'top']
            for role in preferred:
                pool = [m for m in relievers if m['role'] == role and available(m)]
                if pool:
                    # freshest first: not tired, then the most days since he last pitched
                    if pick == 'freshest':
                        pool.sort(key=lambda m: ((day - 1) in m['days'], -(day - max(m['days']) if m['days'] else 99), m['name']))
                    elif pick == 'pecking' or manager_rng.random() >= avoid_yesterday:
                        # pecking order: the manager's most trusted available man (T1 before T2, O1 before O2), whether or not he pitched yesterday
                        pool.sort(key=lambda m: m['name'])
                    else:
                        # this time the manager avoids a man who pitched yesterday if he can, and otherwise keeps to his pecking order
                        pool.sort(key=lambda m: ((day - 1) in m['days'], m['name']))
                    return appearance(pool[0])
            pool = [m for m in relievers if m['name'] not in used_today]
            if pool:
                pool.sort(key=lambda m: (-(day - max(m['days']) if m['days'] else 99), m['name']))
                a = appearance(pool[0])
                a['emergency'] = True
                return a
            a = {'name': 'PP', 'role': 'position player', 'member': None, 'pitches': 0, 'batters': 0, 'runs': 0, 'outs': 0, 'cap': 10 ** 6, 'tired': False, 'emergency': True}
            return a

        def start_half(inning, lead, current):
            if current is None:
                return appearance(staff[game % 5])
            if current['role'] == 'D':
                return current
            if with_d and inning <= 9 and (lead > 10 - inning or (blowout_rule and inning >= blowout_rule[1] and lead <= -blowout_rule[0])):
                return appearance(d_member)
            if current['role'] == 'starter':
                if current['pitches'] < pull_pitches - 12 and current['batters'] < 27 and current['runs'] < 5:
                    return current
                return choose(inning, lead)
            close_late = (inning >= close_late_from and -1 <= lead <= 3) or inning >= 10
            if current['role'] == 'other' and not close_late and current['pitches'] < 20 and current['outs'] < 6:
                return current
            return choose(inning, lead)

        def rate_key(a):
            if a['role'] == 'starter':
                return f"starter|{min(a['batters'] // 9 + 1, 3)}"
            if a['role'] == 'position player':
                return 'position player'
            return a['role'] + ('|tired' if a['tired'] else '')

        manager = {'start_half': start_half, 'relieve': lambda inning, lead, state: choose(inning, lead), 'rate_key': rate_key}
        result = engine.play_game_managed(inputs, our_tier, opponent_tier, we_are_home, manager, factor_us, factor_them, seed=seed * 1000 + game)
        won = result['score_us'] > result['score_them']
        totals['wins'] += won
        totals['runs_us'] += result['score_us']
        totals['runs_them'] += result['score_them']
        totals['extra_innings'] += result['innings'] > 9
        locked = any(kind == 'field' and inning <= 9 and lead > 10 - inning for inning, kind, lead, runs, name in result['halves'])
        totals['lock_games'] += locked
        totals['blown_locks'] += locked and not won
        for inning, kind, lead, runs, name in result['halves']:
            if kind == 'field' and name != 'D' and ((inning >= close_late_from and -1 <= lead <= 3) or inning >= 10):
                totals['close_late_halves'] += 1
                totals['close_late_by_top'] += name.startswith('T')
        for a in result['appearances']:
            if a['role'] == 'starter':
                totals['starter_outs'] += a['outs']
                totals['starter_pitches'] += a['pitches']
            elif a['role'] == 'D':
                totals['d_innings'] += a['outs'] / 3
            elif a['member'] is not None:
                m = a['member']
                totals['relief_appearances'] += 1
                totals['relief_outs'] += a['outs']
                totals['relief_pitches'] += a['pitches']
                totals['top_appearances'] += a['role'] == 'top'
                totals['back_to_back'] += (day - 1) in m['days']
                totals['third_day'] += (day - 1) in m['days'] and (day - 2) in m['days']
                totals['long_relief'] += a['outs'] > 3
                totals['tired_appearances'] += a['tired']
                totals['emergency'] += a['emergency']
                m['days'][day] = a['pitches']
            else:
                totals['emergency'] += 1
    return totals


def run_arm(n_seasons, **kwargs):
    return pd.DataFrame([play_season(1000 + s, **kwargs) for s in range(n_seasons)])


# Step 7 Calibrate the manager on three real measures, one setting each
# The pull count on outs per start (real 15.56), where the close-late window starts on top-reliever appearances (real 157 a season), and how often the manager avoids a man who pitched yesterday on the share of relief appearances that are back to back (real 0.165). Everything else in Step 8 is a check, not a target.
# Season seeds are shared across every run in this script, and every half-inning restarts its random numbers from the same seed in every version of a game, so differences between versions come from the change being tested, not from the luck of the draw.
set_rates('top reliever|shrunk', tired_multiplier['central'])
pull_rows = []
for pull in [86, 88, 90, 92, 94]:
    a = run_arm(150, with_d=False, fatigue=True, pull_pitches=pull)
    pull_rows.append([pull, a['starter_outs'].mean() / 162, a['starter_pitches'].mean() / 162, a['relief_appearances'].mean()])
pull_table = pd.DataFrame(pull_rows, columns=['pull_pitches', 'outs_per_start', 'pitches_per_start', 'relief_appearances'])
print(pull_table.round(2).to_string(index=False))
pull_pitches = int(pull_table.loc[(pull_table['outs_per_start'] - real_starter['outs per start']).abs().idxmin(), 'pull_pitches'])
window_rows = []
for start in [7, 8, 9]:
    a = run_arm(150, with_d=False, fatigue=True, pull_pitches=pull_pitches, close_late_from=start)
    window_rows.append([start, a['top_appearances'].mean()])
window_table = pd.DataFrame(window_rows, columns=['close_late_from', 'top_appearances'])
print(window_table.round(1).to_string(index=False))
close_late_from = int(window_table.loc[(window_table['top_appearances'] - real_usage['top-reliever appearances per team-season']).abs().idxmin(), 'close_late_from'])
avoid_rows = []
for avoid in [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    a = run_arm(150, with_d=False, fatigue=True, pull_pitches=pull_pitches, close_late_from=close_late_from, avoid_yesterday=avoid)
    avoid_rows.append([avoid, (a['back_to_back'] / a['relief_appearances']).mean()])
avoid_table = pd.DataFrame(avoid_rows, columns=['avoid_yesterday', 'back_to_back_share'])
print(avoid_table.round(3).to_string(index=False))
avoid_yesterday = float(avoid_table.loc[(avoid_table['back_to_back_share'] - real_usage['share of relief appearances back to back']).abs().idxmin(), 'avoid_yesterday'])
settings = {'pull_pitches': pull_pitches, 'close_late_from': close_late_from, 'avoid_yesterday': avoid_yesterday}
print('chosen:', settings)
####
#  pull_pitches  outs_per_start  pitches_per_start  relief_appearances
#            86           14.91              79.86              500.54
#            88           15.24              81.65              490.03
#            90           15.57              83.43              479.21
#            92           15.88              85.14              468.65
#            94           16.19              86.84              458.11
#  close_late_from  top_appearances
#                7            212.2
#                8            164.1
#                9             94.7
#  avoid_yesterday  back_to_back_share
#             0.65               0.188
#             0.70               0.178
#             0.75               0.167
#             0.80               0.157
#             0.85               0.145
#             0.90               0.133
# chosen: {'pull_pitches': 90, 'close_late_from': 8, 'avoid_yesterday': 0.75}
####
# Pull count 90 gives 15.57 outs a start (it was calibrated with the default window and avoidance settings, which barely touch starters). The close-late window from the 8th gives 164 top-reliever appearances against 157 real, and avoiding yesterday's reliever 75 percent of the time gives 16.7 percent back to back against 16.5.


# Step 8 Validate the baseline season (no D, fatigue on) against real bullpen use
n_main = 2000
arm_a = run_arm(n_main, with_d=False, fatigue=True, **settings)
real_lock = {'lock games per season': 62.6, 'lost locks per season': 2.5}
validation = pd.DataFrame([
    ['outs per start (the calibration target)', real_starter['outs per start'], arm_a['starter_outs'].mean() / 162],
    ['pitches per start', real_starter['pitches per start'], arm_a['starter_pitches'].mean() / 162],
    ['relief appearances per season', real_usage['relief appearances per team-season'], arm_a['relief_appearances'].mean()],
    ['relief innings per season', real_usage['relief innings per team-season'], arm_a['relief_outs'].mean() / 3],
    ['top-reliever appearances per season', real_usage['top-reliever appearances per team-season'], arm_a['top_appearances'].mean()],
    ['share of relief appearances back to back', real_usage['share of relief appearances back to back'], (arm_a['back_to_back'] / arm_a['relief_appearances']).mean()],
    ['share on a third day running', real_usage['share on a third day running'], (arm_a['third_day'] / arm_a['relief_appearances']).mean()],
    ['share of relief appearances longer than 3 outs', real_usage['share of relief appearances longer than 3 outs'], (arm_a['long_relief'] / arm_a['relief_appearances']).mean()],
    ['pitches per relief appearance', real_usage['pitches per relief appearance'], (arm_a['relief_pitches'] / arm_a['relief_appearances']).mean()],
    ['runs allowed per game (real: league average)', 4.4367, arm_a['runs_them'].mean() / 162],
    ['extra-inning share', 0.0868, arm_a['extra_innings'].mean() / 162],
    ['lock games per season (Phase 1)', real_lock['lock games per season'], arm_a['lock_games'].mean()],
    ['lost locks per season (Phases 1 and 2)', real_lock['lost locks per season'], arm_a['blown_locks'].mean()],
], columns=['measure', 'real', 'simulated'])
validation['simulated / real'] = validation['simulated'] / validation['real']
print(f'{n_main} seasons, wins per season {arm_a["wins"].mean():.2f} (standard deviation {arm_a["wins"].std():.2f}), emergencies (nobody rested enough) per season {arm_a["emergency"].mean():.2f}')
print(validation.round(3).to_string(index=False))
####
# 2000 seasons, wins per season 79.66 (standard deviation 6.50), emergencies (nobody rested enough) per season 0.93
#                                        measure    real  simulated  simulated / real
#        outs per start (the calibration target)  15.563     15.560             1.000
#                              pitches per start  83.530     83.352             0.998
#                  relief appearances per season 525.867    479.170             0.911
#                      relief innings per season 591.531    600.600             1.015
#            top-reliever appearances per season 157.275    164.876             1.048
#       share of relief appearances back to back   0.165      0.169             1.021
#                   share on a third day running   0.007      0.001             0.142
# share of relief appearances longer than 3 outs   0.242      0.318             1.315
#                  pitches per relief appearance  18.294     20.704             1.132
#   runs allowed per game (real: league average)   4.437      4.566             1.029
#                             extra-inning share   0.087      0.099             1.140
#                lock games per season (Phase 1)  62.600     60.918             0.973
#         lost locks per season (Phases 1 and 2)   2.500      2.908             1.163
####
# On the three targets, pitches per start and relief innings, the baseline matches real bullpens. The misses: extra innings run 14 percent high (as Phase 3 found); the simulated manager makes no mid-inning matchup changes, so the same relief innings come in fewer, longer appearances (479 a season against 526; 32 percent longer than three outs against 24; 20.7 pitches against 18.3), and he almost never uses a reliever three days running (0.1 percent against 0.7). The staff allows 3 percent more runs than league average and loses 2.91 locks a season against 2.50 real, because the top relievers are shrunk to their persistent talent: the simulated bullpen is somewhat worse in lock spots than real ones, which is why Step 11 takes the locked wins from the real games rather than from the simulation.

fig, ax = plt.subplots(figsize=(7.5, 4.2))
v = validation.iloc[::-1].reset_index(drop=True)
ax.axvline(1, color=ink_muted, lw=1)
ax.scatter(v['simulated / real'], range(len(v)), s=36, color='#2a78d6', edgecolor=surface, linewidth=0.8, zorder=3)
ax.set_yticks(range(len(v)), v['measure'], fontsize=8)
ax.set_xlabel('simulated / real (1 = exact)')
ax.grid(axis='y', visible=False)
fig.suptitle('Figure 12. The simulated baseline season against real bullpen use', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure12_season_validation.png', dpi=200)
plt.close(fig)


# Step 9 The value of rest: D against the 13th pitcher, with fatigue and without
# Four versions of every season: without D or with D (lock rule), each with fatigue on and with every pitcher always fresh. All four face the same schedule, opponents and game factors.
# (with D - without D) with fatigue = locked wins + everything availability adds or costs (fresher top relievers, minus the 13th pitcher's spot)
# (with D - without D) without fatigue = locked wins only, against a bullpen that is never tired
# Value of rest = the first minus the second, season by season
def compare(n, label, **kwargs):
    arms = {}
    for with_d in [False, True]:
        for fatigue in [True, False]:
            arms[(with_d, fatigue)] = run_arm(n, with_d=with_d, fatigue=fatigue, **kwargs)
    gain_fatigue = arms[(True, True)]['wins'] - arms[(False, True)]['wins']
    gain_fresh = arms[(True, False)]['wins'] - arms[(False, False)]['wins']
    rest = gain_fatigue - gain_fresh
    return {'scenario': label, 'seasons': n, 'gain_with_fatigue': gain_fatigue.mean(), 'gain_with_fatigue_se': gain_fatigue.std() / np.sqrt(n), 'gain_always_fresh': gain_fresh.mean(), 'gain_always_fresh_se': gain_fresh.std() / np.sqrt(n), 'rest_value': rest.mean(), 'rest_value_se': rest.std() / np.sqrt(n)}, arms


main, main_arms = compare(n_main, 'central', **settings)
print({k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in main.items()})
diagnostics = pd.DataFrame({f'{"D" if d else "no D"}, {"fatigue" if f else "always fresh"}': {
    'wins': a['wins'].mean(), 'runs allowed': a['runs_them'].mean(), 'lost locks': a['blown_locks'].mean(), 'D innings': a['d_innings'].mean(),
    'relief appearances': a['relief_appearances'].mean(), 'top-reliever appearances': a['top_appearances'].mean(), 'tired appearances': a['tired_appearances'].mean(),
    'share of close-late half-innings started by a top reliever': (a['close_late_by_top'] / a['close_late_halves']).mean(), 'emergencies': a['emergency'].mean()} for (d, f), a in main_arms.items()})
print(diagnostics.round(3).to_string())
####
# {'scenario': 'central', 'seasons': 2000.0, 'gain_with_fatigue': 2.881, 'gain_with_fatigue_se': 0.042, 'gain_always_fresh': 2.756, 'gain_always_fresh_se': 0.038, 'rest_value': 0.126, 'rest_value_se': 0.021}
#                                                             no D, fatigue  no D, always fresh  D, fatigue  D, always fresh
# wins                                                               79.664              80.191      82.546           82.947
# runs allowed                                                      739.764             735.110     817.751          814.440
# lost locks                                                          2.908               2.789       0.000            0.000
# D innings                                                           0.000               0.000     157.080          157.285
# relief appearances                                                479.170             477.841     375.786          372.444
# top-reliever appearances                                          164.876             165.142     131.324          123.017
# tired appearances                                                  80.863               0.000      65.804            0.000
# share of close-late half-innings started by a top reliever          0.929               0.957       0.927            0.949
# emergencies                                                         0.932               0.100       3.066            0.302
####
# The value of rest is small and well measured: 0.13 wins a season (standard error 0.02). Another way to see the same number: fatigue costs a team without D 0.53 wins a season (80.19 against 79.66) and a team with D 0.40 (82.95 against 82.55), so D removes about a quarter of what fatigue costs.
# Why so little. D frees 34 top-reliever appearances a season (165 to 131) and cuts tired appearances from 81 to 66, but without D the top relievers already start 93 percent of close-late half-innings (96 percent when they are never tired), so there is almost nothing to redeploy them into; with D the share is the same. That is the second reason Phase 2's ceiling was too high. Losing the 13th pitcher also shows up: days when nobody is rested enough rise from 0.9 to 3.1 a season.


# Step 10 How sensitive is the value of rest?
# One assumption changed at a time: the back-to-back penalty at the ends of its 95% interval; the top relievers at their full, unshrunk advantage (as if all of it were talent); the two pure manager styles (always the freshest man, a strict pecking order); our offense in the low and high tiers instead of the middle (the offense threshold from the original question).
n_sens = 1000
sensitivity_rows = [main]
for label, top_key, multiplier, tier, pick in [('penalty low (0.0039 runs per batter)', 'top reliever|shrunk', tired_multiplier['low'], 'middle', 'mixed'),
                                               ('penalty high (0.0130 runs per batter)', 'top reliever|shrunk', tired_multiplier['high'], 'middle', 'mixed'),
                                               ('top relievers unshrunk', 'top reliever', tired_multiplier['central'], 'middle', 'mixed'),
                                               ('manager always picks the freshest', 'top reliever|shrunk', tired_multiplier['central'], 'middle', 'freshest'),
                                               ('manager keeps a strict pecking order', 'top reliever|shrunk', tired_multiplier['central'], 'middle', 'pecking'),
                                               ('low offense', 'top reliever|shrunk', tired_multiplier['central'], 'low', 'mixed'),
                                               ('high offense', 'top reliever|shrunk', tired_multiplier['central'], 'high', 'mixed')]:
    set_rates(top_key, multiplier)
    row, _ = compare(n_sens, label, our_tier=tier, pick=pick, **settings)
    sensitivity_rows.append(row)
set_rates('top reliever|shrunk', tired_multiplier['central'])
sensitivity = pd.DataFrame(sensitivity_rows)
print(sensitivity.round(3).to_string(index=False))
# And the blowout use on top of the lock rule (trailing by 5 or more from the 6th), with fatigue: is it worth it when rest has its full value?
arm_blowout = run_arm(n_main, with_d=True, fatigue=True, blowout_rule=(5, 6), **settings)
blowout_gain = arm_blowout['wins'] - main_arms[(True, True)]['wins']
print(f'lock + blowout against lock only, with fatigue: {blowout_gain.mean():+.3f} wins (standard error {blowout_gain.std() / np.sqrt(n_main):.3f}); D innings {arm_blowout["d_innings"].mean():.1f} against {main_arms[(True, True)]["d_innings"].mean():.1f}')
####
#                              scenario  seasons  gain_with_fatigue  gain_with_fatigue_se  gain_always_fresh  gain_always_fresh_se  rest_value  rest_value_se
#                               central     2000              2.882                 0.042              2.756                 0.038       0.126          0.021
#  penalty low (0.0039 runs per batter)     1000              2.755                 0.056              2.698                 0.052       0.057          0.023
# penalty high (0.0130 runs per batter)     1000              2.907                 0.062              2.698                 0.052       0.209          0.036
#                top relievers unshrunk     1000              2.413                 0.057              2.266                 0.050       0.147          0.033
#     manager always picks the freshest     1000              2.767                 0.058              2.698                 0.052       0.069          0.027
#  manager keeps a strict pecking order     1000              2.839                 0.062              2.698                 0.052       0.141          0.039
#                           low offense     1000              2.682                 0.057              2.574                 0.050       0.108          0.029
#                          high offense     1000              2.983                 0.059              2.873                 0.053       0.110          0.031
# lock + blowout against lock only, with fatigue: -0.209 wins (standard error 0.016); D innings 220.3 against 157.1
####
# Every version lands between 0.06 and 0.21 wins: the size of the back-to-back penalty matters most (0.06 to 0.21 across its 95% interval), then the manager's style (0.07 if he always picks the freshest man, 0.14 with a strict pecking order), then talent (0.15 with the top relievers unshrunk). The value of rest does not depend on the offense (0.11 for both the low and high tiers). What does depend on the offense is D's total value in the simulation: 2.68 wins for the low tier and 2.98 for the high (2.88 for the middle), because better offenses reach more lock states (52 games a season for the low tier and 73 for the high in Phase 1). That is the offense threshold from the original question, and it moves the answer by about 0.3 wins, never close to zero.
# Adding the blowout use costs 0.21 wins a season (standard error 0.016) for 63 more innings from D, even with rest valued in full; Phase 2's conclusion holds.


# Step 11 The ledger with rest priced
# The locked wins come from the real games (Phase 2: 2.50 a season, what real bullpens actually lost from those states). The simulation's own locked-win figure depends on how good its bullpen is in those spots, which Step 8 shows is a little worse than reality, so the simulation is used only for the value of rest, where that level cancels out.
lock_wins_real = 2.50
ledger = sensitivity[['scenario', 'rest_value', 'rest_value_se']].copy()
ledger['locked_wins_real'] = lock_wins_real
ledger['total'] = ledger['locked_wins_real'] + ledger['rest_value']
print(ledger.round(3).to_string(index=False))
print('Phase 2 bracket for the lock rule: floor 2.56, ceiling 3.99')
####
#                              scenario  rest_value  rest_value_se  locked_wins_real  total
#                               central       0.126          0.021               2.5  2.626
#  penalty low (0.0039 runs per batter)       0.057          0.023               2.5  2.557
# penalty high (0.0130 runs per batter)       0.209          0.036               2.5  2.709
#                top relievers unshrunk       0.147          0.033               2.5  2.647
#     manager always picks the freshest       0.069          0.027               2.5  2.569
#  manager keeps a strict pecking order       0.141          0.039               2.5  2.641
#                           low offense       0.108          0.029               2.5  2.608
#                          high offense       0.110          0.031               2.5  2.610
# Phase 2 bracket for the lock rule: floor 2.56, ceiling 3.99
####
# D with the lock rule is worth 2.63 wins a season (2.56 to 2.71 across every sensitivity case): 2.50 of it from locked wins measured in real games, 0.13 from rest. The answer sits at Phase 2's floor (2.56), not its ceiling (3.99), for two measured reasons: most of the top relievers' advantage is luck (Step 3), and top relievers are rarely unavailable when a close late game needs them (Step 9). Injuries are not modelled, so this says nothing about what the rest is worth to pitchers' health.

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
rows = ledger.iloc[::-1].reset_index(drop=True)
for ax, column, se_column, title in [(axes[0], 'rest_value', 'rest_value_se', 'value of rest (wins per season)'), (axes[1], 'total', 'rest_value_se', 'locked wins (real) + rest (wins per season)')]:
    ax.errorbar(rows[column], range(len(rows)), xerr=1.96 * rows[se_column], fmt='o', color='#2a78d6', ecolor=ink_muted, elinewidth=1.2, capsize=3, ms=6)
    ax.set_title(title, fontsize=9, color=ink, pad=14)
    ax.grid(axis='y', visible=False)
axes[0].axvline(0, color=ink_muted, lw=1)
axes[1].axvline(2.56, color='#eb6834', lw=1.2, ls='--')
axes[1].axvline(3.99, color='#eb6834', lw=1.2, ls=':')
axes[1].text(2.56, 1.01, 'Phase 2 floor', transform=axes[1].get_xaxis_transform(), fontsize=7, color=ink_secondary, ha='left', va='bottom')
axes[1].text(3.99, 1.01, 'Phase 2 ceiling', transform=axes[1].get_xaxis_transform(), fontsize=7, color=ink_secondary, ha='right', va='bottom')
axes[0].set_yticks(range(len(rows)), rows['scenario'], fontsize=8)
fig.suptitle('Figure 13. What rest is worth, and where it puts D (bars: 95% intervals)', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure13_rest_value.png', dpi=200)
plt.close(fig)
sensitivity.to_csv(results_dir / 'phase4_sensitivity.csv', index=False)
validation.to_csv(results_dir / 'phase4_validation.csv', index=False)
