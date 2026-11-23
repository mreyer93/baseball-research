#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:40:00 2026

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
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import game_engine as engine

# Phase 3 of the deterministic pitcher project (the plan is in ../PLAN.md)
# Builds the inputs for the plate-appearance game engine (game_engine.py) from the 2022 to 2025 play-by-play, then checks the engine against real baseball before Phase 4 relies on it.
# Phase 4 needs more than Phase 1's inning-level arithmetic: pitchers of different quality, pitch counts, and changes of pitcher. So the engine works batter by batter, and it has to reproduce real scoring, real win expectancy, and the real gap between good and bad pitchers.
# Data: Retrosheet play-by-play (season CSV files), regular season, complete regulation games.
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
real_color = '#86b6ef'
sim_color = '#104281'
plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface, 'axes.edgecolor': ink_muted, 'axes.labelcolor': ink_secondary, 'xtick.color': ink_muted, 'ytick.color': ink_muted, 'text.color': ink, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


# Step 1 Read the play-by-play
columns = ['gid', 'inning', 'top_bot', 'event', 'batteam', 'pitteam', 'score_v', 'score_h', 'pitcher', 'lp', 'pa', 'outs_pre', 'outs_post', 'br1_pre', 'br2_pre', 'br3_pre', 'br1_post', 'br2_post', 'br3_post', 'runs', 'single', 'double', 'triple', 'hr', 'walk', 'hbp', 'k', 'roe', 'xi', 'nump', 'gametype']
plays_list = []
for year in years:
    p = pd.read_csv(csv_dir / str(year) / f'{year}plays.csv', usecols=columns, low_memory=False)
    p = p[p['gametype'] == 'regular'].drop(columns='gametype')
    p['year'] = year
    plays_list.append(p)
plays = pd.concat(plays_list, ignore_index=True)
has_top_ninth = plays[(plays['inning'] == 9) & (plays['top_bot'] == 0)]['gid'].unique()
plays = plays[plays['gid'].isin(has_top_ninth)]
# Substitution rows (NP) and dropped fouls on an error (FLE) change nothing on the field; everything else is either a plate appearance or a runner event
plays = plays[~plays['event'].str.match(r'^(NP|FLE)')].reset_index(drop=True)
plays['state_pre'] = plays['outs_pre'] * 8 + plays['br1_pre'].notna() * 1 + plays['br2_pre'].notna() * 2 + plays['br3_pre'].notna() * 4
plays['state_post'] = np.where(plays['outs_post'] >= 3, engine.END, plays['outs_post'] * 8 + plays['br1_post'].notna() * 1 + plays['br2_post'].notna() * 2 + plays['br3_post'].notna() * 4)
# Plate-appearance outcome, in priority order (an intentional walk is already flagged as a walk; catcher's interference is rare enough to count as a hit by pitch)
plays['outcome'] = np.select([plays['hr'] == 1, plays['triple'] == 1, plays['double'] == 1, plays['single'] == 1, plays['walk'] == 1, (plays['hbp'] == 1) | (plays['xi'] == 1), plays['k'] == 1, plays['roe'] == 1], ['HR', '3B', '2B', '1B', 'BB', 'HBP', 'K', 'ROE'], 'OUT')
plays.loc[plays['pa'] == 0, 'outcome'] = 'runner'
team_seasons = pd.read_csv(results_dir / 'phase1_team_seasons.csv')
tier_lookup = team_seasons.set_index(['year', 'team'])['tier']
plays['bat_tier'] = [tier_lookup[(y, t)] for y, t in zip(plays['year'], plays['batteam'])]
plays['fielding_lead_now'] = np.where(plays['top_bot'] == 0, plays['score_h'] - plays['score_v'], plays['score_v'] - plays['score_h'])
pa = plays[plays['pa'] == 1]
print('games:', plays['gid'].nunique(), ' rows:', len(plays), ' plate appearances:', len(pa), ' runner events:', (plays['pa'] == 0).sum())
print('outcome shares:', (pa['outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES)).round(4).to_dict())
####
# games: 9697  rows: 754637  plate appearances: 730295  runner events: 24342
# outcome shares: {'K': 0.2249, 'BB': 0.0833, 'HBP': 0.0116, '1B': 0.1421, '2B': 0.0433, '3B': 0.0037, 'HR': 0.0303, 'ROE': 0.0059, 'OUT': 0.4549}
####
# 730,295 plate appearances and 24,342 runner events (steals, wild pitches, pickoffs and the like) in the 9,697 games Phases 1 and 2 used.


# Step 2 Pitcher groups: whose plate appearances are whose
# Roles come from Phase 2 (starter, the team's top three relievers by leverage, other relievers, position players). The mop-up group is defined by the game state as in Phase 1: the 6th inning or later with the fielding team 5 or more runs ahead or behind at the start of the half-inning, whoever is pitching.
appearances = pd.read_csv(results_dir / 'phase2_appearances.csv')
plays = plays.merge(appearances[['gid', 'id', 'role_detail']].rename(columns={'id': 'pitcher'}), on=['gid', 'pitcher'], how='left')
half_start_lead = plays.groupby(['gid', 'inning', 'top_bot'])['fielding_lead_now'].transform('first')
plays['mop_up'] = (plays['inning'] >= 6) & (half_start_lead.abs() >= 5)
pa = plays[plays['pa'] == 1]
league_rates = pa['outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES).values
pitcher_rates = {'league': league_rates}
for group in ['starter', 'top reliever', 'other reliever', 'position player']:
    pitcher_rates[group] = pa.loc[pa['role_detail'] == group, 'outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES).fillna(0).values
pitcher_rates['mop_up'] = pa.loc[pa['mop_up'], 'outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES).values
print(pd.DataFrame(pitcher_rates, index=engine.OUTCOMES).T.round(4).to_string())
print('plate appearances per group:', {g: int((pa['role_detail'] == g).sum()) for g in ['starter', 'top reliever', 'other reliever', 'position player']}, ' mop-up:', int(pa['mop_up'].sum()))
####
#                       K      BB     HBP      1B      2B      3B      HR     ROE     OUT
# league           0.2249  0.0833  0.0116  0.1421  0.0433  0.0037  0.0303  0.0059  0.4549
# starter          0.2187  0.0769  0.0105  0.1439  0.0449  0.0037  0.0325  0.0060  0.4629
# top reliever     0.2732  0.0894  0.0117  0.1292  0.0346  0.0030  0.0228  0.0056  0.4305
# other reliever   0.2224  0.0935  0.0134  0.1422  0.0429  0.0038  0.0285  0.0059  0.4475
# position player  0.0239  0.0766  0.0231  0.2166  0.0690  0.0024  0.0566  0.0060  0.5257
# mop_up           0.2202  0.0797  0.0128  0.1449  0.0452  0.0037  0.0307  0.0061  0.4568
# plate appearances per group: {'starter': 425335, 'top reliever': 76547, 'other reliever': 225906, 'position player': 2507}  mop-up: 71522
####
# The groups differ the way they should: top relievers strike out 27.3 percent of batters and allow home runs on 2.3 percent (league 22.5 and 3.0), and position players strike out 2.4 percent and allow home runs on 5.7 percent. The mop-up group sits almost exactly at league average, which matches Phase 1's finding that mop-up innings cost only about 0.03 runs more than average late innings.


# Step 3 Batter rates, runner transitions and pitch counts
# Batter rates by offense tier, home or away, and lineup slot (3 x 2 x 9 = 54 cells, roughly 14,000 plate appearances each). These already contain the league's mix of pitchers, which is why a league-average pitcher leaves them unchanged.
batting_rates = {}
for (tier, home, slot), g in pa.groupby(['bat_tier', 'top_bot', 'lp']):
    batting_rates[(tier, int(home), int(slot) - 1)] = g['outcome'].value_counts(normalize=True).reindex(engine.OUTCOMES).fillna(0).values
print('batter cells:', len(batting_rates), ' smallest:', int(pa.groupby(['bat_tier', 'top_bot', 'lp']).size().min()))

# Transitions: for each outcome (and runner events) in each base-out state, what actually happened next (new state, runs). Where a cell has fewer than 30 cases, the cases from the same runners with any number of outs stand in, shifted to the right number of outs.
transitions = {}
sparse_cells = 0
plays['outs_added'] = plays['outs_post'] - plays['outs_pre']
plays['bases_pre'] = plays['state_pre'] % 8
plays['bases_post'] = np.where(plays['state_post'] == engine.END, -1, plays['state_post'] % 8)
grouped = {k: g for k, g in plays.groupby(['outcome', 'state_pre'])}
pooled = {k: g for k, g in plays.groupby(['outcome', 'bases_pre'])}
for kind in engine.OUTCOMES + ['runner']:
    for state in range(24):
        g = grouped.get((kind, state))
        if g is not None and len(g) >= 30:
            counts = g.groupby(['state_post', 'runs']).size()
            results = list(counts.index)
        else:
            sparse_cells += 1
            outs, bases = divmod(state, 8)
            g = pooled.get((kind, bases))
            if g is None:
                continue
            # A case that ended the inning from fewer outs says nothing about where the runners would have been, so it is used only if it also ends the inning here
            shifted = [(engine.END if outs + a >= 3 else (outs + a) * 8 + b, r) for a, b, r in zip(g['outs_added'], g['bases_post'], g['runs']) if b >= 0 or outs + a >= 3]
            counts = pd.Series(shifted).value_counts()
            results = list(counts.index)
        transitions[(engine.OUTCOMES.index(kind) if kind != 'runner' else 'runner', state)] = (list(np.cumsum(counts.values / counts.values.sum())), [(int(s), int(r)) for s, r in results])
# Chance of a runner event before the next plate appearance, by state
row_counts = plays.groupby(['state_pre', plays['pa'] == 0]).size().unstack(fill_value=0)
runner_event_chance = (row_counts[True] / row_counts.sum(axis=1)).reindex(range(24)).fillna(0).to_numpy(copy=True)
# With the bases empty there is nobody to move; the handful of such rows are scoring oddities
runner_event_chance[[0, 8, 16]] = 0
pitches_by_outcome = {engine.OUTCOMES.index(o): pa.loc[(pa['outcome'] == o) & pa['nump'].notna(), 'nump'].astype(int).sample(n=min(20000, int(((pa['outcome'] == o) & pa['nump'].notna()).sum())), random_state=1).tolist() for o in engine.OUTCOMES}
print('transition cells:', len(transitions), ' filled from the pooled cases:', sparse_cells)
print('chance of a runner event before the next plate appearance, runner on 1st only, by outs:', runner_event_chance[[1, 9, 17]].round(4).tolist())
print('mean pitches per plate appearance:', round(pa['nump'].mean(), 3))
####
# batter cells: 54  smallest: 11578
# transition cells: 237  filled from the pooled cases: 22
# chance of a runner event before the next plate appearance, runner on 1st only, by outs: [0.0835, 0.0977, 0.1265]
# mean pitches per plate appearance: 3.808
####
# 237 of 240 transition cells are filled (the three missing are runner events with the bases empty, which the engine never draws), 22 of them from the pooled cases.

inputs = {'league_rates': league_rates, 'pitcher_rates': pitcher_rates, 'batting_rates': batting_rates, 'transitions': transitions, 'runner_event_chance': runner_event_chance, 'pitches_by_outcome': pitches_by_outcome}
with open(results_dir / 'phase3_engine_inputs.pkl', 'wb') as f:
    pickle.dump(inputs, f)
inputs = engine.load_inputs(results_dir / 'phase3_engine_inputs.pkl')


# Step 4 Calibrate the game factor: how much one day differs from another
# First run, with no game factor: the engine got average scoring right but bunched game scores too tightly (standard deviation of team runs per game 2.94 against 3.17 real) and produced too many extra-inning games (10.0 percent against 8.7). Every simulated game was played in the same run environment, which real games are not. The sd = 0 row below is that first run.
# One target only: the standard deviation of team runs per game. Mean scoring, extra innings, run expectancy and win expectancy are then checked without having been tuned on.
# Each simulated game copies the offense tiers of a randomly drawn real game, so the mix of matchups matches 2022 to 2025
real_games = plays.groupby('gid').agg(year=('year', 'first'), visitor=('batteam', 'first'), home=('pitteam', 'first')).reset_index()
real_games['visitor_tier'] = [tier_lookup[(y, t)] for y, t in zip(real_games['year'], real_games['visitor'])]
real_games['home_tier'] = [tier_lookup[(y, t)] for y, t in zip(real_games['year'], real_games['home'])]
last_play = plays.groupby('gid').tail(1).set_index('gid')
real_final_v = last_play['score_v'] + last_play['runs'].where(last_play['top_bot'] == 0, 0)
real_final_h = last_play['score_h'] + last_play['runs'].where(last_play['top_bot'] == 1, 0)
real_innings = plays.groupby('gid')['inning'].max()
real_sd = pd.concat([real_final_v, real_final_h]).std()
league_plan = lambda side, inning, lead: 'league'
calibration_rows = []
for sd in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    scores = np.array([[r['score_v'], r['score_h'], r['innings']] for r in [engine.play_game(inputs, vt, ht, league_plan, game_factor_sd=sd) for vt, ht in zip(*real_games.sample(n=20000, replace=True, random_state=3)[['visitor_tier', 'home_tier']].values.T)]])
    team_runs = np.concatenate([scores[:, 0], scores[:, 1]])
    calibration_rows.append([sd, team_runs.mean(), team_runs.std(ddof=1), (scores[:, 2] > 9).mean()])
calibration = pd.DataFrame(calibration_rows, columns=['game_factor_sd', 'runs_per_team_game', 'sd_team_runs', 'extra_inning_share'])
print(calibration.round(4).to_string(index=False))
print(f'real: runs per team-game {pd.concat([real_final_v, real_final_h]).mean():.4f}, sd {real_sd:.4f}, extra-inning share {(real_innings > 9).mean():.4f}')
chosen_sd = float(calibration.loc[(calibration['sd_team_runs'] - real_sd).abs().idxmin(), 'game_factor_sd'])
print('chosen game factor sd:', chosen_sd)
inputs['game_factor_sd'] = chosen_sd
with open(results_dir / 'phase3_engine_inputs.pkl', 'wb') as f:
    pickle.dump({k: v for k, v in inputs.items() if k != 'cum_cache'}, f)
####
#  game_factor_sd  runs_per_team_game  sd_team_runs  extra_inning_share
#            0.00              4.4375        2.9459              0.0966
#            0.05              4.4304        2.9469              0.1007
#            0.10              4.4648        3.0162              0.0989
#            0.15              4.4842        3.1468              0.0995
#            0.20              4.4616        3.2182              0.0946
#            0.25              4.5178        3.3928              0.0946
#            0.30              4.5325        3.5120              0.0915
#            0.35              4.5675        3.7275              0.0907
# real: runs per team-game 4.4367, sd 3.1678, extra-inning share 0.0868
# chosen game factor sd: 0.15
####
# With no game factor the spread of team runs per game is 2.95 against 3.17 real. A factor of 0.15 brings it to 3.15 in this calibration run. The cost is about 1 percent more scoring (4.48 runs against 4.44). The extra-inning share hardly moves with the factor (0.097 to 0.100 at 0.15, within noise), so the game factor does not cure the excess of tied games; Step 11 looks at why.


# Step 5 Simulate a league of games with league-average pitching and the chosen game factor
n_games = 40000
run_expectancy_sim = (np.zeros(24), np.zeros(24))
started = time.time()
sim_halves = []
sim_games = []
draws = real_games.sample(n=n_games, replace=True, random_state=2)
for game_i, (vt, ht) in enumerate(zip(draws['visitor_tier'], draws['home_tier'])):
    result = engine.play_game(inputs, vt, ht, league_plan, run_expectancy=run_expectancy_sim)
    sim_games.append([game_i, vt, ht, result['score_v'], result['score_h'], result['innings']])
    for inning, top_bot, lead, runs, pitcher, batters, pitches in result['halves']:
        sim_halves.append([game_i, inning, top_bot, lead, runs, batters, pitches, (vt, ht)[top_bot], (ht, vt)[top_bot]])
sim_games = pd.DataFrame(sim_games, columns=['game', 'visitor_tier', 'home_tier', 'score_v', 'score_h', 'innings'])
sim_halves = pd.DataFrame(sim_halves, columns=['game', 'inning', 'top_bot', 'fielding_lead', 'runs', 'batters', 'pitches', 'bat_tier', 'field_tier'])
print(f'simulated {n_games} games')


# Step 6 Test: scoring, home advantage, extra innings and pitch counts against the real games
plays['pa_pitches'] = plays['nump'].fillna(0).where(plays['pa'] == 1, 0)
real_half = plays.groupby(['gid', 'inning', 'top_bot']).agg(runs=('runs', 'sum'), fielding_lead=('fielding_lead_now', 'first'), bat_tier=('bat_tier', 'first'), pitches=('pa_pitches', 'sum')).reset_index()
full_real = real_half[(real_half['inning'] <= 9) & ~((real_half['inning'] == 9) & (real_half['top_bot'] == 1))]
full_sim = sim_halves[(sim_halves['inning'] <= 9) & ~((sim_halves['inning'] == 9) & (sim_halves['top_bot'] == 1))]
rows = []
for label, real_value, sim_value in [
        ('runs per half-inning (full regulation half-innings)', full_real['runs'].mean(), full_sim['runs'].mean()),
        ('share of scoreless half-innings', (full_real['runs'] == 0).mean(), (full_sim['runs'] == 0).mean()),
        ('share of half-innings with 3 or more runs', (full_real['runs'] >= 3).mean(), (full_sim['runs'] >= 3).mean()),
        ('runs per team-game', (real_final_v.mean() + real_final_h.mean()) / 2, (sim_games['score_v'].mean() + sim_games['score_h'].mean()) / 2),
        ('standard deviation of team runs per game', pd.concat([real_final_v, real_final_h]).std(), pd.concat([sim_games['score_v'], sim_games['score_h']]).std()),
        ('home team win share', (real_final_h > real_final_v).mean(), (sim_games['score_h'] > sim_games['score_v']).mean()),
        ('extra-inning share', (real_innings > 9).mean(), (sim_games['innings'] > 9).mean()),
        ('pitches per half-inning (full regulation half-innings)', full_real['pitches'].mean(), full_sim['pitches'].mean())]:
    rows.append([label, real_value, sim_value])
for tier in tiers:
    rows.append([f'runs per half-inning, {tier} offense', full_real.loc[full_real['bat_tier'] == tier, 'runs'].mean(), full_sim.loc[full_sim['bat_tier'] == tier, 'runs'].mean()])
for inning in [1, 5, 8]:
    rows.append([f'runs per half-inning, inning {inning}', full_real.loc[full_real['inning'] == inning, 'runs'].mean(), full_sim.loc[full_sim['inning'] == inning, 'runs'].mean()])
scoring_check = pd.DataFrame(rows, columns=['measure', 'real', 'simulated'])
print(scoring_check.round(4).to_string(index=False))
####
# simulated 40000 games
#                                                measure    real  simulated
#    runs per half-inning (full regulation half-innings)  0.4951     0.4963
#                        share of scoreless half-innings  0.7308     0.7286
#              share of half-innings with 3 or more runs  0.0568     0.0567
#                                     runs per team-game  4.4367     4.4823
#               standard deviation of team runs per game  3.1678     3.1237
#                                    home team win share  0.5298     0.5228
#                                     extra-inning share  0.0868     0.1005
# pitches per half-inning (full regulation half-innings) 16.1691    16.2171
#                      runs per half-inning, low offense  0.4385     0.4448
#                   runs per half-inning, middle offense  0.4909     0.4931
#                     runs per half-inning, high offense  0.5573     0.5524
#                         runs per half-inning, inning 1  0.5148     0.5430
#                         runs per half-inning, inning 5  0.5102     0.4910
#                         runs per half-inning, inning 8  0.4809     0.4970
####
# Scoring, scoreless innings, big innings, the three offense tiers, pitch counts and home advantage all land within about 1 to 2 percent of the real games. Two misses. Extra innings: 10.0 percent of games against 8.7 (Step 11). And the pattern across innings: the engine scores 0.543 runs in the 1st against 0.515 real, 0.491 in the 5th against 0.510, and 0.497 in the 8th against 0.481. Real pitching changes through a game (fresh starters in the 1st, starters facing the order a third time in the 5th or 6th, bullpens in the 8th) and league-average pitching everywhere flattens that. Phase 4 has to model starters by time through the order and relievers late to get it right.

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
r = np.arange(0, 8)
real_pmf = full_real['runs'].value_counts(normalize=True).reindex(r, fill_value=0)
sim_pmf = full_sim['runs'].value_counts(normalize=True).reindex(r, fill_value=0)
axes[0].bar(r - 0.2, real_pmf, width=0.4, color=real_color, label='real 2022 to 2025')
axes[0].bar(r + 0.2, sim_pmf, width=0.4, color=sim_color, label='simulated')
axes[0].set_yscale('log')
axes[0].set_xlabel('runs in a half-inning')
axes[0].set_ylabel('share of half-innings (log scale)')
axes[0].legend(frameon=False, fontsize=8)
g = np.arange(0, 16)
real_game = pd.concat([real_final_v, real_final_h]).value_counts(normalize=True).reindex(g, fill_value=0)
sim_game = pd.concat([sim_games['score_v'], sim_games['score_h']]).value_counts(normalize=True).reindex(g, fill_value=0)
axes[1].bar(g - 0.2, real_game, width=0.4, color=real_color)
axes[1].bar(g + 0.2, sim_game, width=0.4, color=sim_color)
axes[1].set_xlabel('runs by one team in a game')
axes[1].set_ylabel('share of team-games')
fig.suptitle('Figure 9. Scoring in simulated and real games', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure9_engine_scoring.png', dpi=200)
plt.close(fig)


# Step 7 Test: run expectancy by base-out state (the average runs still to come in the inning from each of the 24 states), innings 1 to 8
real_rows = plays[plays['inning'] <= 8].copy()
real_rows['runs_before'] = real_rows.groupby(['gid', 'inning', 'top_bot'])['runs'].cumsum() - real_rows['runs']
real_rows['half_total'] = real_rows.groupby(['gid', 'inning', 'top_bot'])['runs'].transform('sum')
real_re = (real_rows['half_total'] - real_rows['runs_before']).groupby(real_rows['state_pre']).mean().reindex(range(24))
sim_re = pd.Series(run_expectancy_sim[0] / run_expectancy_sim[1], index=range(24))
re_table = pd.DataFrame({'outs': [s // 8 for s in range(24)], 'runners': [''.join(b if (s % 8) >> i & 1 else '-' for i, b in enumerate('123')) for s in range(24)], 'real': real_re.values, 'simulated': sim_re.values})
re_table['difference'] = re_table['simulated'] - re_table['real']
print(re_table.round(3).to_string(index=False))
print('largest absolute difference:', round(re_table['difference'].abs().max(), 3), ' mean absolute difference:', round(re_table['difference'].abs().mean(), 3))
####
#  outs runners  real  simulated  difference
#     0     --- 0.498      0.499       0.001
#     0     1-- 0.896      0.896      -0.000
#     0     -2- 1.145      1.107      -0.038
#     0     12- 1.521      1.531       0.010
#     0     --3 1.351      1.324      -0.028
#     0     1-3 1.847      1.772      -0.075
#     0     -23 2.000      1.934      -0.066
#     0     123 2.397      2.372      -0.024
#     1     --- 0.266      0.270       0.004
#     1     1-- 0.529      0.528      -0.001
#     1     -2- 0.690      0.681      -0.009
#     1     12- 0.946      0.962       0.016
#     1     --3 0.966      0.914      -0.053
#     1     1-3 1.208      1.185      -0.022
#     1     -23 1.412      1.351      -0.061
#     1     123 1.607      1.641       0.034
#     2     --- 0.102      0.106       0.004
#     2     1-- 0.227      0.234       0.007
#     2     -2- 0.326      0.331       0.005
#     2     12- 0.459      0.471       0.012
#     2     --3 0.365      0.367       0.002
#     2     1-3 0.507      0.524       0.018
#     2     -23 0.568      0.613       0.046
#     2     123 0.797      0.820       0.023
# largest absolute difference: 0.075  mean absolute difference: 0.023
####
# Run expectancy is within 0.023 runs on average and 0.075 at worst (it was 0.033 and 0.119 before the game factor). The engine is still a little low with a runner on third and fewer than two outs (by 0.02 to 0.075 in seven of those eight states; bases loaded with one out is 0.034 high), probably because real batters change their approach there (contact over power, the infield drawn in), which batter rates that ignore the base-out state cannot know.


# Step 8 Test: win expectancy (the share of games won from each inning, half and lead) against the real games
# Seen from the fielding team at the start of each regulation half-inning, as in Phases 1 and 2. The real table is also scored against itself, which is the best any lookup by inning, half and lead can do in-sample.
last_winner_v = (real_final_v > real_final_h)
real_half['fielding_won'] = np.where(real_half['top_bot'] == 0, ~last_winner_v.loc[real_half['gid']].values, last_winner_v.loc[real_half['gid']].values)
sim_halves['fielding_won'] = np.where(sim_halves['top_bot'] == 0, (sim_games.loc[sim_halves['game'], 'score_h'] > sim_games.loc[sim_halves['game'], 'score_v']).values, (sim_games.loc[sim_halves['game'], 'score_v'] > sim_games.loc[sim_halves['game'], 'score_h']).values)
for frame in [real_half, sim_halves]:
    frame['lead_clipped'] = frame['fielding_lead'].clip(-8, 8)
real_reg = real_half[real_half['inning'] <= 9].copy()
sim_reg = sim_halves[sim_halves['inning'] <= 9]
real_we = real_reg.groupby(['inning', 'top_bot', 'lead_clipped'])['fielding_won'].agg(['mean', 'size'])
sim_we = sim_reg.groupby(['inning', 'top_bot', 'lead_clipped'])['fielding_won'].agg(['mean', 'size'])
we = real_we.join(sim_we, lsuffix='_real', rsuffix='_sim', how='inner')
we_big = we[(we['size_real'] >= 200) & (we['size_sim'] >= 200)]
print(f'win expectancy cells with 200 or more games in both: {len(we_big)} of {len(we)}; mean absolute difference {(we_big["mean_sim"] - we_big["mean_real"]).abs().mean():.4f}, weighted by real games {((we_big["mean_sim"] - we_big["mean_real"]).abs() * we_big["size_real"]).sum() / we_big["size_real"].sum():.4f}, largest {(we_big["mean_sim"] - we_big["mean_real"]).abs().max():.4f}')
real_reg['sim_we'] = sim_we['mean'].reindex(pd.MultiIndex.from_frame(real_reg[['inning', 'top_bot', 'lead_clipped']])).values
real_reg['real_we'] = real_we['mean'].reindex(pd.MultiIndex.from_frame(real_reg[['inning', 'top_bot', 'lead_clipped']])).values
scored = real_reg.dropna(subset=['sim_we'])
print(f'Brier score on the real games: simulated table {((scored["sim_we"] - scored["fielding_won"]) ** 2).mean():.4f}, real table in-sample {((scored["real_we"] - scored["fielding_won"]) ** 2).mean():.4f} (Phase 1 analytic model: 0.1615)')
####
# win expectancy cells with 200 or more games in both: 192 of 274; mean absolute difference 0.0088, weighted by real games 0.0084, largest 0.0427
# Brier score on the real games: simulated table 0.1622, real table in-sample 0.1620 (Phase 1 analytic model: 0.1615)
####
# Win expectancy is within 0.9 points on average and 4.3 at worst, and as a lookup table it scores almost exactly as well on the real games as the real table does on itself (Brier 0.1622 against 0.1620).

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].plot([0, 1], [0, 1], color=ink_muted, lw=1, ls='--')
axes[0].scatter(we_big['mean_real'], we_big['mean_sim'], s=np.clip(we_big['size_real'] / 60, 6, 60), color=sim_color, alpha=0.6, edgecolor=surface, linewidth=0.6)
axes[0].set_xlabel('real share of games won')
axes[0].set_ylabel('simulated share of games won')
axes[0].set_title('win expectancy, inning x half x lead (dot size: real games)', fontsize=9, color=ink)
axes[1].plot([0, 2.5], [0, 2.5], color=ink_muted, lw=1, ls='--')
axes[1].scatter(re_table['real'], re_table['simulated'], s=26, color=sim_color, edgecolor=surface, linewidth=0.8)
axes[1].set_xlabel('real runs still to come')
axes[1].set_ylabel('simulated runs still to come')
axes[1].set_title('run expectancy, 24 base-out states', fontsize=9, color=ink)
fig.suptitle('Figure 10. Win expectancy and run expectancy, simulated against real', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure10_engine_expectancy.png', dpi=200)
plt.close(fig)


# Step 9 Test: do pitchers of different quality allow the right number of runs?
# For each pitcher group, simulate half-innings with that group pitching the whole way (random offense tier, home or away, lineup position and game factor) and compare runs per batter with the real group.
# First version compared against runs scored while the group was on the mound. That is the wrong yardstick for pitchers who enter mid-inning: relievers inherit runners who score on their watch but are charged to the man before, and starters leave runners behind. It made relievers look about 0.018 runs per batter worse than the engine and starters 0.005 better. Runs charged to the pitcher (from the per-pitcher game lines) are the fair comparison for the role groups; the mop-up and position-player groups are compared on runs while on the mound, since they mostly pitch whole innings.
pitching_lines = pd.concat([pd.read_csv(csv_dir / str(y) / f'{y}pitching.csv', usecols=['gid', 'id', 'p_r', 'p_bfp', 'gametype']) for y in years])
pitching_lines = pitching_lines[pitching_lines['gametype'] == 'regular'].merge(appearances[['gid', 'id', 'role_detail']], on=['gid', 'id'], how='inner')
real_group_rows = []
for group in ['league', 'starter', 'top reliever', 'other reliever', 'position player', 'mop_up']:
    mask = plays['mop_up'] if group == 'mop_up' else (plays['pa'] > -1 if group == 'league' else plays['role_detail'] == group)
    lines = pitching_lines if group == 'league' else pitching_lines[pitching_lines['role_detail'] == group]
    charged = lines['p_r'].sum() / lines['p_bfp'].sum() if group != 'mop_up' else np.nan
    real_group_rows.append([group, plays.loc[mask, 'runs'].sum() / plays.loc[mask, 'pa'].sum(), charged, int(plays.loc[mask, 'pa'].sum())])
group_check = pd.DataFrame(real_group_rows, columns=['group', 'real_on_mound', 'real_charged', 'real_batters'])
levels = engine.game_factor_levels(inputs['game_factor_sd'])
sim_rates = []
for group in group_check['group']:
    runs_total = 0
    batters_total = 0
    for _ in range(30000):
        runs, _, batters, _ = engine.half_inning(inputs, random.choice(tiers), random.randint(0, 1), random.randint(0, 8), group, factor=random.choice(levels))
        runs_total += runs
        batters_total += batters
    sim_rates.append(runs_total / batters_total)
group_check['simulated'] = sim_rates
group_check['yardstick'] = np.where(group_check['group'].isin(['mop_up', 'position player']), group_check['real_on_mound'], group_check['real_charged'])
group_check['difference'] = group_check['simulated'] - group_check['yardstick']
print(group_check.round(4).to_string(index=False))
gap_real = group_check.set_index('group').loc['other reliever', 'real_charged'] - group_check.set_index('group').loc['top reliever', 'real_charged']
gap_sim = group_check.set_index('group').loc['other reliever', 'simulated'] - group_check.set_index('group').loc['top reliever', 'simulated']
print(f'gap between other and top relievers, runs per batter: real (charged) {gap_real:.4f}, simulated {gap_sim:.4f}')
mop_halves = real_half[(real_half['inning'] >= 6) & (real_half['fielding_lead'].abs() >= 5) & ~((real_half['inning'] == 9) & (real_half['top_bot'] == 1)) & (real_half['inning'] <= 9)]
sim_mop = np.mean([engine.half_inning(inputs, random.choice(tiers), random.randint(0, 1), random.randint(0, 8), 'mop_up', factor=random.choice(levels))[0] for _ in range(30000)])
print(f'mop-up runs per half-inning: real {mop_halves["runs"].mean():.3f} (Phase 1 from the game logs: 0.520), simulated {sim_mop:.3f}')
####
#           group  real_on_mound  real_charged  real_batters  simulated  yardstick  difference
#          league         0.1178        0.1178        730295     0.1140     0.1178     -0.0038
#         starter         0.1102        0.1185        425335     0.1193     0.1185      0.0008
#    top reliever         0.1071        0.0963         76547     0.0925     0.0963     -0.0038
#  other reliever         0.1345        0.1227        225906     0.1206     0.1227     -0.0021
# position player         0.2349        0.2154          2507     0.2364     0.2349      0.0014
#          mop_up         0.1200           NaN         71522     0.1183     0.1200     -0.0017
# gap between other and top relievers, runs per batter: real (charged) 0.0264, simulated 0.0281
# mop-up runs per half-inning: real 0.520 (Phase 1 from the game logs: 0.520), simulated 0.507
####
# Against runs charged, every group is within 0.004 runs per batter (about 4 percent or less). The gap between other relievers and top relievers, which drives the redeployment value in Phase 4, comes out at 0.0281 against 0.0264 real, so the engine overstates it by about 6 percent; Phase 4 should either shrink it or report the redeployment value with that caveat.


# Step 10 Test: D in the engine, against the Phase 1 arithmetic
# D concedes exactly one run per half-inning by construction; check it. Then play out six game states from the start of the home team's fielding half (home team middle offense, visitor a random tier, random lineup positions), once with D and once with a mop-up reliever pitching the rest of regulation for the home team, and compare the win rates with the Phase 1 calculation for the same states.
d_runs = [engine.half_inning(inputs, random.choice(tiers), 0, random.randint(0, 8), 'D')[0] for _ in range(1000)]
print('D runs per half-inning over 1000 simulated half-innings: min', min(d_runs), 'max', max(d_runs))
with open(results_dir / 'phase1_pmfs.json') as f:
    saved_pmfs = json.load(f)
tier_inning_pmf = {tier: {int(i): np.array(v) for i, v in saved_pmfs['tier_inning_pmf'][tier].items()} for tier in tiers}
mop_up_pmf = np.array(saved_pmfs['mop_up_pmf'])


def add_innings(pmf_list):
    total = np.array([1.0])
    for pmf in pmf_list:
        total = np.convolve(total, pmf)
    return total


def win_probability(lead, our_total_pmf, their_total_pmf):
    margin_pmf = np.convolve(our_total_pmf, their_total_pmf[::-1])
    margin = lead + np.arange(len(margin_pmf)) - (len(their_total_pmf) - 1)
    return margin_pmf[margin > 0].sum() + 0.5 * margin_pmf[margin == 0].sum()


d_rows = []
n_each = 6000
for lead, inning in [(2, 9), (1, 9), (3, 7), (0, 7), (5, 5), (-5, 6)]:
    n = 10 - inning
    ours = add_innings([tier_inning_pmf['middle'][j] for j in range(inning, 10)])
    analytic_d = win_probability(lead, ours, np.eye(n + 1)[n])
    analytic_mop = win_probability(lead, ours, add_innings([mop_up_pmf] * n))
    sim_result = {}
    for pitcher in ['D', 'mop_up']:
        plan = lambda side, i, l, first=inning, p=pitcher: p if (side == 'home' and i >= first and i <= 9) else 'league'
        wins = 0
        for _ in range(n_each):
            start = {'inning': inning, 'score_v': 5, 'score_h': 5 + lead, 'slot_v': random.randint(0, 8), 'slot_h': random.randint(0, 8)}
            result = engine.play_game(inputs, random.choice(tiers), 'middle', plan, start=start)
            wins += result['score_h'] > result['score_v']
        sim_result[pitcher] = wins / n_each
    d_rows.append([lead, inning, n, analytic_d, sim_result['D'], analytic_mop, sim_result['mop_up'], analytic_d - analytic_mop, sim_result['D'] - sim_result['mop_up']])
d_check = pd.DataFrame(d_rows, columns=['lead', 'inning', 'half_innings_left', 'phase1_d', 'engine_d', 'phase1_mop_up', 'engine_mop_up', 'phase1_d_minus_mop_up', 'engine_d_minus_mop_up'])
print(d_check.round(3).to_string(index=False))
print(f'standard error of each engine win rate: at most {0.5 / np.sqrt(n_each):.4f}')
####
# D runs per half-inning over 1000 simulated half-innings: min 1 max 1
#  lead  inning  half_innings_left  phase1_d  engine_d  phase1_mop_up  engine_mop_up  phase1_d_minus_mop_up  engine_d_minus_mop_up
#     2       9                  1     1.000     1.000          0.921          0.925                  0.079                  0.075
#     1       9                  1     0.630     0.651          0.831          0.837                 -0.201                 -0.186
#     3       7                  3     0.796     0.802          0.886          0.894                 -0.091                 -0.092
#     0       7                  3     0.169     0.188          0.480          0.511                 -0.311                 -0.323
#     5       5                  5     0.892     0.894          0.931          0.943                 -0.039                 -0.049
#    -5       6                  4     0.007     0.009          0.042          0.052                 -0.035                 -0.042
# standard error of each engine win rate: at most 0.0065
####
# D concedes exactly one run in every half-inning. Played out through the engine, D minus a mop-up reliever agrees with the Phase 1 arithmetic within 1.5 points in all six states, and the lock state (a 2-run lead with one half-inning left) is a certain win in both.

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
x = np.arange(len(group_check))
axes[0].bar(x - 0.2, group_check['yardstick'], width=0.4, color=real_color, label='real')
axes[0].bar(x + 0.2, group_check['simulated'], width=0.4, color=sim_color, label='simulated')
axes[0].set_xticks(x, [g.replace('_', '-').replace(' ', '\n') for g in group_check['group']])
axes[0].set_ylabel('runs per batter faced')
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
axes[0].set_title('pitcher groups (real: runs charged; on the mound\nfor mop-up and position players)', fontsize=9, color=ink)
labels = [f'{l:+d}, {n} left' for l, n in zip(d_check['lead'], d_check['half_innings_left'])]
x = np.arange(len(d_check))
axes[1].axhline(0, color=ink_muted, lw=1)
axes[1].bar(x - 0.2, 100 * d_check['phase1_d_minus_mop_up'], width=0.4, color=real_color, label='Phase 1 arithmetic')
axes[1].bar(x + 0.2, 100 * d_check['engine_d_minus_mop_up'], width=0.4, color=sim_color, label='engine')
axes[1].set_xticks(x, labels, fontsize=7)
axes[1].set_xlabel('lead when D enters, half-innings left\n(home team, middle offense)')
axes[1].set_ylabel('D minus mop-up, win probability (points)')
axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title('D against the Phase 1 calculation', fontsize=9, color=ink)
fig.suptitle('Figure 11. Pitcher quality and D inside the engine', x=0.01, ha='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure11_engine_pitchers.png', dpi=200)
plt.close(fig)


# Step 11 What does and does not explain the extra extra-inning games
# The game factor fixed the spread of team scores but not the excess of games tied after nine (Step 6). Two candidate explanations, tested.
# Candidate 1, score-dependent bullpen use: real teams protect leads with their best relievers and give up on deficits with their worst, which might push scores apart. A crude usage rule (starters through the 5th, top relievers with a lead of 0 to 3 or down 1 in the 8th or later, mop-up men at 5 or more either way, other relievers otherwise) against league-average pitching everywhere.
def crude_usage(side, inning, lead):
    if inning <= 5:
        return 'starter'
    if 0 <= lead <= 3 or (lead == -1 and inning >= 8):
        return 'top reliever'
    if abs(lead) >= 5:
        return 'mop_up'
    return 'other reliever'


usage_rows = []
for label, plan in [('league everywhere', league_plan), ('crude score-dependent usage', crude_usage)]:
    results = [engine.play_game(inputs, vt, ht, plan) for vt, ht in zip(*real_games.sample(n=30000, replace=True, random_state=4)[['visitor_tier', 'home_tier']].values.T)]
    v = np.array([r['score_v'] for r in results])
    h = np.array([r['score_h'] for r in results])
    usage_rows.append([label, (np.array([r['innings'] for r in results]) > 9).mean(), np.concatenate([v, h]).mean(), (v - h).std()])
usage_rows.append(['real', (real_innings > 9).mean(), pd.concat([real_final_v, real_final_h]).mean(), (real_final_v - real_final_h).std()])
print(pd.DataFrame(usage_rows, columns=['games', 'extra_inning_share', 'runs_per_team_game', 'sd_of_final_margin']).round(4).to_string(index=False))
# Candidate 2, the two teams' scores moving together: correlation of the visitor's nine-inning score with the home team's eight-inning score (the bottom of the 9th is cut short or skipped, so it is left out of both)
real_reg_v = plays[(plays['top_bot'] == 0) & (plays['inning'] <= 9)].groupby('gid')['runs'].sum()
real_reg_h = plays[(plays['top_bot'] == 1) & (plays['inning'] <= 8)].groupby('gid')['runs'].sum().reindex(real_reg_v.index, fill_value=0)
sim_reg_v = sim_halves[(sim_halves['top_bot'] == 0) & (sim_halves['inning'] <= 9)].groupby('game')['runs'].sum()
sim_reg_h = sim_halves[(sim_halves['top_bot'] == 1) & (sim_halves['inning'] <= 8)].groupby('game')['runs'].sum().reindex(sim_reg_v.index, fill_value=0)
print(f'correlation of the two teams\' regulation scores: real {np.corrcoef(real_reg_v, real_reg_h)[0, 1]:+.3f}, simulated {np.corrcoef(sim_reg_v, sim_reg_h)[0, 1]:+.3f}')
####
#                       games  extra_inning_share  runs_per_team_game  sd_of_final_margin
#           league everywhere              0.0972              4.4576              4.2956
# crude score-dependent usage              0.1040              4.3574              4.3299
#                        real              0.0868              4.4367              4.4807
# correlation of the two teams' regulation scores: real -0.033, simulated +0.003
####
# Both candidates fail. Score-dependent bullpen use makes ties more common, not less (10.4 percent against 9.7), because putting top relievers into close games suppresses scoring in exactly the games that are level. The real scores are slightly negatively correlated (-0.033) where the engine's are not (+0.003), and real final margins are wider (standard deviation 4.48 against 4.30); together those explain only part of the gap. I have not found the rest.
# Consequence for Phase 4: about 1.3 points too many extra-inning games, roughly two a team-season, which slightly overstates bullpen demand. Small next to the effects Phase 4 is after, but it goes on the list of things to recheck once Phase 4 has real starters and a usage rule. The game factor will need recalibrating then too, since varying starter quality supplies part of the day-to-day spread it now stands in for.

scoring_check.to_csv(results_dir / 'phase3_scoring_check.csv', index=False)
group_check.to_csv(results_dir / 'phase3_group_check.csv', index=False)
d_check.to_csv(results_dir / 'phase3_d_check.csv', index=False)
