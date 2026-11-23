#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:10:00 2026

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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

# Phase 2 of the deterministic pitcher project (the plan is in ../PLAN.md, Phase 1 is phase1_analytic.py)
# Phase 1 found the usage rule (bring D in when our lead is greater than the half-innings left, which I call a lock) and that it adds about 2.5 wins a season by itself. This phase prices the rest of the ledger:
#   1. the blowout-loss use: what it costs in wins to also bring D in when we trail by a lot, and how many innings that takes off the staff
#   2. the roster spot: D takes the 13th pitcher's place, so does he take more work off the staff than that pitcher was doing?
#   3. the rest: what the innings and appearances D takes off everyone else are worth in wins
# The rest is priced one way only here, the cheapest one to measure: a reliever who pitched yesterday allows more runs today (measured in Step 9), so every appearance D removes can turn a later back-to-back appearance into a rested one. Injury risk, starters going deeper, and a manager re-planning his bullpen around D are left for Phase 4, so this is a floor on the value of rest, not an estimate of all of it.
# Data: Retrosheet play-by-play and per-pitcher game lines (the season CSV files), 2022 to 2025, regular season.
# The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

project_dir = pathlib.Path(__file__).resolve().parent.parent
csv_dir = project_dir / 'data' / 'retrosheet' / 'csv'
fig_dir = project_dir / 'figures'
results_dir = project_dir / 'results'
years = [2022, 2023, 2024, 2025]
tiers = ['low', 'middle', 'high']
rng = np.random.default_rng(20261014)

surface = '#fcfcfb'
ink = '#0b0b0b'
ink_secondary = '#52514e'
ink_muted = '#898781'
plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface, 'axes.edgecolor': ink_muted, 'axes.labelcolor': ink_secondary, 'xtick.color': ink_muted, 'ytick.color': ink_muted, 'text.color': ink, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


# Step 1 Read the play-by-play, the per-pitcher game lines and the player files
plays_list = []
pitching_list = []
players_list = []
for year in years:
    p = pd.read_csv(csv_dir / str(year) / f'{year}plays.csv', usecols=['gid', 'inning', 'top_bot', 'batteam', 'pitteam', 'score_v', 'score_h', 'pitcher', 'outs_pre', 'outs_post', 'runs', 'nump', 'pa', 'gametype'], low_memory=False)
    p = p[p['gametype'] == 'regular'].drop(columns='gametype')
    p['year'] = year
    plays_list.append(p)
    pt = pd.read_csv(csv_dir / str(year) / f'{year}pitching.csv', usecols=['gid', 'id', 'team', 'p_seq', 'p_ipouts', 'p_bfp', 'p_r', 'p_gs', 'date', 'number', 'gametype'])
    pt = pt[pt['gametype'] == 'regular'].drop(columns='gametype')
    pt['year'] = year
    pitching_list.append(pt)
    ap = pd.read_csv(csv_dir / str(year) / f'{year}allplayers.csv', usecols=['id', 'team', 'g', 'g_p'])
    ap['year'] = year
    players_list.append(ap)
plays = pd.concat(plays_list, ignore_index=True)
pitching = pd.concat(pitching_list, ignore_index=True)
players = pd.concat(players_list, ignore_index=True)
plays['outs'] = (plays['outs_post'] - plays['outs_pre']).clip(lower=0)
# The pitch count is repeated on rows that do not end a plate appearance (a wild pitch mid at-bat), so count it only where the plate appearance ends
plays['pitches'] = plays['nump'].fillna(0).where(plays['pa'] == 1, 0)

# Keep the same games as Phase 1: complete regulation games, meaning the visitor batted in the 9th
has_top_ninth = plays[(plays['inning'] == 9) & (plays['top_bot'] == 0)]['gid'].unique()
plays = plays[plays['gid'].isin(has_top_ninth)]
pitching = pitching[pitching['gid'].isin(has_top_ninth)]
last_play = plays.groupby('gid').tail(1).set_index('gid')
final_v = last_play['score_v'] + last_play['runs'].where(last_play['top_bot'] == 0, 0)
final_h = last_play['score_h'] + last_play['runs'].where(last_play['top_bot'] == 1, 0)
visitor_team = plays[plays['top_bot'] == 0].groupby('gid')['batteam'].first()
home_team = plays[plays['top_bot'] == 0].groupby('gid')['pitteam'].first()
winner = pd.Series(np.where(final_v > final_h, visitor_team.loc[final_v.index], home_team.loc[final_v.index]), index=final_v.index)

# A position player pitching: someone who played more games at other positions than at pitcher that season (All-Star rosters excluded). A start is always a starter, which keeps a two-way starter out of this group.
players = players[~players['team'].isin(['ALS', 'NLS'])].groupby(['year', 'id'])[['g', 'g_p']].sum().reset_index()
position_players = set(zip(players.loc[players['g_p'] < 0.5 * players['g'], 'year'], players.loc[players['g_p'] < 0.5 * players['g'], 'id']))
pitching['role'] = np.where(pitching['p_seq'] == 1, 'starter', np.where([(y, i) in position_players for y, i in zip(pitching['year'], pitching['id'])], 'position player', 'reliever'))
print('games:', plays['gid'].nunique(), ' plays:', len(plays), ' pitcher appearances:', len(pitching))
print('appearances by role:', pitching['role'].value_counts().to_dict())
print('ties after the last play:', (final_v == final_h).sum())
####
# games: 9697  plays: 757246  pitcher appearances: 82964
# appearances by role: {'reliever': 63104, 'starter': 19394, 'position player': 466}
# ties after the last play: 0
####
# The same 9,697 games as Phase 1. Position players made 466 pitching appearances in four seasons.


# Step 2 The Phase 1 model: per-inning run distributions and win probability
with open(results_dir / 'phase1_pmfs.json') as f:
    saved_pmfs = json.load(f)
tier_inning_pmf = {tier: {int(i): np.array(v) for i, v in saved_pmfs['tier_inning_pmf'][tier].items()} for tier in tiers}
league_inning_pmf = {int(i): np.array(v) for i, v in saved_pmfs['league_inning_pmf'].items()}
mop_up_pmf = np.array(saved_pmfs['mop_up_pmf'])
team_seasons = pd.read_csv(results_dir / 'phase1_team_seasons.csv')
tier_lookup = team_seasons.set_index(['year', 'team'])['tier']


def add_innings(pmf_list):
    total = np.array([1.0])
    for pmf in pmf_list:
        total = np.convolve(total, pmf)
    return total


def win_probability(lead, our_total_pmf, their_total_pmf):
    # Chance that lead + our runs - their runs ends above zero; a tie after nine counts as half a win
    margin_pmf = np.convolve(our_total_pmf, their_total_pmf[::-1])
    margin = lead + np.arange(len(margin_pmf)) - (len(their_total_pmf) - 1)
    return margin_pmf[margin > 0].sum() + 0.5 * margin_pmf[margin == 0].sum()


# Our remaining batting, from the start of our fielding half of inning i: the home team (fielding the top) still bats in innings i to 9, the visitor (fielding the bottom) in i+1 to 9
our_remaining = {(tier, i, perspective): add_innings([tier_inning_pmf[tier][j] for j in range(i if perspective == 'home' else i + 1, 10)]) for tier in tiers for i in range(1, 10) for perspective in ['home', 'visitor']}
league_remaining = {i: add_innings([league_inning_pmf[j] for j in range(i, 10)]) for i in range(1, 10)}
mop_up_remaining = {i: add_innings([mop_up_pmf] * (10 - i)) for i in range(1, 10)}
d_remaining = {i: np.eye(11 - i)[10 - i] for i in range(1, 10)}
wp_cache = {}


def fielding_wp(tier, lead, i, perspective, pitcher):
    # Win probability for the fielding team at the start of its half of inning i, with real pitching ('league'), a mop-up reliever ('mop_up') or D ('d') for the rest of regulation
    key = (tier, lead, i, perspective, pitcher)
    if key not in wp_cache:
        their = {'league': league_remaining, 'mop_up': mop_up_remaining, 'd': d_remaining}[pitcher][i]
        wp_cache[key] = win_probability(lead, our_remaining[(tier, i, perspective)], their)
    return wp_cache[key]


def run_value(tier, lead, i, perspective):
    # What one more run allowed costs the fielding team in win probability, at the start of its half of inning i (extra innings are treated as a 9th inning)
    i = min(i, 9)
    lead = int(np.clip(lead, -12, 12))
    return fielding_wp(tier, lead, i, perspective, 'league') - fielding_wp(tier, lead - 1, i, perspective, 'league')


# Step 3 Half-innings, each pitcher's work in each half-inning, and where each appearance began
half = plays.groupby(['gid', 'inning', 'top_bot'], sort=False).agg(year=('year', 'first'), team=('pitteam', 'first'), score_v=('score_v', 'first'), score_h=('score_h', 'first'), runs=('runs', 'sum')).reset_index()
half['perspective'] = np.where(half['top_bot'] == 0, 'home', 'visitor')
half['lead'] = np.where(half['top_bot'] == 0, half['score_h'] - half['score_v'], half['score_v'] - half['score_h'])
half['tier'] = [tier_lookup[(y, t)] for y, t in zip(half['year'], half['team'])]
half['won'] = half['team'].values == winner.loc[half['gid']].values
regulation = half[half['inning'] <= 9].copy()

work = plays.groupby(['gid', 'pitteam', 'pitcher', 'inning'], sort=False).agg(outs=('outs', 'sum'), pitches=('pitches', 'sum')).reset_index()
entry = plays.groupby(['gid', 'pitcher'], sort=False).agg(entry_inning=('inning', 'first'), entry_top_bot=('top_bot', 'first'), entry_score_v=('score_v', 'first'), entry_score_h=('score_h', 'first')).reset_index().rename(columns={'pitcher': 'id'})
pitching = pitching.merge(entry, on=['gid', 'id'], how='left')
work = work.merge(pitching[['gid', 'id', 'role']].rename(columns={'id': 'pitcher'}), on=['gid', 'pitcher'], how='left')
print('appearances with no play in the play-by-play (entered and faced no one):', pitching['entry_inning'].isna().sum())
print('work rows with no matching pitcher line:', work['role'].isna().sum())
####
# appearances with no play in the play-by-play (entered and faced no one): 0
# work rows with no matching pitcher line: 13
####
# Every appearance has at least one play, and 13 rows of in-game work (out of several hundred thousand) have no matching pitcher line, which is too few to matter.


# Step 4 How much a run was worth at the moment each pitcher came in, and a check that it adds up to a sensible runs-per-win figure
# This is the leverage measure for everything below: the drop in the fielding team's win probability from one more run, at the start of the half-inning the pitcher entered, with the lead as it stood when he came in. It ignores outs and runners on base, so it is a coarse version of the usual leverage index.
pitching['tier'] = [tier_lookup[(y, t)] for y, t in zip(pitching['year'], pitching['team'])]
pitching['entry_perspective'] = np.where(pitching['entry_top_bot'] == 0, 'home', 'visitor')
pitching['entry_lead'] = np.where(pitching['entry_top_bot'] == 0, pitching['entry_score_h'] - pitching['entry_score_v'], pitching['entry_score_v'] - pitching['entry_score_h'])
has_entry = pitching['entry_inning'].notna()
pitching['run_value'] = np.nan
pitching.loc[has_entry, 'run_value'] = [run_value(t, l, int(i), p) for t, l, i, p in zip(pitching.loc[has_entry, 'tier'], pitching.loc[has_entry, 'entry_lead'], pitching.loc[has_entry, 'entry_inning'], pitching.loc[has_entry, 'entry_perspective'])]
regulation['run_value'] = [run_value(t, l, i, p) for t, l, i, p in zip(regulation['tier'], regulation['lead'], regulation['inning'], regulation['perspective'])]
mean_rv_half = regulation['run_value'].mean()
mean_rv_runs = (regulation['run_value'] * regulation['runs']).sum() / regulation['runs'].sum()
print(f'mean win value of a run: over all half-innings {mean_rv_half:.4f} (about {1 / mean_rv_half:.1f} runs per win), over the half-innings where runs actually scored {mean_rv_runs:.4f} (about {1 / mean_rv_runs:.1f} runs per win)')
print('mean win value of a run when the pitcher entered, by role:', pitching.groupby('role')['run_value'].mean().round(4).to_dict())

# Each team-season's top relievers: the three with the highest average run value at entry, among relievers with at least 20 appearances for that team (the closer and the main setup men)
relievers = pitching[pitching['role'] == 'reliever']
reliever_seasons = relievers.groupby(['year', 'team', 'id']).agg(appearances=('gid', 'size'), mean_run_value=('run_value', 'mean')).reset_index()
eligible_relievers = reliever_seasons[reliever_seasons['appearances'] >= 20].sort_values('mean_run_value', ascending=False)
top_relievers = set(map(tuple, eligible_relievers.groupby(['year', 'team']).head(3)[['year', 'team', 'id']].values))
pitching['role_detail'] = pitching['role']
pitching.loc[[(r == 'reliever') and ((y, t, i) in top_relievers) for r, y, t, i in zip(pitching['role'], pitching['year'], pitching['team'], pitching['id'])], 'role_detail'] = 'top reliever'
pitching.loc[pitching['role_detail'] == 'reliever', 'role_detail'] = 'other reliever'
work = work.drop(columns='role').merge(pitching[['gid', 'id', 'role_detail']].rename(columns={'id': 'pitcher'}), on=['gid', 'pitcher'], how='left')
print('mean win value of a run at entry, top relievers', round(pitching.loc[pitching['role_detail'] == 'top reliever', 'run_value'].mean(), 4), ' other relievers', round(pitching.loc[pitching['role_detail'] == 'other reliever', 'run_value'].mean(), 4))
####
# mean win value of a run: over all half-innings 0.0990 (about 10.1 runs per win), over the half-innings where runs actually scored 0.0965 (about 10.4 runs per win)
# mean win value of a run when the pitcher entered, by role: {'position player': 0.0017, 'reliever': 0.1116, 'starter': 0.0982}
# mean win value of a run at entry, top relievers 0.157  other relievers 0.0923
####
# Check passed: the Phase 1 model, which was never fitted to wins, implies about 10.1 runs per win, in line with the commonly quoted figure of about 10. The measure also sorts roles the way it should: position players come in when a run is worth almost nothing (0.0017 wins), and the top three relievers on each team come in when a run is worth 70 percent more than for the rest of the bullpen (0.157 against 0.092).


# Step 5 When D would come in, under each usage rule
# Lock: the first fielding half-inning where our lead is greater than the half-innings left (Phase 1). D pitches the rest of the game and the game cannot reach extra innings.
# Blowout (T, I): the first fielding half-inning from the Ith inning on where we trail by T or more. D pitches the rest of regulation; if we come all the way back to a tie, the real bullpen pitches the extra innings.
team_games = regulation.groupby(['gid', 'team']).agg(year=('year', 'first'), tier=('tier', 'first'), won=('won', 'first'), perspective=('perspective', 'first')).reset_index()
locks = regulation[regulation['lead'] > 10 - regulation['inning']].groupby(['gid', 'team'])['inning'].min().rename('lock_inning')
team_games = team_games.merge(locks, on=['gid', 'team'], how='left')
blowout_rules = [(t, i) for i in [5, 6, 7] for t in [4, 5, 6, 7, 8]]
for t, i in blowout_rules:
    first = regulation[(regulation['inning'] >= i) & (regulation['lead'] <= -t)].groupby(['gid', 'team'])['inning'].min().rename(f'blowout_{t}_{i}')
    team_games = team_games.merge(first, on=['gid', 'team'], how='left')
print('team-games:', len(team_games), ' with a lock:', team_games['lock_inning'].notna().sum(), ' lock games lost by the real team:', (team_games['lock_inning'].notna() & ~team_games['won']).sum())
print('lock wins D adds per team-season:', round((team_games['lock_inning'].notna() & ~team_games['won']).sum() / team_seasons.shape[0], 2), '(Phase 1 from the game logs: 2.5)')
####
# team-games: 19394  with a lock: 7511  lock games lost by the real team: 300
# lock wins D adds per team-season: 2.5 (Phase 1 from the game logs: 2.5)
####
# Check passed: the play-by-play gives the same 300 lost locks over 120 team-seasons, 2.5 a season, as the Phase 1 game logs.


# Step 6 The cost of also using D in blowout losses
# For each team-game where the blowout rule fires (and no lock came first), the cost is the model's win probability with a mop-up reliever minus its win probability with D, at the moment D would enter. Check: the mop-up model's win probability against how often those teams actually came back to win.
state_index = regulation.set_index(['gid', 'team', 'inning'])[['lead']]
fielding_innings = regulation.groupby(['gid', 'team'])['inning'].apply(np.array)


def halves_from(gids, teams, entry_innings):
    # The fielding half-innings D would actually pitch: every one of ours from his entry through the 9th that was played. A trailing visitor never fields the bottom of the 9th, so 10 minus the entry inning would overcount.
    # First version: d_innings = (10 - entry_inning).sum(), which counted that unplayed bottom of the 9th (73 blowout innings a season at 5+ from the 6th where the outs recorded say 62)
    return np.array([(fielding_innings[(g, t)] >= e).sum() for g, t, e in zip(gids, teams, entry_innings)])


blowout_rows = []
for t, i in blowout_rules:
    column = f'blowout_{t}_{i}'
    fired = team_games[team_games[column].notna() & (team_games['lock_inning'].isna() | (team_games['lock_inning'] > team_games[column]))].copy()
    fired['entry_inning'] = fired[column].astype(int)
    fired['lead'] = state_index.loc[list(zip(fired['gid'], fired['team'], fired['entry_inning'])), 'lead'].values
    fired['wp_mop_up'] = [fielding_wp(tr, l, e, p, 'mop_up') for tr, l, e, p in zip(fired['tier'], fired['lead'], fired['entry_inning'], fired['perspective'])]
    fired['wp_d'] = [fielding_wp(tr, l, e, p, 'd') for tr, l, e, p in zip(fired['tier'], fired['lead'], fired['entry_inning'], fired['perspective'])]
    seasons = team_seasons.shape[0]
    blowout_rows.append([t, i, len(fired) / seasons, halves_from(fired['gid'], fired['team'], fired['entry_inning']).sum() / seasons, (fired['wp_mop_up'] - fired['wp_d']).sum() / seasons, fired['won'].mean(), fired['wp_mop_up'].mean(), fired['wp_d'].mean()])
blowout = pd.DataFrame(blowout_rows, columns=['trailing_by', 'from_inning', 'games_per_season', 'd_innings_per_season', 'wins_cost_per_season', 'real_comeback_rate', 'model_mop_up_rate', 'model_d_rate'])
blowout['wins_cost_per_100_innings'] = 100 * blowout['wins_cost_per_season'] / blowout['d_innings_per_season']
print(blowout.round(3).to_string(index=False))
####
#  trailing_by  from_inning  games_per_season  d_innings_per_season  wins_cost_per_season  real_comeback_rate  model_mop_up_rate  model_d_rate  wins_cost_per_100_innings
#            4            5            34.142               113.292                 1.003               0.031              0.036         0.007                      0.885
#            5            5            23.767                74.550                 0.374               0.016              0.019         0.003                      0.502
#            6            5            16.133                48.667                 0.137               0.008              0.010         0.002                      0.281
#            7            5            10.475                30.033                 0.044               0.004              0.005         0.001                      0.148
#            8            5             6.908                18.933                 0.015               0.002              0.002         0.000                      0.078
#            4            6            32.917                93.008                 0.712               0.025              0.028         0.006                      0.765
#            5            6            23.108                62.533                 0.265               0.013              0.014         0.003                      0.423
#            6            6            15.717                41.308                 0.096               0.007              0.007         0.001                      0.231
#            7            6            10.275                25.925                 0.030               0.002              0.004         0.001                      0.116
#            8            6             6.742                16.350                 0.010               0.001              0.002         0.000                      0.059
#            4            7            31.167                68.425                 0.425               0.014              0.019         0.005                      0.622
#            5            7            21.842                46.525                 0.148               0.008              0.009         0.002                      0.318
#            6            7            14.967                31.383                 0.052               0.005              0.005         0.001                      0.167
#            7            7             9.900                20.242                 0.016               0.000              0.002         0.000                      0.081
#            8            7             6.517                13.125                 0.005               0.000              0.001         0.000                      0.040
####
# The blowout use costs wins at every threshold, and the price rises steeply with how small the deficit is: 0.06 wins per 100 innings at 8 or more from the 6th, 0.42 at 5 or more, 0.77 at 4 or more. When D starts matters much less than the deficit (in Figure 6 the three curves lie nearly on top of each other).
# The model's comeback rate with a mop-up reliever sits slightly above the real comeback rate (0.028 against 0.025 at 4 or more from the 6th), the same lean Phase 1 found for teams that are behind, so these costs are probably overstated, by roughly 10 to 35 percent at the 4-run and 5-run thresholds (the 6-run to 8-run thresholds have too few comebacks to tell). Cutting the costs by a third does not change the conclusion in Step 10.

fig, ax = plt.subplots(figsize=(6.2, 4))
inning_colors = {5: '#86b6ef', 6: '#2a78d6', 7: '#104281'}
for i in [5, 6, 7]:
    b = blowout[blowout['from_inning'] == i]
    ax.plot(b['d_innings_per_season'], b['wins_cost_per_season'], 'o-', color=inning_colors[i], lw=2, ms=5)
    ax.annotate(f'from the {i}th', (b['d_innings_per_season'].iloc[0], b['wins_cost_per_season'].iloc[0]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color=ink_secondary)
    for _, row in b.iterrows():
        ax.annotate(f'{int(row["trailing_by"])}', (row['d_innings_per_season'], row['wins_cost_per_season']), xytext=(0, -11), textcoords='offset points', ha='center', fontsize=7, color=ink_muted)
ax.set_xlabel('innings D pitches in blowout losses, per season')
ax.set_ylabel('wins lost per season')
ax.set_title('Figure 6. The price of the blowout-loss use (points labelled by deficit)', loc='left', fontsize=10, color=ink)
fig.tight_layout()
fig.savefig(fig_dir / 'figure6_blowout_tradeoff.png', dpi=200)
plt.close(fig)


# Step 7 Who would have thrown D's innings
# Under a usage rule, every appearance that began at or after D's entry is removed, and the work of a pitcher who was already in the game is cut off at that point
def removed_under(entry_inning, cap_at_nine):
    # entry_inning: the inning D enters, by (gid, team); cap_at_nine: True for the blowout use, where extra innings still go to the real bullpen
    e = entry_inning.dropna().rename('d_entry').reset_index()
    w = work.merge(e, left_on=['gid', 'pitteam'], right_on=['gid', 'team'], how='inner')
    w = w[(w['inning'] >= w['d_entry']) & ((w['inning'] <= 9) | (not cap_at_nine))]
    a = pitching.merge(e, on=['gid', 'team'], how='inner')
    a = a[(a['entry_inning'] >= a['d_entry']) & ((a['entry_inning'] <= 9) | (not cap_at_nine))]
    return w, a


def policy_entries(blowout_rule):
    # The inning D enters under the lock rule plus, if given, a blowout rule; whichever fires first. Returns the lock entries and the blowout entries separately, since the blowout use leaves extra innings to the bullpen.
    lock_entry = team_games.set_index(['gid', 'team'])['lock_inning']
    if blowout_rule is None:
        return lock_entry, lock_entry * np.nan
    blow_entry = team_games.set_index(['gid', 'team'])[f'blowout_{blowout_rule[0]}_{blowout_rule[1]}']
    blow_first = blow_entry.notna() & (lock_entry.isna() | (blow_entry < lock_entry))
    return lock_entry.where(~blow_first), blow_entry.where(blow_first)


who_rows = []
role_order = ['starter', 'top reliever', 'other reliever', 'position player']
for label, rule in [('lock', None), ('blowout, trailing by 5 from the 6th', (5, 6))]:
    lock_entry, blow_entry = policy_entries(rule)
    entries = lock_entry if rule is None else blow_entry
    w, a = removed_under(entries, cap_at_nine=rule is not None)
    seasons = team_seasons.shape[0]
    for role in role_order:
        who_rows.append([label, role, w.loc[w['role_detail'] == role, 'outs'].sum() / 3 / seasons, w.loc[w['role_detail'] == role, 'pitches'].sum() / seasons, (a['role_detail'] == role).sum() / seasons])
who = pd.DataFrame(who_rows, columns=['use', 'role', 'innings_per_season', 'pitches_per_season', 'appearances_removed_per_season'])
print(who.round(1).to_string(index=False))
####
#                                 use            role  innings_per_season  pitches_per_season  appearances_removed_per_season
#                                lock         starter                31.4               498.0                             0.0
#                                lock    top reliever                51.0               808.3                            50.7
#                                lock  other reliever                83.6              1368.3                            74.3
#                                lock position player                 0.3                 5.5                             0.4
# blowout, trailing by 5 from the 6th         starter                 0.9                13.6                             0.0
# blowout, trailing by 5 from the 6th    top reliever                 4.8                78.1                             4.7
# blowout, trailing by 5 from the 6th  other reliever                53.2               880.1                            29.4
# blowout, trailing by 5 from the 6th position player                 3.6                54.1                             3.4
####
# The surprise is the top relievers. About a third of D's lock innings (51 of 166) and 51 appearances a season come from each team's top three relievers, because a lock includes every 9th inning with a lead of 2 or more and every 8th with 3 or more, which is where closers and setup men pitch. Starters lose about 31 innings and 500 pitches a season, from games where the lock arrives early.
# Blowout innings are almost all other relievers (53 of 62), plus 3.6 innings a season of position-player pitching.

role_colors = {'starter': '#2a78d6', 'top reliever': '#eb6834', 'other reliever': '#1baf7a', 'position player': '#eda100'}
fig, ax = plt.subplots(figsize=(8, 2.6))
for row_i, label in enumerate(['lock', 'blowout, trailing by 5 from the 6th']):
    left = 0
    for role in role_order:
        value = who.loc[(who['use'] == label) & (who['role'] == role), 'innings_per_season'].iloc[0]
        ax.barh(row_i, value, left=left, color=role_colors[role], edgecolor=surface, linewidth=2, height=0.55, label=role if row_i == 0 else None)
        if value >= 8:
            ax.text(left + value / 2, row_i, f'{value:.0f}', ha='center', va='center', fontsize=8, color='white' if role in ['starter', 'top reliever'] else ink)
        left += value
ax.set_yticks([0, 1], ['lock', 'blowout (5+ from the 6th)'])
ax.invert_yaxis()
ax.grid(axis='y', visible=False)
ax.set_xlabel('innings per team-season that D would take')
ax.legend(frameon=False, ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0))
ax.set_title('Figure 7. Whose innings D would take', loc='left', fontsize=10, color=ink, pad=24)
fig.tight_layout()
fig.savefig(fig_dir / 'figure7_whose_innings.png', dpi=200)
plt.close(fig)


# Step 8 The 13th pitcher: does D take more work off the bullpen than the roster spot he uses was doing?
# The 13th pitcher is the least-used man in the bullpen, so an average bullpen slot (relief innings and appearances divided by the 8 relief spots) is an upper bound on his workload
relief = pitching[pitching['role'] == 'reliever']
slot = relief.groupby(['year', 'team']).agg(relief_innings=('p_ipouts', lambda x: x.sum() / 3), relief_appearances=('gid', 'size')).reset_index()
slot['slot_innings'] = slot['relief_innings'] / 8
slot['slot_appearances'] = slot['relief_appearances'] / 8
lock_entry, _ = policy_entries(None)
w_lock, a_lock = removed_under(lock_entry, cap_at_nine=False)
removed_relief = w_lock[w_lock['role_detail'].isin(['top reliever', 'other reliever'])].groupby([w_lock['gid'].str[3:7].astype(int).rename('year'), 'pitteam'])['outs'].sum().div(3).rename('relief_innings_removed').reset_index().rename(columns={'pitteam': 'team'})
removed_apps = a_lock[a_lock['role'] == 'reliever'].groupby(['year', 'team']).size().rename('relief_appearances_removed').reset_index()
slot = slot.merge(removed_relief, on=['year', 'team'], how='left').merge(removed_apps, on=['year', 'team'], how='left').fillna(0)
print(slot[['relief_innings', 'slot_innings', 'relief_innings_removed', 'relief_appearances', 'slot_appearances', 'relief_appearances_removed']].mean().round(1).to_string())
print('team-seasons where the lock use alone removes more relief innings than an average bullpen slot pitched:', (slot['relief_innings_removed'] > slot['slot_innings']).mean().round(3), ' more relief appearances:', (slot['relief_appearances_removed'] > slot['slot_appearances']).mean().round(3))
####
# relief_innings                591.5
# slot_innings                   73.9
# relief_innings_removed        134.5
# relief_appearances            525.9
# slot_appearances               65.7
# relief_appearances_removed    125.0
# team-seasons where the lock use alone removes more relief innings than an average bullpen slot pitched: 0.975  more relief appearances: 0.975
####
# D more than covers the roster spot he takes. The lock use alone takes 134.5 relief innings and 125 relief appearances a season off the bullpen, about 1.8 to 1.9 times what an average bullpen slot throws, and more than a slot in 97.5 percent of team-seasons (117 of 120). The 13th pitcher, being the least-used man, throws less than an average slot, so the 12 real pitchers left would be doing less relief work than the 13 were.


# Step 9 What pitching on consecutive days costs a reliever
# Back-to-back: the reliever also pitched yesterday (or earlier the same day, in a doubleheader). The comparison is within each pitcher-season (each pitcher against himself that year), so it is not just that the relievers who pitch back to back are better or worse ones.
# What it does not remove: back-to-back outings lean toward close late games and the best relievers, and the opposing hitters may differ. I have no clean way around that here, so the estimate is a measured association, not a proven cause.
relief = relief.sort_values(['id', 'date', 'number']).copy()
relief['day'] = pd.to_datetime(relief['date'].astype(str))
relief['previous_day'] = relief.groupby('id')['day'].shift(1)
relief['back_to_back'] = (relief['day'] - relief['previous_day']).dt.days <= 1
relief_bf = relief[relief['p_bfp'] > 0]
groups = relief_bf.groupby(['year', 'id', 'back_to_back'])[['p_r', 'p_bfp']].sum().unstack('back_to_back')
groups = groups.dropna()
r_b2b = groups[('p_r', True)] / groups[('p_bfp', True)]
r_rest = groups[('p_r', False)] / groups[('p_bfp', False)]
weight = groups[('p_bfp', True)] * groups[('p_bfp', False)] / (groups[('p_bfp', True)] + groups[('p_bfp', False)])
diffs = (r_b2b - r_rest).values
weights = weight.values
penalty = (diffs * weights).sum() / weights.sum()
boot = np.array([(diffs[idx] * weights[idx]).sum() / weights[idx].sum() for idx in [rng.integers(0, len(diffs), len(diffs)) for _ in range(2000)]])
penalty_low, penalty_high = np.percentile(boot, [2.5, 97.5])
bf_per_inning = relief['p_bfp'].sum() / (relief['p_ipouts'].sum() / 3)
print(f'relief appearances: {len(relief)}, back to back: {relief["back_to_back"].mean():.3f} of them')
print(f'pitcher-seasons with both kinds of appearance: {len(diffs)}')
print(f'runs allowed per batter, rested {relief_bf.loc[~relief_bf["back_to_back"], "p_r"].sum() / relief_bf.loc[~relief_bf["back_to_back"], "p_bfp"].sum():.4f}, back to back {relief_bf.loc[relief_bf["back_to_back"], "p_r"].sum() / relief_bf.loc[relief_bf["back_to_back"], "p_bfp"].sum():.4f} (raw, across pitchers)')
print(f'back-to-back penalty within pitcher-season: {penalty:.4f} runs per batter (95% interval {penalty_low:.4f} to {penalty_high:.4f}), about {penalty * bf_per_inning * 9:.2f} runs per 9 innings ({penalty_low * bf_per_inning * 9:.2f} to {penalty_high * bf_per_inning * 9:.2f})')
####
# relief appearances: 63104, back to back: 0.165 of them
# pitcher-seasons with both kinds of appearance: 1591
# runs allowed per batter, rested 0.1158, back to back 0.1176 (raw, across pitchers)
# back-to-back penalty within pitcher-season: 0.0085 runs per batter (95% interval 0.0039 to 0.0130), about 0.33 runs per 9 innings (0.15 to 0.50)
####
# Pitching on consecutive days costs a reliever about 0.33 runs per 9 innings (0.15 to 0.50), measured against himself. The raw comparison across pitchers shows almost nothing (0.1176 against 0.1158), because back-to-back outings go disproportionately to better relievers; that is why the within-pitcher comparison matters. About one relief appearance in six (16.5 percent) is back to back.


# Step 10 What the rest D buys, and what losing the 13th pitcher takes back
# Rest dividend: take out every relief appearance D removes, recompute which of the remaining relief appearances are back to back, and value each one that becomes rested at (penalty x batters faced x win value of a run when he came in). Usage is held fixed: the same pitchers pitch the same remaining appearances, only better rested.
# 13th pitcher: with D on the roster, the 13th pitcher's appearances that D does not cover have to go to someone else in the bullpen. I approximate that at an average bullpen slot's appearances (the upper bound from Step 8), scaled by the share of appearances D does not remove, each one back to back at the league rate, at the average batters faced and run value of a relief appearance. Moving his innings to better pitchers would gain a little; I leave that out, which favours the 13th pitcher.
mean_relief_bf = relief['p_bfp'].mean()
mean_relief_rv = pitching.loc[pitching['role'] == 'reliever', 'run_value'].mean()
b2b_rate = relief['back_to_back'].mean()
relief_key = relief.set_index(['gid', 'id']).index


def rest_dividend(removed_appearances):
    removed_keys = set(zip(removed_appearances['gid'], removed_appearances['id']))
    is_removed = np.array([k in removed_keys for k in relief_key])
    kept = relief[~is_removed].copy()
    kept['previous_day_new'] = kept.groupby('id')['day'].shift(1)
    kept['back_to_back_new'] = (kept['day'] - kept['previous_day_new']).dt.days <= 1
    flipped = kept[kept['back_to_back'] & ~kept['back_to_back_new']]
    per_season = flipped.groupby(['year', 'team']).apply(lambda f: pd.Series({'rested_appearances': len(f), 'rested_top_reliever': (f['role_detail'] == 'top reliever').sum(), 'batter_run_value': (f['p_bfp'] * f['run_value'].fillna(mean_relief_rv)).sum()}), include_groups=False)
    removed_relief_per_season = pd.Series(is_removed, index=relief.index).groupby([relief['year'], relief['team']]).sum().rename('relief_appearances_removed')
    return per_season.reindex(removed_relief_per_season.index).fillna(0).join(removed_relief_per_season)


# An upper bracket on what the rest buys: D frees appearances from the top relievers (Step 7). If the manager spent every one of them on the highest-leverage spots that other relievers currently pitch, each such spot would get a better pitcher. Value = (other relievers' runs per batter - top relievers' runs per batter) x batters faced x win value of a run, over the N highest-leverage appearances by other relievers that D does not remove, where N is the number of top-reliever appearances D frees. This assumes the top reliever is always available for those spots, so it is a ceiling, not an estimate.
top_rate = relief.loc[relief['role_detail'] == 'top reliever', 'p_r'].sum() / relief.loc[relief['role_detail'] == 'top reliever', 'p_bfp'].sum()
other_rate = relief.loc[relief['role_detail'] == 'other reliever', 'p_r'].sum() / relief.loc[relief['role_detail'] == 'other reliever', 'p_bfp'].sum()
talent_gap = other_rate - top_rate
print(f'runs allowed per batter: top relievers {top_rate:.4f}, other relievers {other_rate:.4f}, gap {talent_gap:.4f} (about {talent_gap * bf_per_inning * 9:.2f} runs per 9 innings)')


def redeploy_ceiling(removed_appearances):
    removed_keys = set(zip(removed_appearances['gid'], removed_appearances['id']))
    freed = removed_appearances[removed_appearances['role_detail'] == 'top reliever'].groupby(['year', 'team']).size()
    others = relief[(relief['role_detail'] == 'other reliever') & ~np.array([k in removed_keys for k in relief_key])].copy()
    others['batter_run_value'] = others['p_bfp'] * others['run_value'].fillna(mean_relief_rv)
    others = others.sort_values('run_value', ascending=False)
    ceiling = {}
    for (y, t), o in others.groupby(['year', 'team']):
        ceiling[(y, t)] = talent_gap * o['batter_run_value'].iloc[:int(freed.get((y, t), 0))].sum()
    return np.mean(list(ceiling.values()))


ledger_rows = []
policies = [('lock only', None)] + [(f'lock + blowout {t}+ from the 6th', (t, 6)) for t in [8, 7, 6, 5, 4]]
for label, rule in policies:
    lock_entry, blow_entry = policy_entries(rule)
    _, a_lock_p = removed_under(lock_entry, cap_at_nine=False)
    if rule is None:
        a_all = a_lock_p
    else:
        _, a_blow_p = removed_under(blow_entry, cap_at_nine=True)
        a_all = pd.concat([a_lock_p, a_blow_p])
    dividend = rest_dividend(a_all).join(slot.set_index(['year', 'team'])[['relief_appearances', 'slot_appearances']])
    redistributed = dividend['slot_appearances'] * (1 - dividend['relief_appearances_removed'] / dividend['relief_appearances'])
    slot_cost = redistributed * b2b_rate * penalty * mean_relief_bf * mean_relief_rv
    lock_wins = (lock_entry.notna().values & ~team_games['won'].values).sum() / team_seasons.shape[0]
    blow_cost = 0.0 if rule is None else blowout.loc[(blowout['trailing_by'] == rule[0]) & (blowout['from_inning'] == rule[1]), 'wins_cost_per_season'].iloc[0]
    rest = penalty * dividend['batter_run_value'].mean()
    rest_low = penalty_low * dividend['batter_run_value'].mean()
    rest_high = penalty_high * dividend['batter_run_value'].mean()
    d_innings = sum(halves_from(e.dropna().index.get_level_values('gid'), e.dropna().index.get_level_values('team'), e.dropna().values).sum() for e in [lock_entry, blow_entry]) / team_seasons.shape[0]
    ceiling = redeploy_ceiling(a_all)
    ledger_rows.append([label, d_innings, dividend['relief_appearances_removed'].mean(), dividend['rested_appearances'].mean(), dividend['rested_top_reliever'].mean(), lock_wins, -blow_cost, rest, rest_low, rest_high, -slot_cost.mean(), lock_wins - blow_cost + rest - slot_cost.mean(), ceiling, lock_wins - blow_cost + rest + ceiling - slot_cost.mean()])
ledger = pd.DataFrame(ledger_rows, columns=['policy', 'd_innings', 'relief_apps_removed', 'apps_rested', 'top_reliever_apps_rested', 'lock_wins', 'blowout_cost', 'rest_dividend', 'rest_low', 'rest_high', 'slot_cost', 'net_wins', 'redeploy_ceiling', 'net_with_ceiling'])
print(ledger[['policy', 'd_innings', 'relief_apps_removed', 'apps_rested', 'top_reliever_apps_rested']].round(1).to_string(index=False))
print(ledger[['policy', 'lock_wins', 'blowout_cost', 'rest_dividend', 'rest_low', 'rest_high', 'slot_cost', 'net_wins', 'redeploy_ceiling', 'net_with_ceiling']].round(3).to_string(index=False))
####
# runs allowed per batter: top relievers 0.0963, other relievers 0.1227, gap 0.0264 (about 1.01 runs per 9 innings)
#                         policy  d_innings  relief_apps_removed  apps_rested  top_reliever_apps_rested
#                      lock only      164.9                125.0         15.4                       6.5
# lock + blowout 8+ from the 6th      181.2                132.1         16.0                       6.7
# lock + blowout 7+ from the 6th      190.8                137.5         16.4                       6.8
# lock + blowout 6+ from the 6th      206.2                146.3         16.9                       7.0
# lock + blowout 5+ from the 6th      227.4                159.1         17.5                       7.1
# lock + blowout 4+ from the 6th      257.6                179.2         18.7                       7.5
#                         policy  lock_wins  blowout_cost  rest_dividend  rest_low  rest_high  slot_cost  net_wins  redeploy_ceiling  net_with_ceiling
#                      lock only      2.500        -0.000          0.096     0.044      0.147     -0.038     2.558             1.432             3.990
# lock + blowout 8+ from the 6th      2.500        -0.010          0.099     0.045      0.152     -0.037     2.552             1.453             4.005
# lock + blowout 7+ from the 6th      2.500        -0.030          0.101     0.046      0.155     -0.036     2.535             1.468             4.003
# lock + blowout 6+ from the 6th      2.492        -0.096          0.105     0.048      0.161     -0.036     2.465             1.494             3.959
# lock + blowout 5+ from the 6th      2.492        -0.265          0.109     0.049      0.167     -0.034     2.301             1.530             3.831
# lock + blowout 4+ from the 6th      2.492        -0.712          0.117     0.053      0.180     -0.033     1.864             1.586             3.451
####
# Lock only is the best rule: net 2.56 wins a season at the floor, 2.50 of it from locked wins, 0.10 (0.04 to 0.15) from rest, minus 0.04 for the 13th pitcher's appearances going to others. Adding the blowout use never pays at the floor, and at the ceiling only the 7-run and 8-run versions break even (4.00 and 4.01 against 3.99). Use D in blowouts only on days when the bullpen is short, which is a Phase 4 question.
# The floor value of rest is small because most of the 125 appearances D removes were not followed by an outing the next day; only 15 a season turn a back-to-back appearance into a rested one. The ceiling is large (1.43 wins) because top relievers allow about 1 run per 9 innings less than the rest of the bullpen and the freed appearances would go to the highest-leverage spots other relievers now pitch. It is a ceiling for two reasons: it assumes the top reliever is always available for those spots, and it takes the top-against-other gap at face value, when some of that gap is luck that earned pitchers the top roles in the first place. So the honest range for D with the lock rule is about 2.5 to 4 wins a season, and Phase 4 is where the value of rest gets pinned down.


# Step 11 The ledger, as a figure
component_colors = {'lock_wins': '#2a78d6', 'rest_dividend': '#1baf7a', 'blowout_cost': '#eb6834', 'slot_cost': '#eda100'}
component_labels = {'lock_wins': 'locked wins', 'rest_dividend': 'rest (back-to-back penalty avoided)', 'blowout_cost': 'blowout-loss use', 'slot_cost': "13th pitcher's appearances"}
fig, ax = plt.subplots(figsize=(9.5, 3.8))
for row_i, row in ledger.iterrows():
    right = 0
    left = 0
    for comp in ['lock_wins', 'rest_dividend', 'blowout_cost', 'slot_cost']:
        value = row[comp]
        if value >= 0:
            ax.barh(row_i, value, left=right, color=component_colors[comp], edgecolor=surface, linewidth=2, height=0.6)
            right += value
        else:
            ax.barh(row_i, value, left=left, color=component_colors[comp], edgecolor=surface, linewidth=2, height=0.6)
            left += value
    ax.plot([row['net_wins'], row['net_with_ceiling']], [row_i, row_i], color=ink_muted, lw=1)
    ax.plot(row['net_wins'], row_i, 'D', color=ink, ms=5)
    ax.plot(row['net_with_ceiling'], row_i, 'D', mfc=surface, mec=ink, ms=5)
    ax.annotate(f'{row["net_wins"]:.2f} to {row["net_with_ceiling"]:.2f}', (row['net_with_ceiling'], row_i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=8, color=ink_secondary)
ax.axvline(0, color=ink_muted, lw=1)
ax.set_yticks(range(len(ledger)), ledger['policy'])
ax.invert_yaxis()
ax.grid(axis='y', visible=False)
ax.set_xlabel('wins per team-season\nfilled diamond: net; open diamond: ceiling, if every freed top-reliever appearance were redeployed')
ax.legend([matplotlib.patches.Patch(color=component_colors[c]) for c in component_colors], [component_labels[c] for c in component_colors], frameon=False, ncol=2, fontsize=8, loc='lower center', bbox_to_anchor=(0.4, 1.0))
ax.set_title('Figure 8. The ledger: what D adds and costs, by usage rule', loc='left', fontsize=10, color=ink, pad=34)
fig.tight_layout()
fig.savefig(fig_dir / 'figure8_ledger.png', dpi=200)
plt.close(fig)
ledger.to_csv(results_dir / 'phase2_ledger.csv', index=False)
blowout.to_csv(results_dir / 'phase2_blowout.csv', index=False)


# Step 12 Save what Phase 3 needs: each appearance's role (starter, top reliever, other reliever, position player)
pitching[['gid', 'id', 'year', 'team', 'role', 'role_detail', 'run_value']].to_csv(results_dir / 'phase2_appearances.csv', index=False)
