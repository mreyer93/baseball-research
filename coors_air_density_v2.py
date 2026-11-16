#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:48:44 2026

@author: reyer
"""

# Coors Field air density from Statcast pitch tracking
# Question: does the drag and spin-induced (Magnus) acceleration that Statcast fits to every
# pitch fall at Coors Field by the amount that air density predicts? If it does, the loss of
# movement at Coors is physics that a forecast can apply as a known factor.
# Data: Baseball Savant pitch-level search, Rockies pitchers only, regular season 2025 and 2026,
# home (Coors Field) and road; MLB Stats API schedule with game-time temperature and venue
# elevation. Same pitchers in both samples, so the comparison is within pitcher.
# v2 (15 November 2026): file header, data paths relative to this file, a download step (Step 1),
# longer variable names, notes under each result, and Figures 1 to 3. The numbers are the same as v1.

import json
import os
import time
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
os.makedirs(DATA, exist_ok=True)
G = 32.174  # ft/s^2

# Rockies colors for the figures: purple for Coors, gray for the road, silver for the grid
PURPLE, LAVENDER, GRAY, SILVER = '#33006F', '#A88FD6', '#6B7178', '#C4CED4'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': SILVER, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 200})

# Step 1 Download (or load cached) the Rockies pitchers' pitches, home and road, 2025 and 2026
# Baseball Savant returns at most 25,000 rows per search, so each season and side is its own
# request (about 12,000 rows each). The schedule gives each game's temperature and venue elevation.
def get_file(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as r:
            open(path, 'wb').write(r.read())
        time.sleep(1)
    return path

SAVANT = 'https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={season}%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=COL%7C&home_road={side}&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details'
SCHEDULE = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=115&season={season}&gameType=R&hydrate=weather,venue(location)'
frames = []
for season in (2025, 2026):
    get_file(SCHEDULE.format(season=season), os.path.join(DATA, f'sched_{season}.json'))
    for side in ('Home', 'Road'):
        savant_file = get_file(SAVANT.format(season=season, side=side), os.path.join(DATA, f'col_{season}_{side}.csv'))
        savant = pd.read_csv(savant_file, low_memory=False)
        savant = savant.assign(season=season, coors=(side == 'Home'))
        frames.append(savant)
pitches = pd.concat(frames, ignore_index=True)
pitches = pitches[pitches['game_type'] == 'R']
pitches = pitches.dropna(subset=['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'pitch_type', 'release_speed']).copy()
print('pitches', len(pitches), 'pitchers', pitches['pitcher'].nunique())
print(pitches.groupby(['season', 'coors']).size())

####
# pitches 48175 pitchers 54
# season  coors
# 2025    False    11506
#         True     12496
# 2026    False    11940
#         True     12233
# dtype: int64
####

# Step 2 Split each pitch's fitted acceleration into drag and Magnus parts
# Statcast fits a constant acceleration (ax, ay, az) and the velocity (vx0, vy0, vz0) at y = 50 ft.
# Removing gravity leaves the aerodynamic acceleration. Its component against the velocity is
# drag; the component perpendicular to the velocity is the spin-induced (Magnus) force. Both are
# proportional to air density times speed squared, so dividing by speed squared leaves a
# quantity proportional to air density for a given pitcher and pitch type.
velocity = pitches[['vx0', 'vy0', 'vz0']].to_numpy()
accel = pitches[['ax', 'ay', 'az']].to_numpy() + np.array([0.0, 0.0, G])
speed = np.linalg.norm(velocity, axis=1)
velocity_unit = velocity / speed[:, None]
accel_drag = -(accel * velocity_unit).sum(axis=1)
accel_perp = accel + accel_drag[:, None] * velocity_unit
accel_magnus = np.linalg.norm(accel_perp, axis=1)
pitches['k_drag'] = accel_drag / speed**2
pitches['k_magnus'] = accel_magnus / speed**2

# Step 3 Game-time air density from temperature and venue elevation (MLB Stats API)
# Standard-atmosphere pressure at the venue's elevation, ideal gas at the game-time temperature,
# dry air. This ignores humidity and the day's barometric pressure, each worth about 1 to 2 percent.
rows = []
for season in (2025, 2026):
    sched = json.load(open(os.path.join(DATA, f'sched_{season}.json')))
    for day in sched['dates']:
        for game in day['games']:
            temp = game.get('weather', {}).get('temp')
            elev = game.get('venue', {}).get('location', {}).get('elevation')
            if temp is None or elev is None:
                continue
            rows.append({'game_pk': game['gamePk'], 'temp_f': float(temp), 'elev_ft': float(elev), 'venue': game['venue']['name'], 'condition': game['weather'].get('condition')})
games = pd.DataFrame(rows).drop_duplicates('game_pk')
elev_m = games['elev_ft'] * 0.3048
pressure_pa = 101325.0 * (1 - 2.25577e-5 * elev_m) ** 5.25588
temp_k = (games['temp_f'] - 32) * 5 / 9 + 273.15
games['rho'] = pressure_pa / (287.05 * temp_k)
pitches = pitches.merge(games, on='game_pk', how='left').copy()
print('pitches without weather', pitches['rho'].isna().sum())
print(pitches.groupby('coors')[['elev_ft', 'temp_f', 'rho']].mean().round(3))
rho_ratio_pred = (pitches.loc[pitches['coors'], 'rho'].mean() / pitches.loc[~pitches['coors'], 'rho'].mean())
print('predicted Coors / road air density ratio', round(rho_ratio_pred, 3))

####
# pitches without weather 470
#         elev_ft  temp_f    rho
# coors
# False   376.787  73.642  1.176
# True   5190.000  76.422  0.980
# predicted Coors / road air density ratio 0.833
#
# The air at Coors is about 17 percent thinner than in the road parks these pitchers visited
# (0.980 against 1.176 kg per cubic meter), almost all of it from elevation.
####

# Step 4 Within-pitcher Coors / road ratios of drag and Magnus, by pitch type
# A group is one pitcher, pitch type and season with at least 30 pitches at Coors and 30 on the road.
# Ratio of medians within each group, then a pitch-weighted mean of log ratios across groups.
# Interval: bootstrap over pitchers (resample whole pitchers, 2,000 times).
groups = pitches.groupby(['pitcher', 'pitch_type', 'season', 'coors'])
medians = groups[['k_drag', 'k_magnus', 'pfx_x', 'pfx_z', 'release_speed', 'release_spin_rate', 'rho']].median()
counts = groups.size().rename('n')
medians = medians.join(counts).reset_index()
home = medians[medians['coors']].drop(columns='coors')
road = medians[~medians['coors']].drop(columns='coors')
pair = home.merge(road, on=['pitcher', 'pitch_type', 'season'], suffixes=('_coors', '_road'))
pair = pair[(pair['n_coors'] >= 30) & (pair['n_road'] >= 30)].copy()
pair['w'] = np.minimum(pair['n_coors'], pair['n_road'])
for q in ('k_drag', 'k_magnus', 'rho'):
    pair[f'log_{q}'] = np.log(pair[f'{q}_coors'] / pair[f'{q}_road'])
print('pitcher-pitch-season groups', len(pair), 'pitchers', pair['pitcher'].nunique())

def weighted_ratio(df, col):
    return float(np.exp(np.average(df[col], weights=df['w'])))

rng = np.random.default_rng(20261113)
pitcher_ids = pair['pitcher'].unique()
by_pitcher = {p: pair[pair['pitcher'] == p] for p in pitcher_ids}
boot = {'log_k_drag': [], 'log_k_magnus': []}
for b in range(2000):
    take = rng.choice(pitcher_ids, size=len(pitcher_ids), replace=True)
    boot_sample = pd.concat([by_pitcher[p] for p in take])
    for col in boot:
        boot[col].append(weighted_ratio(boot_sample, col))
for col, label in (('log_k_drag', 'drag'), ('log_k_magnus', 'Magnus')):
    lo, hi = np.percentile(boot[col], [5, 95])
    print(f'{label:6s} Coors/road ratio {weighted_ratio(pair, col):.3f}  90% interval {lo:.3f} to {hi:.3f}')
print(f'air density ratio predicted for the same games {weighted_ratio(pair, "log_rho"):.3f}')
for season in (2025, 2026):
    sub = pair[pair['season'] == season]
    print(season, 'drag', round(weighted_ratio(sub, 'log_k_drag'), 3), 'Magnus', round(weighted_ratio(sub, 'log_k_magnus'), 3), 'air density', round(weighted_ratio(sub, 'log_rho'), 3), 'groups', len(sub))

####
# pitcher-pitch-season groups 165 pitchers 35
# drag   Coors/road ratio 0.846  90% interval 0.840 to 0.851
# Magnus Coors/road ratio 0.807  90% interval 0.799 to 0.817
# air density ratio predicted for the same games 0.830
# 2025 drag 0.853 Magnus 0.813 air density 0.834 groups 80
# 2026 drag 0.838 Magnus 0.801 air density 0.825 groups 85
#
# Air density predicts 0.830. Drag came out a little above it (0.846) and Magnus a little below
# (0.807), and both seasons agree. So the lost movement is mostly the air. Why Magnus falls about 4
# points more than drag, I can't tell from this data (camera calibration at Coors, or a real change
# in lift at lower air density).
####

# Figure 1 Coors / road ratios of drag and Magnus against the air density ratio
fig, ax = plt.subplots(figsize=(12, 4.6))
rho_pair = weighted_ratio(pair, 'log_rho')
for y, (col, label) in enumerate((('log_k_magnus', 'Magnus (spin) force'), ('log_k_drag', 'Drag'))):
    lo, hi = np.percentile(boot[col], [5, 95])
    ax.plot([lo, hi], [y, y], color=PURPLE, lw=4, solid_capstyle='round')
    ax.plot(weighted_ratio(pair, col), y, 'o', color=PURPLE, ms=11)
    if weighted_ratio(pair, col) < rho_pair:
        ax.text(lo - 0.004, y, f'{weighted_ratio(pair, col):.3f} (90% {lo:.3f} to {hi:.3f})', ha='right', va='center', fontsize=12)
    else:
        ax.text(hi + 0.004, y, f'{weighted_ratio(pair, col):.3f} (90% {lo:.3f} to {hi:.3f})', ha='left', va='center', fontsize=12)
ax.axvline(rho_pair, color=GRAY, ls='--', lw=1.8)
ax.text(rho_pair - 0.002, 1.42, f'air density predicts {rho_pair:.3f}', ha='right', va='center', color='#333333', fontsize=12)
ax.axvline(1.0, color='black', lw=0.9)
ax.text(0.998, 1.42, 'no change', ha='right', va='center', color='#333333', fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Magnus (spin) force', 'Drag'])
ax.set_ylim(-0.6, 1.7)
ax.set_xlim(0.68, 1.01)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Coors over road, same pitcher and pitch type (1.0 = no change)')
ax.set_title(f'Drag and spin force at Coors relative to the road, Rockies pitchers 2025 and 2026 ({len(pair)} groups)', loc='left', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_air_density_v2_fig1.png'))
plt.close(fig)

# Step 5 Game-level test: does drag track air density across every park, not only Coors?
# Each pitch's k_drag is divided by its own pitcher-pitch type-season median (all games), which
# removes pitcher differences. The game median of that relative drag is regressed on the game's
# relative air density (rho over the mean rho of the pitcher's games is close to rho over the
# overall mean, so the overall mean is used). Physics predicts a slope of 1 through (1, 1).
pitches['k_drag_rel'] = pitches['k_drag'] / pitches.groupby(['pitcher', 'pitch_type', 'season'])['k_drag'].transform('median')
pitches['k_magnus_rel'] = pitches['k_magnus'] / pitches.groupby(['pitcher', 'pitch_type', 'season'])['k_magnus'].transform('median')
game_medians = pitches.groupby('game_pk').agg(k_drag_rel=('k_drag_rel', 'median'), k_magnus_rel=('k_magnus_rel', 'median'), rho=('rho', 'first'), coors=('coors', 'first'), n=('k_drag_rel', 'size'), venue=('venue', 'first'), temp_f=('temp_f', 'first'))
game_medians = game_medians.dropna()
game_medians['rho_rel'] = game_medians['rho'] / game_medians['rho'].mean()
for ycol in ('k_drag_rel', 'k_magnus_rel'):
    X_all = np.column_stack([np.ones(len(game_medians)), game_medians['rho_rel']])
    beta_all, *_ = np.linalg.lstsq(X_all, game_medians[ycol] / game_medians[ycol].mean(), rcond=None)
    road_games = game_medians[~game_medians['coors']]
    X_road = np.column_stack([np.ones(len(road_games)), road_games['rho_rel']])
    beta_road, *_ = np.linalg.lstsq(X_road, road_games[ycol] / road_games[ycol].mean(), rcond=None)
    r_road = np.corrcoef(road_games['rho_rel'], road_games[ycol])[0, 1]
    print(f'{ycol}: all games slope {beta_all[1]:.2f}; road games only slope {beta_road[1]:.2f}, correlation {r_road:.2f}, games {len(road_games)}')

####
# k_drag_rel: all games slope 0.85; road games only slope 0.30, correlation 0.19, games 159
# k_magnus_rel: all games slope 1.08; road games only slope 0.58, correlation 0.42, games 159
#
# Across all games the slopes are near the physics value of 1 (0.85 and 1.08), but that is mostly
# Coors against everywhere else (Figure 2). Among the road parks alone the relationship is much
# weaker (slopes 0.30 and 0.58, correlations 0.19 and 0.42): there the spread in air density is
# small next to the game-to-game noise, so game-level temperature does not resolve the effect well.
####

# Figure 2 Game by game: relative drag and spin force against relative air density
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
x_line = np.array([game_medians['rho_rel'].min() - 0.02, game_medians['rho_rel'].max() + 0.02])
for ax, ycol, title in zip(axes, ('k_drag_rel', 'k_magnus_rel'), ('Drag', 'Magnus (spin) force')):
    y_rel = game_medians[ycol] / game_medians[ycol].mean()
    road_mask = ~game_medians['coors']
    ax.scatter(game_medians.loc[road_mask, 'rho_rel'], y_rel[road_mask], s=26, color=GRAY, alpha=0.7, label=f'road games ({road_mask.sum()})')
    ax.scatter(game_medians.loc[~road_mask, 'rho_rel'], y_rel[~road_mask], s=26, color=PURPLE, alpha=0.8, label=f'Coors games ({(~road_mask).sum()})')
    ax.plot(x_line, x_line, color='black', ls='--', lw=1.4, label='physics: proportional to air density')
    ax.set_title(title, loc='left')
    ax.set_xlabel('game air density / average air density')
axes[0].set_ylabel('game median, relative to the\npitcher and pitch type median')
axes[0].legend(fontsize=11, frameon=False, loc='upper left')
fig.suptitle('Drag and spin force against air density, game by game, 2025 and 2026', x=0.01, ha='left', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_air_density_v2_fig2.png'))
plt.close(fig)

# Step 6 What a hitter sees: movement in inches and velocity, same pitchers, Coors and road
# pfx_x and pfx_z are Statcast's spin-induced movement (gravity removed), in feet.
pair['dz_in'] = (pair['pfx_z_coors'] - pair['pfx_z_road']) * 12
pair['dx_in'] = (pair['pfx_x_coors'].abs() - pair['pfx_x_road'].abs()) * 12
pair['dv'] = pair['release_speed_coors'] - pair['release_speed_road']
pair['dspin'] = pair['release_spin_rate_coors'] - pair['release_spin_rate_road']
pair['brk_road'] = np.hypot(pair['pfx_x_road'], pair['pfx_z_road']) * 12
pair['brk_coors'] = np.hypot(pair['pfx_x_coors'], pair['pfx_z_coors']) * 12
summary = []
for pt, sub in pair.groupby('pitch_type'):
    if sub['w'].sum() < 1500:
        continue
    wavg = lambda c: np.average(sub[c], weights=sub['w'])
    summary.append({'pitch_type': pt, 'groups': len(sub), 'pitches': int(sub['n_coors'].sum() + sub['n_road'].sum()),
                    'vert_road_in': round(wavg('pfx_z_road') * 12, 1), 'vert_coors_in': round(wavg('pfx_z_coors') * 12, 1),
                    'horiz_road_in': round(np.average(sub['pfx_x_road'].abs(), weights=sub['w']) * 12, 1),
                    'horiz_coors_in': round(np.average(sub['pfx_x_coors'].abs(), weights=sub['w']) * 12, 1),
                    'total_break_ratio': round(np.average(sub['brk_coors'] / sub['brk_road'], weights=sub['w']), 3),
                    'velo_diff_mph': round(wavg('dv'), 2), 'spin_diff_rpm': round(wavg('dspin'), 0)})
summary = pd.DataFrame(summary).sort_values('pitches', ascending=False)
print(summary.to_string(index=False))

####
# pitch_type  groups  pitches  vert_road_in  vert_coors_in  horiz_road_in  horiz_coors_in  total_break_ratio  velo_diff_mph  spin_diff_rpm
#         FF      42    16677          14.7           11.9            8.1             6.3              0.796          -0.18           25.0
#         SL      27     5637           2.3            1.8            3.1             2.9              0.845          -0.05           16.0
#         SI      22     5051           6.5            5.5           15.5            12.1              0.796          -0.07           11.0
#         CH      21     4131           6.1            5.0           13.4            10.1              0.774          -0.19           26.0
#         ST      15     3822           0.9           -0.6           13.7            11.4              0.821           0.02           -2.0
#
# Velocity and spin are nearly unchanged (within 0.2 mph and 26 rpm), but total break falls by
# 15 to 23 percent (ratios 0.774 to 0.845). A four-seamer's ride drops from 14.7 to 11.9 inches.
####

# Figure 3 Movement by pitch type, same pitchers, road and Coors
PITCH_NAMES = {'FF': 'Four-seam', 'SL': 'Slider', 'SI': 'Sinker', 'CH': 'Changeup', 'ST': 'Sweeper'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
order = summary.iloc[::-1].reset_index(drop=True)
for ax, (road_col, coors_col, title) in zip(axes, (('vert_road_in', 'vert_coors_in', 'Vertical movement (in)'), ('horiz_road_in', 'horiz_coors_in', 'Horizontal movement (in)'))):
    for y, row in order.iterrows():
        ax.plot([row[road_col], row[coors_col]], [y, y], color=SILVER, lw=5, solid_capstyle='round', zorder=1)
        ax.plot(row[road_col], y, 'o', color=GRAY, ms=11, zorder=2)
        ax.plot(row[coors_col], y, 'o', color=PURPLE, ms=11, zorder=3)
        right = max(row[road_col], row[coors_col])
        ax.text(right + 0.5, y, f'{row[road_col]:.1f} to {row[coors_col]:.1f}', va='center', fontsize=11)
    ax.set_title(title, loc='left')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(min(order[road_col].min(), order[coors_col].min()) - 1.5, max(order[road_col].max(), order[coors_col].max()) + 4.5)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([f"{PITCH_NAMES.get(p, p)} ({p})" for p in order['pitch_type']])
axes[0].plot([], [], 'o', color=GRAY, ms=10, label='road')
axes[0].plot([], [], 'o', color=PURPLE, ms=10, label='Coors')
axes[0].legend(frameon=False, loc='lower right', fontsize=11)
fig.suptitle('Movement by pitch type, same pitchers at Coors and on the road (spin-induced, gravity removed)', x=0.01, ha='left', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'coors_air_density_v2_fig3.png'))
plt.close(fig)

# Step 7 Did the lost movement cost the same pitchers swings and misses at home?
# Swinging strikes per swing, same pitcher-pitch type-season groups as Step 4, Coors against road.
# Bunts are left out. Interval: bootstrap over pitchers.
WHIFF = {'swinging_strike', 'swinging_strike_blocked'}
SWING = WHIFF | {'foul', 'foul_tip', 'hit_into_play'}
keys = pair[['pitcher', 'pitch_type', 'season']]
swings = pitches.merge(keys, on=['pitcher', 'pitch_type', 'season'])
swings = swings[swings['description'].isin(SWING)].copy()
swings['whiff'] = swings['description'].isin(WHIFF)
def whiff_gap(df):
    t = df.groupby('coors')['whiff'].mean()
    return t.get(True, np.nan), t.get(False, np.nan)
swings_by_pitcher = {p: swings[swings['pitcher'] == p] for p in swings['pitcher'].unique()}
for label, sub in (('all pitches', swings), ('four-seam', swings[swings['pitch_type'] == 'FF'])):
    whiff_coors, whiff_road = whiff_gap(sub)
    ids = sub['pitcher'].unique()
    diffs = []
    for b in range(2000):
        take = rng.choice(ids, size=len(ids), replace=True)
        boot_sample = pd.concat([swings_by_pitcher[p][swings_by_pitcher[p]['pitch_type'] == 'FF'] if label == 'four-seam' else swings_by_pitcher[p] for p in take])
        boot_coors, boot_road = whiff_gap(boot_sample)
        diffs.append(boot_coors - boot_road)
    lo, hi = np.percentile(diffs, [5, 95])
    print(f'{label}: whiffs per swing Coors {whiff_coors:.3f}, road {whiff_road:.3f}, difference {whiff_coors - whiff_road:+.3f} (90% interval {lo:+.3f} to {hi:+.3f}), swings {len(sub)}')

####
# all pitches: whiffs per swing Coors 0.209, road 0.200, difference +0.009 (90% interval -0.000 to +0.019), swings 21324
# four-seam: whiffs per swing Coors 0.138, road 0.150, difference -0.011 (90% interval -0.027 to +0.004), swings 8159
#
# This surprised me: the same pitchers got as many swings and misses per swing at Coors as on
# the road, even though their pitches moved 15 to 23 percent less. So I would not scale a pitch
# model's inputs by air density and assume the outcomes follow without testing it.
####
