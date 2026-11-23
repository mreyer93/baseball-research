#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:20:00 2026

@author: reyer
"""

# The game engine for the deterministic pitcher project, shared by Phase 3 (which builds its inputs and validates it) and Phase 4 (the season simulation).
# Kept as a module rather than pasted into each script so that the engine Phase 4 uses is exactly the one Phase 3 validated.
#
# How a game is played:
#   A half-inning is a sequence of plate appearances. Before each one, a runner event can happen first (a steal, a wild pitch, a pickoff), with the chance of one depending on the base-out state.
#   Each plate appearance ends in one of nine outcomes. Its chances come from the batter (his team's offense tier, home or away, and his lineup slot) and the pitcher, combined by the odds-ratio rule: batter rate x pitcher rate / league rate, rescaled to add to one. A league-average pitcher leaves the batter's rates unchanged.
#   What the outcome does to the runners (who advances, who scores, who is out) is drawn from what actually happened after that outcome in that base-out state in 2022 to 2025.
#   Each team's batters get a game factor for the whole game: the chances of every way of reaching base are multiplied by it before rescaling. It stands for what makes one day different from another (the opposing starter's form, the park, the weather) and is drawn from nine levels of a log-normal spread whose width is calibrated in Phase 3.
#   Extra innings start with a runner on second (the automatic runner), and the bottom of the 9th or later stops the moment the home team leads.
# D (the deterministic pitcher) skips all of this: the first batter he faces in a half-inning hits a home run, the next three strike out, and nothing else happens. He only ever starts a half-inning.

import random
import bisect
import pickle
import statistics
import numpy as np

OUTCOMES = ['K', 'BB', 'HBP', '1B', '2B', '3B', 'HR', 'ROE', 'OUT']
HR = OUTCOMES.index('HR')
K = OUTCOMES.index('K')
END = 24  # the third out; states 0 to 23 are outs x 8 + (runner on 1st) + 2 x (runner on 2nd) + 4 x (runner on 3rd)
AUTOMATIC_RUNNER_STATE = 2  # nobody out, runner on second
ON_BASE = [OUTCOMES.index(o) for o in ['BB', 'HBP', '1B', '2B', '3B', 'HR', 'ROE']]


def game_factor_levels(sd):
    # Nine equally likely levels of a log-normal spread with this standard deviation (on the log scale), centred so the levels average to one
    z = [statistics.NormalDist().inv_cdf((k + 0.5) / 9) for k in range(9)]
    levels = [float(np.exp(sd * zk)) for zk in z]
    return [round(float(level / np.mean(levels)), 6) for level in levels]


def load_inputs(path):
    with open(path, 'rb') as f:
        inputs = pickle.load(f)
    inputs['cum_cache'] = {}
    return inputs


def draw_index(cum):
    return min(bisect.bisect_right(cum, random.random()), len(cum) - 1)


def matchup_cum(inputs, tier, home, slot, pitcher, factor=1.0):
    # Cumulative chances of the nine outcomes for this batter against this pitcher (odds-ratio rule) on a day with this game factor, cached because the same matchups repeat millions of times
    key = (tier, home, slot, pitcher, factor)
    if key not in inputs['cum_cache']:
        p = inputs['batting_rates'][(tier, home, slot)] * inputs['pitcher_rates'][pitcher] / inputs['league_rates']
        p[ON_BASE] = p[ON_BASE] * factor
        inputs['cum_cache'][key] = list(np.cumsum(p / p.sum()))
    return inputs['cum_cache'][key]


def transition(inputs, kind, state):
    # kind is an outcome index, or 'runner' for a runner event; returns (new state, runs scored)
    cum, results = inputs['transitions'][(kind, state)]
    return results[draw_index(cum)]


def d_transition(outcome, state):
    # D's two outcomes, exactly: a home run scores every runner plus the batter and clears the bases; a strikeout adds an out and moves no one
    outs, bases = divmod(state, 8)
    if outcome == HR:
        return outs * 8, 1 + bin(bases).count('1')
    return (END, 0) if outs == 2 else ((outs + 1) * 8 + bases, 0)


def half_inning(inputs, tier, home, lineup_slot, pitcher, start_state=0, walkoff_margin=None, run_expectancy=None, factor=1.0):
    # Plays one half-inning. lineup_slot is the index (0 to 8) of the first batter due up. walkoff_margin: for the home team in the 9th or later, the game ends as soon as runs exceed it.
    # run_expectancy, if given, is a pair of arrays (total runs still to come, visits) by base-out state, filled in for the Phase 3 check
    state = start_state
    runs = 0
    batters = 0
    pitches = 0
    visited = []
    while state != END:
        if pitcher != 'D' and random.random() < inputs['runner_event_chance'][state]:
            visited.append((state, runs))
            state, scored = transition(inputs, 'runner', state)
            runs += scored
        else:
            visited.append((state, runs))
            if pitcher == 'D':
                outcome = HR if batters == 0 else K
                state, scored = d_transition(outcome, state)
                pitches += 1 if outcome == HR else 3
            else:
                outcome = draw_index(matchup_cum(inputs, tier, home, lineup_slot, pitcher, factor))
                state, scored = transition(inputs, outcome, state)
                pitches += random.choice(inputs['pitches_by_outcome'][outcome])
            runs += scored
            batters += 1
            lineup_slot = (lineup_slot + 1) % 9
        if walkoff_margin is not None and runs > walkoff_margin:
            break
    if run_expectancy is not None:
        for s, before in visited:
            run_expectancy[0][s] += runs - before
            run_expectancy[1][s] += 1
    return runs, lineup_slot, batters, pitches


def play_game(inputs, visitor_tier, home_tier, plan, start=None, run_expectancy=None, game_factor_sd=None):
    # plan(fielding_side, inning, fielding_lead) returns the pitcher for that half-inning ('league', a pitcher group, or 'D')
    # start, if given, resumes a game at the top of an inning: dict with inning, score_v, score_h, slot_v, slot_h (lineup indexes due up)
    s = start or {'inning': 1, 'score_v': 0, 'score_h': 0, 'slot_v': 0, 'slot_h': 0}
    inning, score_v, score_h, slot_v, slot_h = s['inning'], s['score_v'], s['score_h'], s['slot_v'], s['slot_h']
    halves = []
    sd = inputs.get('game_factor_sd', 0.0) if game_factor_sd is None else game_factor_sd
    levels = game_factor_levels(sd)
    factor_v = random.choice(levels)
    factor_h = random.choice(levels)
    while True:
        start_state = AUTOMATIC_RUNNER_STATE if inning >= 10 else 0
        re_pass = run_expectancy if inning <= 8 else None
        # Top: the visitor bats, the home team fields
        pitcher = plan('home', inning, score_h - score_v)
        runs, slot_v, batters, pitches = half_inning(inputs, visitor_tier, 0, slot_v, pitcher, start_state, None, re_pass, factor_v)
        halves.append((inning, 0, score_h - score_v, runs, pitcher, batters, pitches))
        score_v += runs
        if inning >= 9 and score_h > score_v:
            break
        # Bottom: the home team bats, the visitor fields
        pitcher = plan('visitor', inning, score_v - score_h)
        margin = score_v - score_h if inning >= 9 else None
        runs, slot_h, batters, pitches = half_inning(inputs, home_tier, 1, slot_h, pitcher, start_state, margin, re_pass, factor_h)
        halves.append((inning, 1, score_v - score_h, runs, pitcher, batters, pitches))
        score_h += runs
        if inning >= 9 and score_v != score_h:
            break
        inning += 1
    return {'score_v': score_v, 'score_h': score_h, 'innings': inning, 'halves': halves, 'factor_v': factor_v, 'factor_h': factor_h}


# Added for Phase 4 (2026-10-14): managed half-innings and games, for a team whose pitchers are tracked one by one. The functions above are unchanged, so Phase 3's validation still describes them.
# A pitcher here is a dict with at least: 'name', 'pitches' and 'batters' (so far in this appearance), 'runs' and 'outs' (while he was on the mound), and 'cap' (he is taken out mid-inning once his pitches reach it). D is the pitcher named 'D'.
def half_inning_managed(inputs, tier, home, lineup_slot, current, rate_key, relieve, start_state=0, walkoff_margin=None, factor=1.0):
    # rate_key(pitcher) gives the pitcher_rates key for his next batter (it can depend on his fatigue or how many times he has seen the order); relieve(state) gives the next pitcher when the current one reaches his cap
    state = start_state
    runs = 0
    d_batters = 0
    used = [current]
    while state != END:
        if current['name'] != 'D' and current['pitches'] >= current['cap']:
            current = relieve(state)
            used.append(current)
        outs_before = state // 8
        if current['name'] != 'D' and random.random() < inputs['runner_event_chance'][state]:
            state, scored = transition(inputs, 'runner', state)
        else:
            if current['name'] == 'D':
                outcome = HR if d_batters == 0 else K
                d_batters += 1
                state, scored = d_transition(outcome, state)
                current['pitches'] += 1 if outcome == HR else 3
            else:
                outcome = draw_index(matchup_cum(inputs, tier, home, lineup_slot, rate_key(current), factor))
                state, scored = transition(inputs, outcome, state)
                current['pitches'] += random.choice(inputs['pitches_by_outcome'][outcome])
            current['batters'] += 1
            lineup_slot = (lineup_slot + 1) % 9
        current['outs'] += (3 if state == END else state // 8) - outs_before
        current['runs'] += scored
        runs += scored
        if walkoff_margin is not None and runs > walkoff_margin:
            break
    return runs, lineup_slot, current, used


def play_game_managed(inputs, our_tier, opponent_tier, we_are_home, manager, factor_us, factor_them, seed=None):
    # Our fielding half-innings are managed: manager['start_half'](inning, our_lead, current) picks who starts the half-inning (current is whoever finished the last one, None at the start), manager['relieve'](inning, our_lead, state) replaces a pitcher mid-inning, manager['rate_key'](pitcher) as above. The opponent's pitching is league average throughout.
    # The game factors are passed in rather than drawn here, so that two versions of a season can face exactly the same days
    # seed, if given, restarts the random numbers at the start of every half-inning from (seed, inning, half), so that two versions of a game start every half-inning from the same random numbers and differ only where the change being tested makes them differ
    inning, score_us, score_them, slot_us, slot_them = 1, 0, 0, 0, 0
    current = None
    appearances = []
    halves = []
    while True:
        start_state = AUTOMATIC_RUNNER_STATE if inning >= 10 else 0
        # our_half True means we field; the home team fields the top of the inning
        for our_half in ([True, False] if we_are_home else [False, True]):
            if seed is not None:
                random.seed(seed * 100 + inning * 2 + (0 if our_half == we_are_home else 1))
            batting_home = 1 if (our_half != we_are_home) else 0
            if our_half:
                # we field: the opponent bats
                lead = score_us - score_them
                current = manager['start_half'](inning, lead, current)
                if not appearances or appearances[-1] is not current:
                    appearances.append(current)
                margin = (score_us - score_them) if (we_are_home is False and inning >= 9) else None
                runs, slot_them, current, used = half_inning_managed(inputs, opponent_tier, batting_home, slot_them, current, manager['rate_key'], lambda state, i=inning, l=lead: manager['relieve'](i, l, state), start_state, margin, factor_them)
                for p in used[1:]:
                    appearances.append(p)
                halves.append((inning, 'field', lead, runs, used[0]['name']))
                score_them += runs
            else:
                lead = score_them - score_us
                margin = (score_them - score_us) if (we_are_home and inning >= 9) else None
                runs, slot_us, _, _ = half_inning(inputs, our_tier, batting_home, slot_us, 'league', start_state, margin, None, factor_us)
                halves.append((inning, 'bat', -lead, runs, None))
                score_us += runs
            # the game ends when the home team leads after the top of the 9th or later, or anyone leads after a full 9th or later
            home_score, visitor_score = (score_us, score_them) if we_are_home else (score_them, score_us)
            top_done = (our_half if we_are_home else not our_half)
            if inning >= 9 and top_done and home_score > visitor_score:
                return {'score_us': score_us, 'score_them': score_them, 'innings': inning, 'halves': halves, 'appearances': appearances}
        if inning >= 9 and score_us != score_them:
            return {'score_us': score_us, 'score_them': score_them, 'innings': inning, 'halves': halves, 'appearances': appearances}
        inning += 1
