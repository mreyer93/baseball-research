# The deterministic pitcher

Research plan. Started 2026-10-14 from an interview question; personal research, possibly
published later. Not an application deliverable.

## The question

A pitcher gives up one home run and then strikes out the next three batters, every inning,
without exception. He cannot be injured and has no workload limit. Is he worth a Major
League roster spot, in what roles, and what would we tell a manager or front office?

## Where the interview answer stands

- The roster comparison is against the 13th pitcher on a 26-man roster (the mop-up or long
  reliever, usually near replacement level), not against a better pitcher. To verify: the
  13-pitcher cap and how it applies.
- With a lead L larger than the number of full innings remaining n, and entering at the
  start of an inning, he wins with certainty. No real pitcher does that, so in those states
  he is the best option on the staff, not merely an acceptable one.
- Blowout losses are a second use the interview answer missed. A team that scores one run a
  game still has plenty of them.
- The offense threshold is real, but it lives in the states with L at most n: there he needs
  our offense to score at least n - L + 1 runs. A tie sends the game to extra innings, where
  the automatic runner on second turns him into a two-run-per-inning pitcher. He never
  pitches a tie or an extra inning.
- He has zero variance. Low variance is worth more when ahead and less when behind in a
  close game, so he isolates the value of variance in a reliever.
- Context-neutral metrics (Wins Above Replacement (WAR), Fielding Independent Pitching
  (FIP), earned run average (ERA)) price him as terrible; Win Probability Added (WPA) would
  put him near zero.

Illustrative magnitude (guessed inputs, to be replaced by measured ones): in blowout losses
he allows about 0.45 runs per inning more than a mop-up reliever, at roughly 0.01 win
probability per run, so about 0.005 wins per inning, or 0.5 wins over 100 innings. The
benefit (rest for the rest of the staff) is unmeasured. The answer is a close balance, which
is why the project is worth doing.

## Definitions to settle

1. The pattern: reset every inning, cycle continuously through batters (so a mid-inning entry
   shifts where the home run falls), or once per appearance. A full clean inning costs exactly
   one run under all three; they differ on mid-inning entries and inherited runners.
2. Rules: three-batter minimum (he always satisfies it), inherited runners score on the home
   run, automatic runner in extra innings, home team in the bottom of the 9th.
3. Generalisation D(r): a deterministic pitcher allowing r runs per inning. r = 0.5 is roughly
   a league-average ERA with zero variance.

Working convention for Phase 1: he enters only at the start of an inning and pitches every
remaining regulation inning, conceding exactly one run per inning.

## Research questions

1. In which game states does he beat any real pitcher on win probability, and how does that
   depend on our team's distribution of runs per inning?
2. How many innings per season is he eligible for, as a function of runs scored and allowed?
3. What are the innings he absorbs worth (fresher top relievers, lower starter pitch counts,
   less shuttling of relievers between Triple-A and the Majors)?
4. Net wins per season against the actual 13th pitcher.
5. Break-even: which team qualities and fatigue assumptions make him a net negative?
6. Revealed preference: what does position-player pitching (which managers already accept in
   lopsided games to rest the bullpen) imply about the value of bullpen rest?
7. How each metric prices him, and what his market value would be.

## Phases

1. Analytic model. His win probability for every (L, n) from empirical per-inning run
   distributions at several offense levels, against a mop-up reliever. No simulation.
   Script: `scripts/phase1_analytic.py`.
2. Game-state frequencies from Retrosheet: eligible innings per team-season against run
   differential.
3. Game simulator (base-out Markov chain or inning level), validated against real score
   distributions and empirical win-expectancy tables.
4. Season simulator: 13-pitcher staff, talent levels, rest and consecutive-day limits, a fatigue
   penalty, a usage policy; 162-game seasons with and without him.
5. Sensitivity and break-even over runs scored, runs allowed, fatigue strength, and r.
6. Position-player pitching: game states, run cost, bullpen rest gained.
7. Write-up as .docx, possibly a public post later.

Every phase gets a printed validation or backtest. Phases 1 and 2 answer most of the
interview question on their own.

## Phase 1 result (2026-10-14)

`scripts/phase1_analytic.py`, log in `results/phase1_run_log.txt`, figures 1 to 5.

- The model (per-inning run distributions added as independent) is calibrated within 2.2
  points against real outcomes and matches an in-sample lookup table (Brier 0.1615 against
  0.1620). It slightly understates how well real teams hold leads.
- The usage rule does not depend on the offense: use D if and only if the lead is greater
  than the half-innings left. Above that line he cannot lose; on or below it he is worse
  than a mop-up reliever in every state, for every offense tier. Offense narrows that gap by
  up to 8 points but never closes it.
- From 2022 to 2025 a team reached that state in about 63 games a season, after which D
  could have pitched about 165 innings. Real teams lost 4.0 percent of those games, 2.5 per
  team-season, all of which D would have won.
- The interview conclusion reverses: D is the best possible closer for any 2-run lead in the
  9th (3 in the 8th, and so on), before any value from rest is counted.
- Independence understates within-game run variance by 4 to 10 percent; noted for Phase 3.

## Phase 2 result (2026-10-14)

`scripts/phase2_ledger.py`, log in `results/phase2_run_log.txt`, figures 6 to 8. Data:
Retrosheet season CSV files (play-by-play and per-pitcher game lines), 2022 to 2025.

- Checks: the Phase 1 model implies 10.1 runs per win; the play-by-play reproduces the 2.5
  lock wins a season from the game logs.
- Blowout use loses wins at every threshold: 0.06 wins per 100 innings trailing by 8 or more
  from the 6th, 0.42 at 5 or more, 0.77 at 4 or more. The deficit matters, the starting
  inning hardly does.
- Whose innings: of about 166 lock innings a season, 31 are starters', 51 the top three
  relievers' (51 appearances), 84 other relievers'.
- The roster spot: the lock use takes 1.8 to 1.9 times an average bullpen slot's innings
  and appearances off the bullpen, more than a slot in 117 of 120 team-seasons.
- Back-to-back penalty for a reliever, within pitcher-season: 0.33 runs per 9 innings (0.15
  to 0.50).
- Ledger, lock only: 2.56 wins a season at the floor (2.50 locked wins, 0.10 rest through
  back-to-backs avoided, minus 0.04 for the 13th pitcher's appearances), up to 3.99 if every
  freed top-reliever appearance were redeployed to high-leverage spots (a ceiling). Adding
  blowout use never pays at the floor and only breaks even at the ceiling for 7-run and 8-run
  deficits.

Open for Phase 4: where between 2.5 and 4 wins the value of rest really sits (redeployment,
availability day to day, injury), and whether blowout use pays on days the bullpen is short.

## Phase 3 result (2026-10-14)

`scripts/game_engine.py` (the engine, shared with Phase 4) and
`scripts/phase3_engine_validation.py` (builds its inputs, validates it), log in
`results/phase3_run_log.txt`, figures 9 to 11. Inputs saved to
`results/phase3_engine_inputs.pkl`.

- Plate-appearance engine: nine outcomes per plate appearance from batter rates (offense
  tier, home or away, lineup slot) combined with pitcher rates by the odds-ratio rule; runner
  movement drawn from what really happened in each base-out state; runner events between
  plate appearances; automatic runner in extra innings; walk-offs. D is exact. 40,000 games
  take about 6 seconds.
- A per-game factor for each team's batters (log-normal spread 0.15, calibrated on the
  spread of team runs per game) stands in for day-to-day variation.
- Matches real 2022 to 2025 within about 1 to 2 percent on scoring, scoreless and big
  innings, offense tiers, pitch counts and home advantage; run expectancy within 0.023 runs
  on average; win expectancy within 0.9 points on average (Brier 0.1622 against 0.1620
  in-sample); pitcher groups within 0.004 runs per batter of runs charged; D minus mop-up
  within 1.5 points of Phase 1.
- Known misses: 10.0 percent extra-inning games against 8.7 (about two a team-season;
  score-dependent bullpen use makes it worse, not better, so it is not that); scoring by
  inning is too flat (needs starters by time through the order and late relievers); the
  top-against-other reliever gap is 6 percent too wide.

For Phase 4: model starters by time through the order, recalibrate the game factor once
starter quality varies, shrink or caveat the reliever talent gap, recheck extra innings.

## Phase 4 result (2026-10-14)

`scripts/phase4_season.py`, log in `results/phase4_run_log.txt`, figures 12 and 13.

- Season simulator: 162 games with off days, five starters (rates by time through the order),
  three top relievers, five others (the fifth is the 13th pitcher; D replaces him), a manager
  calibrated on three real measures (pull count 90 on outs per start, top-reliever window
  from the 8th on top-reliever appearances, avoiding yesterday's reliever 75 percent of the
  time on the back-to-back share), and fatigue (unavailable after two straight days or 30
  pitches yesterday; back-to-back penalty). Every half-inning restarts its random numbers
  from the same seed in every version of a season, so comparisons are paired.
- Only 42 percent of the top relievers' advantage persists into the next season; the rest
  was luck. The simulation uses the persistent part.
- Value of rest (gain from D with fatigue minus gain with every pitcher always fresh): 0.13
  wins a season (standard error 0.02), 0.06 to 0.21 across every sensitivity case. Top
  relievers already start 93 percent of close-late half-innings without D, so freed
  appearances have almost nowhere to go.
- Total with the lock rule: 2.63 wins a season (2.56 to 2.71), 2.50 of it locked wins from
  real games. Phase 2's floor was right; its ceiling (3.99) was not.
- Blowout use on top of the lock rule costs 0.21 wins a season even with rest valued in full.
- Offense changes the total by about 0.3 wins (more lock states for better offenses), not the
  value of rest.
- Not modelled: injuries, call-ups and options, mid-inning matchup changes.

Remaining phases: 5 (largely answered by Phases 2 and 4), 6 (position-player pitching as
revealed preference), 7 (write-up).

## Weakest link

Phase 4. The value of reliever rest drives the answer and is the least known input: fatigue
studies are confounded by selection (tired relievers are used in different situations) and
causal evidence on rest and injury is thin. The honest headline may be a range.

## To verify before relying on it

- The 13-pitcher cap on the 26-man roster.
- Current position-player pitching thresholds (changed in 2023).
- Automatic runner in regular-season extra innings.
- The collective bargaining agreement expires 2026-12-01; roster rules may change.

## Data

Retrosheet game logs 2022 to 2025 (`data/retrosheet/`), chosen because 2022 brought the
universal designated hitter and ended seven-inning doubleheaders; earlier seasons have
pitchers batting in National League parks, which changes per-inning run distributions.

Required notice for any published use: "The information used here was obtained free of
charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
