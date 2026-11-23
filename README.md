# Baseball research

Research projects in baseball analytics by Matt Reyer (PhD in biophysics, University of Chicago). Every project uses public data only (Baseball Savant, the MLB Stats API, FanGraphs and Retrosheet). Each folder has its own README with the results, figures and limits, and its scripts run top to bottom in numbered steps, with their printed results pasted back in as `####` comment blocks so they can be read without running anything. Data are downloaded by the scripts (or from the sources named in each README) and are not committed.

| Project | Question | Methods | Written |
|---|---|---|---|
| [Infield defense from public tracking data](infield-defense/) | How far can public Statcast data go in crediting infielders, separating them from their team's positioning, and projecting young shortstops? | Gradient-boosted expected-out model with a season-out backtest, attribution checked against Outs Above Average, permutation and bootstrap tests, logistic regression with an offset, Florida State League Statcast linked to later positions | November 2026 |
| [The deterministic pitcher](deterministic-pitcher/) | Is a pitcher who allows exactly one home run and then strikes out the side, every inning, worth a roster spot, and where should he pitch? | Analytic win probability from per-inning run distributions, a validated plate-appearance game engine, paired season simulations with bullpen fatigue | October 2026 |
| [Coors Field forecasting](rockies-forecasting/) | How does Coors Field change what a forecast should expect from pitchers and hitters? | Pitch-tracking physics (drag and Magnus force against air density), empirical Bayes against backtest for shrinkage, a hierarchical Stan model with a Lewandowski-Kurowicka-Joe (LKJ) prior | November 2026 |

## Headline results

### Infield defense

A ground-ball model that knows only the batted ball agrees with Statcast's Outs Above Average at r = 0.40 (shortstop) to 0.71 (first base). The difference between the two is shared by teammates within a season but does not pass to players new to a team, so public data cannot isolate a team's positioning. A shortstop's arm strength pays only on throws of 145 feet or more. One season of Single-A defense, from public tracking data, separated the young shortstops who stayed at the position from those who moved (area under the curve 0.85, on a cohort of 24).

### The deterministic pitcher

He belongs only in games where our lead is larger than the opponent's remaining half-innings: there he cannot lose. That rule is worth 2.63 wins a season (2.56 to 2.71 across sensitivity cases), almost all of it from leads real bullpens lost. The rest he gives the bullpen is worth about 0.13 wins, and using him in blowout losses costs wins.

### Coors Field forecasting

Drag and the spin force at Coors fall to 0.846 and 0.807 of their road values, against 0.830 predicted from air density, but swings and misses per swing do not fall. Empirical Bayes picks a prior about half as strong as the next-season forecast wants and still gets 93 percent of the gain. Rockies hitters' Coors gaps in batting average on balls in play (BABIP) behave like a park effect, while their strikeout gaps vary by hitter.

## Running

Python 3 with numpy, pandas, scipy, matplotlib, scikit-learn and statsmodels; the Stan model also needs cmdstanpy and CmdStan. Run each project's scripts from its own folder, in the order its README gives.
