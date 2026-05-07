# Computing Player Statistics

To compare player statistics to similarity scores, values for
different statistical categories and metrics were drawn from
the Baseball Reference (BRef) database. Further metrics (e.g. batting
average) were then calculated from these "raw" values in order to
properly compute player metrics over the span of their career
falling within the survey period (2016-25). Player data is separated
into batters (non-pitchers) and pitchers, since relevant stats
for these categories differ in most cases. "Raw" values are contained 
in `[player type]_stats_raw.csv` files, and computed statistics are 
contained in `[player type]_stats_calc.csv` files. The columns of
these files, including basic definitions of different stats and
equations for their calculation if necessary, are described below.

## batter_stats_raw.csv

- **person_id:** Unique identification token for each player, used as
    the token for their entity embedding. Maps to BRef IDs; can be 
    entered into the search bar on the [BRef site](https://www.baseball-reference.com/) 
    to navigate to the corresponding player info page.
- **position:** Player's primary position. Determined by counting the
    total occurrences of each position in a player's 
    `primary_pos_season` column (BRef database) for all relevant
    seasons and selecting the most frequently occurring position.
- **years:** Number of regular seasons played in the majors within
  the survey period (max 10). Determined by counting number of rows
    for player in BRef database where `phase_id` is 'reg' (as opposed
    to phases like the postseason) and year is within the survey period.
- **team:** Primary team played for during the survey period. Determined
    in the same way as position, but with `team_id` info. As a result
    of choosing one team for evaluation purposes, this inherently does
    not reflect a player playing for multiple teams, and the calculation
    does not function based on total games played per team.
- **b_games:** Number of games in which the batter appeared.
- **b_ab:** Number of at bats (AB) for the player. Used as the denominator
    for calculating many rate stats. See definition [here](https://www.mlb.com/glossary/standard-stats/at-bat).
- **b_pa:** Number of plate appearances (PA) for the player. At bats are a subset 
  of plate appearances. See definition [here](https://www.mlb.com/glossary/standard-stats/plate-appearance).
- **b_h:** Number of hits for the player. See definition [here](https://www.mlb.com/glossary/standard-stats/hit).
- **b_bb:** Number of walks for the player. Includes intentional walks.
  See definition [here](https://www.mlb.com/glossary/standard-stats/walk).
- **b_hbp:** Number of hit-by-pitches for the player. Batters hit by a pitch
    are automatically awarded first base, but a HBP is not considered
    a walk. See definition [here](https://www.mlb.com/glossary/standard-stats/hit-by-pitch).
- **b_sf:** Number of sacrifice flies for the player. See definition [here](https://www.mlb.com/glossary/standard-stats/sacrifice-fly).
- **b_tb:** Total bases for the player (extracted directly from BRef database).
    Computed as (1 × Singles) + (2 × Doubles) + (3 × Triples) + (4 × Home Runs).
- **b_so:** Number of strikeouts for the player.
- **b_hr:** Number of home runs for the player.
- **b_war:** Amount of WAR (Wins Above Replacement) for the player.
    WAR is an advanced metric that attempts to represent a player's
    value in all aspects of the game (batting, fielding, etc.). 
    Different versions of WAR exist that rely on different statistics
    in their calculations; this WAR value is Baseball Reference WAR,
    referred to as bWAR or rWAR. Details on its calculation
    can be found [here](https://www.baseball-reference.com/glossary/wins-above-replacement/).
- **b_wpa_bat:** WPA (Win Probability Added) for the player based on
    offensive contributions. Relatively self-explanatory; a change of
    +/- 1 in WPA indicates one win added or lost. Details on its
    calculation can be found [here](https://www.baseball-reference.com/about/wpa.shtml).
- **b_leverage_index_avg:** Average leverage index (aLI) for the player.
    Defined as average "pressure" faced by the player, where 1.0 is
    average, < 1.0 is low pressure, and > 1.0 is high pressure.
    See **b_wpa_bat** for details on its calculation. The value in
    the file here is computed by summing the values for each season
    and dividing by the number of seasons to get an overall average.
- **b_wpa_li_adjusted:** WPA/LI, or "situational wins", for the player.
    Sum of each play's WPA divided by its LI. Does not depend on context
    of at bats; WPA described above does. See **b_wpa_bat** for more details.
- **b_clutch:** "Clutch factor" for the player. The difference between
  context-dependent WPA and context-neutral WPA. Calculated with the
    formula WPA/aLI - WPA/LI (as shown above). See **b_wpa_bat** for more details.

## batter_stats_calc.csv

- **person_id, position, years, team:** See above.
- **b_avg:** Batting average. Calculated as hits per at bat (b_h / b_ab).
- **b_obp:** On-base percentage. Calculated as hits + walks + HBP divided by
    at bats + walks + HBP + sacrifice flies ((b_h + b_bb + b_hbp) / (b_ab + b_bb + b_hbp + b_sf)).
- **b_slg:** Slugging percentage. Calculated as total bases per at bat
  (b_tb / b_ab).
- **b_iso:** Isolated power. Only considers a player's extra-base hits
  (i.e. not singles). Calculated as total bases minus hits, divided by at bats
  ((b_tb - b_h) / b_ab).
- **b_ops:** On-base plus slugging. Commonly used to judge a player's
    overall offensive performance. Calculated by adding OBP and SLG
  (b_obp + b_slg), as the name suggests.
- **b_k_rate:** Strikeout rate. Calculated as the percentage of all
    plate appearances ending with a strikeout (b_so / b_pa).
- **b_hr_rate:** Home run rate. Calculated as home runs per at bat
  (b_hr / b_ab).
- **b_clutch:** See **b_clutch** above.
- **b_war_162:** Player's WAR per 162 games (a full baseball season).
  Calculated as total WAR divided by games played divided by 162
  (b_war / (b_games / 162)).

## pitcher_stats_raw.csv

- **person_id, position, years, team:** See above.
- **p_g:** Number of games in which the pitcher appeared.
- **p_gs:** Number of games started by the pitcher (as opposed to
  entering the game later).
- **p_er:** Earned runs for the pitcher. See definition [here](https://www.mlb.com/glossary/standard-stats/earned-run).
- **p_ip_outs:** Number of outs gotten by the pitcher (including
  outs made by the defense while the pitcher is in the game). Used to
    calculate number of innings pitched (p_ip, not stored in files) by
    dividing by 3.
- **p_bb:** Number of walks allowed by the pitcher. Includes intentional walks.
- **p_h:** Number of hits allowed by the pitcher.
- **p_so:** Number of strikeouts (i.e. batters struck out) by the pitcher.
- **p_war:** See **b_war** above.
- **p_game_score_sum:** Total sum of pitcher's game scores. Game score
    is a metric that measures a starting pitcher's effectiveness in a 
    given game. GmSc is not calculated for non-starting pitchers
  (a.k.a. relief pitchers/relievers). Details on its calculation can be found [here](https://www.baseball-reference.com/glossary/game-score/).
- **p_qs:** Number of quality starts by the pitcher. A starting pitcher
    earns a quality start by pitching at least six innings and allowing
    no more than three earned runs. Does not apply to relievers.

## pitcher_stats_calc.csv

- **person_id, position, years, team:** See above.
  - Note on **position**: All pitchers have 'P' listed as their position
    in the BRef database. To determine if they should be a starting pitcher
    or a reliever, I divide their total games by their total games started,
    and use 0.6 (60%) as the threshold above which a pitcher is a starter
    and below which they are a reliever. That is, if at least 60% of their
    appearances are starts, they are a starting pitcher.
- **p_era:** Earned run average. Calculated as the number
    of earned runs allowed per nine innings (p_er / p_ip * 9).
- **p_whip:** Walks plus hits per inning pitched. Extremely
  self-explanatory. Calculated as (p_bb + p_h) / p_ip.
- **p_k9:** Strikeouts per nine innings. Calculated as (p_so / p_ip) * 9.
- **p_bb9:** Walks per nine innings. Calculated as (p_bb / p_ip) * 9.
- **p_k_bb:** Strikeout to walk ratio. Calculated as p_so / p_bb.
- **p_avg_game_score:** Average game score. Calculated as total game score sum
    divided by games started (p_game_score_sum / p_gs). Not calculated for relievers.
- **p_qs_rate:** Quality start rate. Calculated as number of quality starts
    divided by games started (p_qs / p_gs). Not calculated for relievers.
- **p_war_162:** Similar in concept to **b_war_162**, but scaled to
    account for the fact that pitchers don't appear in every game
    of a season (or even most of them). BRef therefore normalizes a
    "single season" as 68 appearances, calculated by adding 
  games and games started (p_g + p_gs). The formula for calculating
      WAR/162 for pitchers is p_war * (68 / (p_g + p_gs)).

## Position Abbreviations
- **1B:** First baseman.
- **2B:** Second baseman.
- **3B:** Third baseman.
- **SS:** Shortstop.
- **LF:** Left fielder.
- **CF:** Center fielder.
- **RF:** Right fielder.
- **OF:** Outfielder (encompasses LF/CF/RF).
- **C:** Catcher.
- **DH:** Designated hitter.
- **SP:** Starting pitcher.
- **RP:** Relief pitcher.