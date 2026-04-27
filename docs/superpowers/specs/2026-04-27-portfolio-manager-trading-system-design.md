# Portfolio Manager Trading System Design

Date: 2026-04-27
Branch: design/portfolio-manager-system
Project: ElonQuantAgent

## 1. Goal

Upgrade the current project from a single-name technical analysis system into a live-trading assistant for a personal US-equities account. The target system must combine:

- single-name trend and entry analysis
- account-level position sizing
- portfolio risk controls
- manager-style allocation guidance
- support for a mixed-horizon book: core holdings plus tactical trend trades

The design is optimized for a user who primarily trades US equities, occasionally may consider Hong Kong equities later, prefers a practical live system, and wants account-level drawdowns managed around a 15% maximum discomfort threshold.

## 2. User profile and investment context

The system is designed around the following user-specific constraints and preferences:

- Personal live-trading account, not paper trading or pure research
- Primary market: US equities
- Secondary market: Hong Kong equities later, not in the first phase
- Mixed holding periods:
  - core positions can be held around six months
  - tactical positions are used for weekly trend capture
- Mixed universe:
  - core large-cap names may be held
  - high-beta growth, narrative-driven, and event-sensitive names are also traded
- Preferred output:
  - both single-name trade recommendations and manager-style portfolio guidance
- Drawdown preference:
  - historical drawdowns reached around 25%
  - target system should help keep drawdown more consistent with a 15% maximum discomfort level

Additional context from the supplied Gemini chat transcripts indicates that the user often trades asset-proxy equities and theme proxies rather than plain vanilla stocks. Examples include crypto-beta names such as MSTR and SBET, along with names like PDD and CRCL. This means the system must model not only stock-level behavior but also underlying factor and proxy exposure.

## 3. Current project assessment

The current repository is a price-driven multi-agent analysis system with the following main components:

- indicator agent
- pattern agent
- trend agent
- decision agent
- `core/trading_graph.py` orchestrating a sequential single-name pipeline

Current strengths:

- already structured around modular analysis stages
- able to compute and narrate technical indicators and chart patterns
- already supports web and programmatic flows
- already produces trade-oriented natural language outputs

Current limitations:

- final output is too coarse, centered around LONG/SHORT style conclusions
- no explicit account state, portfolio state, or risk budgeting layer
- no concept of core vs tactical book
- no market regime control over total exposure
- no concentration or correlation control across related names
- no manager-style dashboard for account-level decisions

The current project behaves more like a single-name analysis engine than an account-level decision system.

## 4. Product direction

The project should evolve into a two-layer portfolio trading system:

- bottom layer: score and classify individual opportunities
- top layer: manage risk, capital allocation, and account-level decisions

The system should answer two different questions in sequence:

1. Is this stock attractive on its own?
2. Is this stock attractive inside the current account and risk budget?

This distinction is the central design principle.

## 5. Recommended approach

Three broad approaches were considered:

1. strengthen only the single-name prediction engine
2. build a dual-layer portfolio system with market-state-driven risk control
3. prioritize a manager dashboard and treat trade recommendations as secondary

Recommended approach: **dual-layer portfolio system with market-state-driven risk control**.

Why this approach was chosen:

- it best matches the user’s real workflow: live trading plus portfolio management
- it supports the user’s mixed holding horizons
- it directly addresses the most important problem: keeping drawdowns closer to the user’s comfort zone
- it can be built incrementally on top of the current repository without discarding the existing agent structure

## 6. Target architecture

### 6.1 High-level layers

The target system should include four logical layers:

1. **Market Regime Layer**
2. **Single-Name Scoring Layer**
3. **Portfolio Construction Layer**
4. **Execution Decision Layer**

### 6.2 Market Regime Layer

Purpose:

- determine whether the system should be in attack, selective attack, neutral, or defense mode
- produce top-down exposure limits before evaluating whether new trades are allowed

Inputs may include:

- major index trend state
- sector and breadth proxies
- volatility proxies
- risk-asset synchronization across themes relevant to the account
- optional proxy inputs such as BTC or ETH when managing crypto-beta equities

Outputs:

- regime classification
- target total exposure range
- target tactical exposure range
- allowed aggressiveness for new risk
- regime-based restrictions on weaker setups

### 6.3 Single-Name Scoring Layer

Purpose:

- replace coarse LONG/SHORT conclusions with a structured scorecard for each candidate and current holding

This layer should still rely on the existing analysis foundation:

- indicator agent
- pattern agent
- trend agent

But the current decision agent should be repurposed into a standardized single-name scorer.

Required outputs for each name:

- trend quality
- entry quality
- valuation or stretch condition
- catalyst quality
- volatility awareness
- correlation penalty relative to the current account
- portfolio fit
- recommended book type: core or tactical
- recommended action: buy, add, hold, trim, exit, watch, or blocked

### 6.4 Portfolio Construction Layer

Purpose:

- combine market regime, account status, current positions, and single-name scorecards into an account-level decision set

This layer should decide:

- whether new risk can be added at all
- how much capital can be allocated
- whether a name belongs in the core or tactical book
- whether a setup is blocked by concentration or correlation constraints
- whether the account should be de-risked even when some individual names still look attractive

### 6.5 Execution Decision Layer

Purpose:

- turn portfolio decisions into user-facing directives that are actionable in live trading

Outputs must include:

- portfolio-level directives
- single-name trade tickets
- explanations for allowed actions
- explanations for blocked actions

## 7. Account framework

### 7.1 Core book

The core book is intended for medium-term and longer holdings, typically around three to six months or longer.

Characteristics:

- used for higher-conviction trend or thesis exposure
- lower turnover
- better tolerance for normal fluctuations
- suitable for large-cap leaders or strong structural themes

Suggested normal range:

- 50% to 70% of account exposure

Suggested initial and maximum position sizing:

- initial position often around 8% to 12%
- maximum around 15%, subject to volatility and concentration controls

### 7.2 Tactical book

The tactical book is intended for weekly trend capture and more agile trading around breakouts, pullbacks, and catalysts.

Characteristics:

- smaller sizing
- faster stop discipline
- focused on timing and asymmetry rather than long-duration conviction

Suggested normal range:

- 10% to 30% of account exposure

Suggested initial and maximum position sizing:

- initial position often around 3% to 6%
- maximum around 8%, subject to volatility and correlation constraints

### 7.3 Cash and defensive reserve

Cash is not treated as leftover capital. It is a deliberate risk reserve.

Suggested normal range:

- 10% to 20%

Under defensive conditions, cash can rise to:

- 30% to 50%

## 8. Risk budgeting framework

### 8.1 Per-trade risk budget

Each trade should first define:

- entry price
- invalidation or stop level
- permitted loss as a percentage of account net asset value

Recommended ranges:

- core book risk per trade: 0.75% to 1.25% NAV
- tactical book risk per trade: 0.35% to 0.75% NAV

Position size should be derived from risk budget and stop distance, not selected ad hoc.

### 8.2 Total account risk budget

The system must also track total stop-based account risk across all active positions.

Recommended control:

- if all stops were hit, theoretical total account loss should generally remain within about 4% to 6% NAV at any one time

This should act as a hard constraint for whether new positions are allowed.

### 8.3 Drawdown defense ladder

Because the user is especially sensitive to drawdowns beyond roughly 15%, the system should include a three-stage defense ladder.

#### Stage 1: Normal operation

- operate under standard regime-based risk rules
- use target allocations for core, tactical, and cash

#### Stage 2: Warning mode

Triggered around 6% to 8% drawdown from the recent high.

Actions:

- raise the minimum quality threshold for new positions
- reduce tactical-book ceiling
- increase cash reserve
- avoid weaker or lower-conviction additions

#### Stage 3: Defense and recovery mode

Triggered around 10% to 12% drawdown from the recent high.

Actions:

- prioritize de-risking over offense
- sharply limit high-beta trial positions
- focus on preserving capital and reducing exposure overlap
- treat the 15% drawdown ceiling as a hard risk objective

## 9. Market-state machine

The system should classify conditions into four states.

### 9.1 Risk-On

Characteristics:

- major risk assets trending well
- strong follow-through in leading names
- volatility manageable
- broad or theme-supported momentum

Implications:

- allow higher total exposure
- allow tactical participation in strong breakouts and continuation setups

### 9.2 Selective Risk-On

Characteristics:

- index-level conditions may be mixed
- only a few themes are working
- strength is concentrated in a narrow leadership group

Implications:

- allow trading only in the strongest themes
- tactical book remains active but selective
- avoid weak branches and over-diversification into mediocre names

### 9.3 Neutral

Characteristics:

- choppy conditions
- weaker breakout follow-through
- lower trend persistence

Implications:

- lower overall exposure
- keep only stronger core positions
- reduce tactical aggressiveness
- raise the bar for new entries

### 9.4 Risk-Off

Characteristics:

- trend damage at the index or theme level
- increased volatility
- high-beta names and proxies sell off together

Implications:

- reduce total exposure sharply
- minimize tactical exposure
- prioritize cash, de-risking, and account protection

## 10. Proxy exposure and factor mapping

A critical requirement for this user is identifying when apparently different stocks represent the same underlying directional bet.

Examples:

- MSTR as a BTC proxy
- SBET or similar names as ETH or treasury-style crypto-beta proxies
- large-cap platform names as mega-cap tech beta
- PDD and similar names as China ADR exposure

The system should include a risk-mapping layer that assigns each holding and candidate to one or more factor buckets such as:

- BTC beta
- ETH beta
- mega-cap tech beta
- China ADR beta
- event-driven beta
- rate-sensitive beta

This mapping should be used to detect hidden concentration. The goal is to avoid a situation where multiple different tickers create the same directional exposure and overwhelm the account.

## 11. Hard portfolio rules

The portfolio manager layer should enforce explicit constraints.

### 11.1 Total exposure by regime

Illustrative default exposure caps:

- Risk-On: 70% to 85%
- Selective Risk-On: 55% to 75%
- Neutral: 35% to 60%
- Risk-Off: 15% to 35%

### 11.2 Tactical-book cap by regime

Illustrative default tactical caps:

- Risk-On: 25% to 30%
- Selective Risk-On: active but selective
- Neutral: 10% to 15%
- Risk-Off: 0% to 5%

### 11.3 Factor-group exposure limits

Illustrative limits:

- BTC proxy exposure: no more than about 25% to 30%
- ETH proxy exposure: no more than about 20% to 25%
- any high-correlation theme group: no more than about 35% to 40%

Exact thresholds should remain configurable.

### 11.4 Volatility-aware single-name caps

Single-name maximum exposure should be adjusted by:

- realized or implied volatility proxy
- correlation to the rest of the book
- whether the name is in the core or tactical book

A 10% position in a broad ETF is not treated the same as a 10% position in a volatile proxy stock.

### 11.5 Drawdown-based de-risking

The system must automatically reduce aggressiveness when account drawdown moves through warning and defense thresholds.

## 12. Single-name scorecard design

The current LONG/SHORT-style output should be replaced with a six-dimension scorecard.

Suggested dimensions:

1. **Trend Score**
2. **Entry Quality Score**
3. **Valuation/Stretch Score**
4. **Catalyst Score**
5. **Correlation Penalty**
6. **Book Fit Score**

This should produce a standardized output for each name including:

- tradability
- core vs tactical suitability
- suggested sizing range
- entry zone
- invalidation level
- first and second profit objectives
- blocked status if the account already carries too much correlated exposure

A key rule is that a stock can be directionally attractive but still not be eligible for new capital because of account-level constraints.

## 13. Output design

The final output should always be split into two views.

### 13.1 Portfolio manager dashboard

This view should come first and answer:

- What state is the market in today?
- How much exposure should the account carry?
- How much tactical exposure is allowed?
- Which factor exposures are crowded or dangerous?
- How much new risk budget remains?
- What are the most important manager-level actions today?

Suggested fixed fields:

- account NAV
- current gross exposure
- cash percentage
- core-book percentage
- tactical-book percentage
- current drawdown
- recent account volatility
- market regime
- rationale for regime
- suggested total exposure range
- remaining risk budget
- factor exposures
- concentration warnings
- top manager actions

### 13.2 Single-name trade ticket

Each candidate and current position should be represented by a structured trading card.

Suggested fields:

- ticker
- role: core, tactical, or not suitable
- total score
- market fit
- account fit
- current action: buy, add, hold, trim, exit, watch, or blocked
- trend assessment
- entry type: breakout, pullback, hold-only, wait-for-confirmation, etc.
- suggested position size
- entry zone
- invalidation level
- profit targets
- trigger condition
- reason for inaction if blocked
- correlation level to current book
- whether it consumes an already crowded factor budget
- whether the setup is stretched or crowded

### 13.3 Rejection logic is a first-class output

The system must be able to clearly explain why a stock should not be added even when its standalone trend remains positive.

Examples:

- trend is positive, but exposure overlaps too much with existing crypto-beta holdings
- valuation and sentiment are too stretched for a fresh entry
- tactical-book budget is unavailable in the current regime
- account is in drawdown defense mode and cannot add high-beta risk

This rejection capability is essential for reducing impulsive additions.

## 14. Data model

The current project state is too single-name oriented. The target design requires four new major structures.

### 14.1 AccountState

Tracks overall account condition, for example:

- total equity or NAV
- available cash
- current holdings
- recent equity high
- current drawdown
- core-book usage
- tactical-book usage

### 14.2 PositionState

Tracks each position, for example:

- ticker
- shares
- market value
- unrealized PnL
- holding days
- book type
- factor or theme mapping
- stop level
- thesis status

### 14.3 SingleNameScore

Tracks structured analysis output, for example:

- trend_score
- entry_score
- valuation_stretch_score
- catalyst_score
- volatility_score
- correlation_penalty
- portfolio_fit_score
- recommended_book
- recommended_action

### 14.4 PortfolioDirective

Tracks account-level outputs, for example:

- market_regime
- target_gross_exposure
- target_core_exposure
- target_tactical_exposure
- remaining_risk_budget
- crowded_exposures
- add_candidates
- trim_candidates
- blocked_candidates

## 15. Implementation prerequisites

Before implementing `regime_agent`, `portfolio_manager_agent`, and the account-level decision pipeline, the project needs a minimal set of prerequisites. These are not optional polish items. They are the foundations that keep the system from collapsing back into prompt-only portfolio logic.

### 15.1 Must-have prerequisites before broader implementation

#### 1. Account-state input structure

The system needs a standardized account input before it can manage sizing, drawdown, or portfolio-level decisions.

Minimum required fields:

- current NAV
- available cash
- current holdings list
- market value, cost basis, and unrealized PnL for each position
- core vs tactical classification
- stop or invalidation level
- recent NAV high or a stable drawdown reference point

Acceptance criteria:

- account state is provided as typed structured input rather than free-form text
- the system can reliably compute gross exposure, cash percentage, core-book usage, tactical-book usage, and current drawdown

#### 2. Position and candidate-universe structures

The system must distinguish between reevaluating existing holdings and scoring fresh opportunities.

Minimum required structures:

- `positions` for current holdings
- `candidates` for new names under evaluation

Each name should include at least:

- ticker
- market
- current price
- expected book type if known
- theme or factor tags, even if initially manual
- whether the account already has exposure

Acceptance criteria:

- the same scoring pipeline can process both holdings and new candidates
- the portfolio layer can clearly distinguish add, hold, trim, and exit from new-entry and blocked-entry decisions

#### 3. Factor-mapping and proxy-exposure table

This is one of the most important prerequisites for the target user. Without it, the system cannot detect when different tickers are really the same directional bet.

The first version can be manually maintained. It does not need to be fully automated.

Suggested first buckets:

- BTC beta
- ETH beta
- mega-cap tech beta
- China ADR beta
- event-driven beta
- rate-sensitive beta

Illustrative mappings:

- MSTR → BTC beta
- SBET → ETH or treasury-style crypto beta
- PDD → China ADR beta

Acceptance criteria:

- every position and candidate maps to one or two primary factor buckets
- the system can compute current exposure by factor bucket
- the portfolio layer can block new additions based on hidden concentration

#### 4. Market-regime input source

Without a regime input, the system cannot enforce the design principle of setting portfolio aggression before approving trades.

The first version does not need a sophisticated macro model, but it does need a repeatable input framework.

Suggested regime inputs:

- major index trend proxies
- breadth or leadership proxies
- volatility proxies
- risk proxies tied to the user’s main trading themes, such as BTC and ETH

Acceptance criteria:

- the system can reliably classify one of four states: Risk-On, Selective Risk-On, Neutral, or Risk-Off
- each state maps to default total-exposure and tactical-exposure ranges

#### 5. Single-name scorecard schema

Before the portfolio layer can work, single-name output must move from LONG/SHORT to a standardized scorecard.

Minimum first-version fields:

- `trend_score`
- `entry_score`
- `valuation_stretch_score`
- `catalyst_score`
- `volatility_score`
- `correlation_penalty`
- `portfolio_fit_score`
- `recommended_book`
- `recommended_action`
- `invalidation_price`
- `suggested_position_range`

Acceptance criteria:

- holdings and candidates produce the same schema
- portfolio logic consumes structured fields rather than parsing narrative LONG/SHORT text

#### 6. Risk-parameter configuration table

The core hard rules must exist as configuration rather than being buried in prompts or scattered constants.

Minimum required parameters:

- total-exposure caps by regime
- tactical-book caps by regime
- initial position guidance for core and tactical books
- per-trade risk budgets for core and tactical books
- factor-exposure caps
- warning and defense drawdown thresholds

Acceptance criteria:

- all critical thresholds can be reviewed and changed in one place
- the portfolio-manager output references one consistent parameter source

### 15.2 Strongly recommended during the first implementation wave

These items are not absolute blockers for the first code changes, but the project will become difficult to validate without them.

#### 7. NAV and drawdown calculation logic

The system should make drawdown logic explicit:

- whether NAV is tracked daily or event-based
- how recent equity highs are defined
- whether unrealized PnL is included
- whether account drawdown and single-position drawdown are treated separately

Acceptance criteria:

- given holdings and an NAV time series, the system can reliably compute current drawdown and drawdown tier
- drawdown tiers can trigger warning or defense behavior automatically

#### 8. Minimal replay and scenario-validation framework

This system must validate not only whether analysis sounds right, but whether decisions remain safe under portfolio constraints.

The first version does not need full quantitative backtesting. Scenario replay is enough.

Minimum capability:

- load a sample account state
- load a sample market regime
- load a set of candidate scorecards
- generate portfolio directives
- verify whether hard rules are violated

Representative scenarios should include:

- Risk-On with room to add risk
- Selective Risk-On with narrow leadership
- Neutral with mixed opportunity quality
- Risk-Off where strong standalone names are still blocked
- concentration caused by multiple crypto-proxy positions

#### 9. Stable portfolio-manager output templates

The system should lock down a first-version structure for the manager dashboard and trade-ticket outputs early.

Minimum templates:

- portfolio-manager dashboard fields
- single-name trade-ticket fields
- blocked-candidate explanation fields
- forced-trim explanation fields

Acceptance criteria:

- output structure remains stable across runs
- the user can directly read whether risk can be added, what kind of risk is allowed, and what is blocked

### 15.3 Items that can be added incrementally later

These matter, but they should not block the high-value first and second stages.

#### 10. Automated factor mapping

The first version can use a manually curated mapping table. Later versions can add:

- rule-based symbol tagging
- correlation-assisted tagging
- theme-library-driven classification

#### 11. Broker or execution integration

The first implementation does not need auto-execution.

Later stages can add:

- automatic position sync
- order export
- broker API integration

#### 12. Broader valuation and fundamental modules

The initial system should prioritize:

- trend quality
- entry quality
- exposure overlap
- portfolio risk budgeting

Valuation only needs to cover stretch, overheating, and premium-risk awareness in the first version.

### 15.4 Recommended implementation order

Implementation should follow this sequence rather than spreading effort across every layer at once:

1. account and position data structures
2. candidate-universe structure
3. factor-mapping table
4. single-name scorecard schema
5. market-regime inputs and state machine
6. risk-parameter configuration
7. portfolio-level blocking and constraint logic
8. NAV and drawdown tiering
9. minimal scenario replay tests
10. portfolio-manager dashboard and trade-ticket outputs
11. later automation layers

### 15.5 Readiness gate for implementation

Implementation should not be treated as truly underway until the following exist:

- `AccountState`
- `PositionState`
- `SingleNameScore`
- `PortfolioDirective`
- a maintainable candidate-universe input path
- a first-version factor-mapping table
- a first-version market-regime input layer
- a unified risk-parameter configuration source
- a minimal scenario test set

If these pieces are missing, the project will likely drift back into a single-name analysis engine with portfolio language layered on top.

## 16. Minimal-change repository strategy

The design should preserve as much of the current code structure as possible.

### 16.1 Keep existing analytical foundations

Retain:

- indicator agent
- pattern agent
- trend agent

These remain the lower-level analytical engines.

### 16.2 Redefine the current decision layer

The current decision agent should stop acting as the final account-level judge.

Instead, it should become a **single_name_scorer** that emits structured scorecards.

### 16.3 Add new upper-layer modules

New modules should include:

- `regime_agent`
- `portfolio_manager_agent`
- `risk_mapper`

These can be integrated without replacing the current single-name logic.

## 17. TradingGraph evolution

The current `core/trading_graph.py` sequentially produces a single-name conclusion.

Target evolution:

### Phase A: single-name analysis pipeline

For each candidate or holding:

- indicator analysis
- pattern analysis
- trend analysis
- single-name scoring

### Phase B: portfolio decision pipeline

Feed the following into the account-level layer:

- market regime
- account state
- current holdings
- candidate scorecards
- holding scorecards

Then produce a complete account-level directive.

This changes the system from:

- one stock in, one conclusion out

into:

- account context plus many names in, portfolio directives plus trade tickets out

## 18. Incremental implementation path

To keep scope controlled, implementation should proceed in three stages, with the first two carrying most of the practical value.

### Stage 1: Standardize inputs and upgrade single-name output

This is the highest-leverage first step because it upgrades the current engine without requiring the full portfolio layer on day one.

Focus on:

- establishing `AccountState` and `PositionState`
- defining candidate-universe input structure
- creating the first factor-mapping table
- converting LONG/SHORT outputs into structured scorecards
- classifying opportunities into core or tactical suitability
- adding sizing, invalidation, and trade-setup structure

This stage creates the standardized inputs and outputs that every later layer depends on.

### Stage 2: Add account context and manager dashboard

This is the second major value stage because it turns the system from a better analyzer into a usable account assistant.

Introduce:

- account and holdings input flow
- total exposure, cash, core-book, and tactical-book tracking
- simple factor-exposure accounting
- market-regime assessment
- account-level risk budget and drawdown tiering
- first-version portfolio-manager dashboard

At this stage, the system learns to answer:

- can the account add risk today?
- what kind of exposure should be added or avoided?
- is the account already too crowded in a hidden factor bucket?

### Stage 3: Add strict portfolio constraints and automation-oriented outputs

Introduce:

- correlation-aware blocking
- factor-group exposure caps
- drawdown-triggered de-risking logic
- blocked-candidate and forced-trim outputs

At this point the system becomes a full live-trading plus manager-style assistant.

## 19. Testing strategy

Testing should cover both analysis correctness and decision safety.

### 19.1 Single-name scoring tests

Validate:

- scorecard schema integrity
- output stability for representative setups
- correct mapping of a name into core vs tactical preference

### 19.2 Account-state tests

Validate:

- drawdown calculations
- core vs tactical utilization
- cash and exposure accounting

### 19.3 Portfolio manager tests

Validate:

- regime-based exposure caps
- factor concentration detection
- blocked new positions when account-level limits are exceeded
- correct downgrade behavior during drawdown thresholds

### 19.4 End-to-end scenario tests

Representative scenarios should include:

- strong Risk-On regime with capacity to add risk
- narrow leadership and selective offense
- neutral regime with mixed single-name signals
- defensive regime where strong standalone names are still blocked
- multiple crypto-proxy positions causing correlation-based blocking

## 20. Error handling and boundary behavior

This design does not aim to create heavy fallback layers everywhere. It should remain practical and focused.

Boundary validation should apply to:

- user-entered account and holdings data
- externally fetched market data
- optional factor-mapping inputs for symbols

Internal system modules should rely on consistent typed structures rather than defensive branching for impossible states.

## 21. Non-goals for the first implementation

The initial implementation should not attempt to solve every portfolio problem at once.

Not first-phase goals:

- full Hong Kong market support
- fundamental model integration for every company
- options portfolio management
- tax optimization
- full institutional optimizer mathematics
- over-generalized support for every asset class

The first implementation should stay tightly focused on the user’s actual workflow.

## 22. Final design summary

The project should evolve from a single-name technical analysis engine into a portfolio-aware live-trading system for a personal US-equities account.

The resulting design is:

- bottom layer: trend, pattern, and indicator analysis converted into structured single-name scorecards
- top layer: market regime, account state, factor exposure, and portfolio risk management
- output layer: manager dashboard plus actionable trade tickets

The defining design rule is:

> a stock can be attractive on its own and still be the wrong trade for the account.

That rule is the core of the system and should guide the implementation plan.
