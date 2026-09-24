# Gold evaluation set

Generated from the PDFs in `data/raw/`. Source of truth is `gold_questions.jsonl`; this file is a readable rendering.
Answers were extracted by Claude from the paper text. Spot-check at least 10 before trusting the scores.

55 questions. Types: numeric (14), factoid (11), method (10), discrepancy (3), comparison (9), aggregation (5), unanswerable (3)

## q01 · numeric · p13

**Q.** In the dynamic logistic model applied to the IDX Composite (June 2019 to December 2021), what were the fit MAPE and RMSPE?

**A.** Fit MAPE 5.067% and RMSPE 6.754%.

## q02 · numeric · p13

**Q.** What MAPE did the dynamic logistic model achieve when forecasting the IDX Composite for 30, 45 and 60 trading days starting 4 November 2019?

**A.** 30 days: MAPE 4.332% (RMSPE 5.572%); 45 days: 4.204% (RMSPE 5.462%); 60 days: 4.446% (RMSPE 5.616%).

*Note:* The 45-day MAPE is lower than the 30-day MAPE for this single start date, so 'shorter is better' does not hold here. The table header also has a typo ('4 Nov - 14 Jan 2019').

## q03 · numeric · p13

**Q.** Across the repeated forecast experiments with different starting days on the IDX Composite, what share of simulations reached 'high accuracy' for each horizon?

**A.** Experiment A (30 days): 97.5%; B (45 days): 89.74%; C (60 days): 84.21%. Average MAPE 4.672%, 5.489%, 6.229% respectively.

*Note:* Text says 40 simulations per experiment, but B's counts sum to 39 and C's to 38 (35/39 = 89.74%, 32/38 = 84.21%).

## q04 · factoid · p13, p02, p01, p03

**Q.** What criteria are used to classify forecasting accuracy from MAPE, and what are the thresholds?

**A.** Lewis (1982) criteria: MAPE below 10% is high accuracy, 10-20% good, 20-50% reasonable, above 50% inaccurate.

*Note:* Multi-source: p13 Table 2, p01 Table 1, p02 and p03 cite Lewis. p05 uses the 10% threshold in its flowchart without the full table.

## q05 · factoid · p02

**Q.** Which banks were modelled in the logistic-differential-equation banking study, and why those banks?

**A.** Bank Mandiri (BMRI), Bank Rakyat Indonesia (BBRI), Bank Central Asia (BBCA) and Bank Negara Indonesia (BBNI), the four largest Indonesian banks by assets according to OJK (May 2021).

## q06 · numeric · p02

**Q.** What was the average fit MAPE across the four banks, and which bank did the logistic model fit best?

**A.** Average 6.331%. BBCA fit best (5.796%), then BMRI (6.289%), BBRI (6.495%), BBNI (6.745%).

## q07 · numeric · p02

**Q.** In the banking logistic-model study, how did forecast error behave during the early COVID-19 period?

**A.** MAPE rose by more than 20% between December 2019 and May 2020; the authors say the model could not capture the data well during that uncertain period.

## q08 · factoid · p03

**Q.** Which companies and time period were used in the discrete-time logistic model study of Indonesian private companies?

**A.** BBCA, HMSP (Sampoerna), ASII (Astra), GGRM (Gudang Garam) and UNVR (Unilever Indonesia); daily closing prices from 10 June 2019 to 26 October 2021, from Yahoo Finance.

## q09 · numeric · p03

**Q.** In the private-company logistic study, which company had the most stable forecast MAPE and what was its high-accuracy percentage?

**A.** UNVR: lowest MAPE standard deviation (2.293) and 97.5% of simulations in the high-accuracy band.

## q10 · method · p03

**Q.** Why did ASII's forecast MAPE exceed 30% in early 2020, according to the authors?

**A.** They attribute it to a decline of more than 95% in annual car wholesales in that period (ASII is an automotive company).

## q11 · factoid · p03

**Q.** What input format and software stack does the GUI in the private-company logistic study use?

**A.** A CSV with exactly two columns, Date and Close, dates as YYYY-MM-DD. Built with Python 3.9.1, Pandas 1.3.4, NumPy 1.20.0 and Matplotlib 3.3.4; results export as PDF or CSV.

## q12 · discrepancy · p03

**Q.** The abstract of the private-company paper gives fit and forecast MAPE averages of 6.158% and 4.736%. Are these averages across all five companies?

**A.** No. 6.158% and 4.736% are HMSP's fit and forecast MAPE in Tables II and IV. Across the five companies, fit MAPE ranges 5.518-6.158% (mean about 5.81%) and forecast MAPE 2.713-5.880% (mean about 4.09%).

*Note:* Tests whether the system reads tables instead of repeating the abstract. Prefer-the-table instruction in the system prompt should drive this.

## q13 · method · p05

**Q.** What equation defines the Dynamic Logistic Velocity-Acceleration model, and how are its parameters interpreted?

**A.** dv/dt = a(t) v^2 + b(t) v, with v = dS/dt, i.e. a second-order nonlinear ODE in the price S. a(t) is the nonlinear component related to market volatility and price sensitivity; b(t) is the linear component describing the short-term trend.

## q14 · factoid · p05

**Q.** What data and event were used to test the velocity-acceleration model?

**A.** Daily IDX Composite (JKSE) closing prices from July 2023 to July 2024, around the Indonesian presidential election of 14 February 2024.

## q15 · numeric · p05

**Q.** What fitting accuracy did the velocity-acceleration model achieve, and when was the error highest?

**A.** Overall fit MAPE 0.775%. Monthly MAPE ranged 0.43-1.25%, highest in May 2024 (1.22%) and June 2024 (1.25%), after the election.

*Note:* The paper also reports the same figure rounded as 0.78%.

## q16 · numeric · p05

**Q.** How did the velocity-acceleration model's forecast MAPE differ before and after the 2024 election for 45, 30 and 15-day windows?

**A.** Before: 0.796%, 0.611%, 0.558%. After: 1.242%, 1.222%, 0.946%. Error was higher after the election, attributed to increased volatility.

## q17 · method · p05

**Q.** Why does the velocity-acceleration fitting algorithm start at n = 4?

**A.** Velocity and acceleration need at least four previous data points. In the forecast algorithm, n = 4 to 7 fall back on actual data where predictions are unavailable, and the first fully model-driven iteration is n = 8.

## q18 · method · p12

**Q.** In the Lotka-Volterra stock study, how are the two 'interacting' stock prices chosen?

**A.** By minimum standard deviation: the months with the smallest and second-smallest standard deviation of monthly prices across years (indices p and q) are taken, since low deviation is treated as more reliable.

## q19 · method · p12

**Q.** Which numerical method solves the Lotka-Volterra system, and how are its starting values obtained?

**A.** The fourth-order Adams-Bashforth-Moulton predictor-corrector method, chosen for efficiency and accuracy; the initial values are computed with a Runge-Kutta method.

## q20 · comparison · p12

**Q.** Which sector had the lowest and which the highest stock growth rate in the Lotka-Volterra study?

**A.** Tourism had the lowest growth rate; pharmaceuticals the highest; telecommunications was stable. KAEF and KLBF are described as positively affected by COVID-19.

## q21 · factoid · p12

**Q.** What data frequency and simulation settings were used in the Lotka-Volterra study?

**A.** Monthly closing prices from January 2016 to December 2021 (Yahoo Finance); initial conditions x(0) = 50 and y(0) = 100, step size h = 0.1, simulated over 100 days.

## q22 · numeric · p08

**Q.** What was the best accuracy of the LSTM model on BBCA, and under what configuration?

**A.** 94.59%, using 1 year of training data and 100 epochs (High price). With 3 years of data the best was 88.47%. Data split 80% training / 20% testing.

*Note:* The abstract states 94.57% while Table 1 and the conclusion state 94.59%.

## q23 · factoid · p08

**Q.** On what date did BBCA and BMRI stock prices reach their 2020 lows?

**A.** 24 March 2020 (the first confirmed Indonesian COVID-19 case was 2 March 2020).

## q24 · comparison · p10

**Q.** Which volatility factor most influences the day-ahead volatility of U.S. oil and gas exploration and production firms?

**A.** U.S. market volatility, followed by the firm's own and industry-level volatility. Oil volatility matters less; world equity volatility adds no incremental information; natural gas is negative in-sample and unrelated out-of-sample.

## q25 · numeric · p10

**Q.** By how much did the all-factor HAR model improve on the benchmark HAR in Lyocsa and Todorova's study?

**A.** Up to 3.88% under the QLIKE loss (3.72% under MSFE).

## q26 · factoid · p10

**Q.** What sample and data did Lyocsa and Todorova use?

**A.** 15 S&P 500 Oil & Gas Exploration & Production firms, January 2007 to December 2017, 1-minute data from Thomson Reuters Tick History via Sirca. Realized volatility averaged across 5, 10, 15 and 30-minute sampling; rolling window of 1000 observations; Model Confidence Set at 90%.

## q27 · numeric · p07

**Q.** What are the short-run and long-run effects of geopolitical risk on oil price volatility in the ARDL study?

**A.** Positive in both. Short run: coefficient on the change in log GPR is 0.1438 (5% level). Long run: 12.9467 (1% level). The USD index is positive in the short run (14.9732) but has no long-run effect. The error-correction term is -0.0281, so 2.81% of disequilibrium is corrected per day. Model ARDL(4,4,1), bounds F = 19.29.

## q28 · factoid · p07

**Q.** What data and volatility measure did Truong et al. use?

**A.** Daily Caldara-Iacoviello GPR index, WTI crude oil price and USD index, 4 January 2010 to 31 December 2022 (3,357 observations). Oil volatility generated from a GARCH(1,1) model.

## q29 · factoid · p14

**Q.** What sample and preferred model did Smales use to study geopolitical risk and oil-stock volatility?

**A.** January 1986 to May 2018, 8,166 daily observations of WTI and the S&P 500 (DataStream). Among BEKK, diagonal, CCC and DCC multivariate GARCH models, the DCC model is preferred. Volatility spillovers run from oil to stock returns; oil futures may hedge geopolitical risk for stock investors.

## q30 · comparison · p14

**Q.** According to Smales, how does a rise in geopolitical risk affect oil returns versus stock returns?

**A.** Positive oil returns and negative stock returns; the effect is larger for oil, consistent with geopolitical risk being tied to supply disruption.

## q31 · numeric · p14

**Q.** When did the GPR index reach its highest level in Smales's sample, and at what value?

**A.** 1,025, at the start of the second Gulf War in March 2003.

## q32 · aggregation · p18

**Q.** According to Ozdemir et al., which commodities are negatively affected, positively affected, or not significantly affected by geopolitical risk?

**A.** Negative: gold, silver, natural gas. Positive: wheat, corn, soybeans, cotton, zinc, nickel, lead, WTI oil, Brent oil. No significant impact: platinum, cocoa, coffee, copper. Negative geopolitical shocks raise volatility more than positive ones. EGARCH models, daily data 4 January 2010 to 30 June 2023, GPR plus GPRACT and GPRTHREAT sub-indices.

## q33 · comparison · p17

**Q.** How did the Heston stochastic volatility model compare with GARCH-type models for crude oil volatility?

**A.** The Heston model, simulated with Euler-Maruyama, produced smaller errors than GARCH, EGARCH and TGARCH. Among the GARCH family, plain GARCH was most accurate (RMSE 0.01681 for WTI, 0.01472 for Brent). Data: EIA daily WTI and Brent spot prices, 4 January 2009 to 31 December 2019.

## q34 · method · p17

**Q.** What condition guarantees that the variance process in the Heston model stays non-negative?

**A.** The Feller condition, 2*beta*theta > sigma^2 (reversion rate times long-run variance, doubled, exceeds the squared volatility of volatility).

## q35 · discrepancy · p19

**Q.** Which markets exhibit mean reversion according to Enow's Hurst exponent analysis?

**A.** The paper reports mean reversion in the Nasdaq (H = 0.573), CAC 40 (0.592), DAX (0.587) and Nikkei 225 (0.592), all significant at 5%, but not the JSE (H = 0.530, p = 12.95%). Period 1 June 2018 to 1 June 2023.

*Note:* Caveat for the human grader: every H is above 0.5, which under the standard interpretation indicates persistence, not mean reversion. The paper's decision rule appears inverted. The system should report the paper's claim; flagging the tension is a bonus, not required.

## q36 · comparison · p01

**Q.** Which of Li et al.'s four dynamic models performed best on the Taiwan stock index, and did it beat the martingale?

**A.** Model D. Its fit MAPE (0.3467%) beat the martingale (0.7608%), but for forecasting the martingale was slightly better (0.4306% vs 0.4658%).

## q37 · method · p01

**Q.** What functional forms underlie Li et al.'s dynamic models, and what is assumed about the coefficients when forecasting?

**A.** Each model is a parameterized ODE in the form of either logistic growth or Newton's law of cooling, with time-varying coefficients. When forecasting, coefficients are assumed constant over a four-day interval. Data: daily TAIEX closes, fit 2 June 2015 to 19 July 2016, forecast 20 July to 30 August 2016.

## q38 · method · p04

**Q.** What test problem and step sizes did Workineh et al. use to compare Euler and fourth-order Runge-Kutta, and which was more accurate?

**A.** y'' - 3y' + 2y = 0 with y(0) = -1, y'(0) = 0 on 0 <= x <= 1 (exact solution e^(2x) - 2e^x), step sizes 0.1, 0.05, 0.025 and 0.0125, computed in MATLAB. RK4 was more accurate, stable and convergent; they give local truncation error as O(h^2) for Euler and O(h^5) for RK4.

## q39 · method · p06

**Q.** What is the key modelling change in Eissa and Elsayed's stochastic pantograph model, and how is it solved?

**A.** A variable (pantograph) delay qt replaces the constant delay of stochastic delay models, with volatility g(x) = sigma*x^(alpha-1). Solved with the stochastic theta Milstein method plus Monte Carlo (400 sample paths), which preserves positivity. Compared with Black-Scholes and constant-delay SDDE on Aaron's (AAN), Alcoa (AA), Tesco (TSCO.L) and Barclays (BCS) over 50 and 150 days, in Python 3.7.

## q40 · discrepancy · p06

**Q.** What MAPE or RMSE did Eissa and Elsayed report for their stochastic pantograph model?

**A.** None. The comparison with Black-Scholes and the constant-delay model is visual (sample-path means against real prices); no numerical error metric is reported.

*Note:* Hallucination trap: any number given as an error metric is wrong.

## q41 · comparison · p16

**Q.** In Mihova et al., how did the modified ODE approach compare with ARIMA when validated during the crisis period?

**A.** The modified ODE consistently had smaller relative errors: maximum relative error below 0.6% in 2020 and below 2.6% in 2022; errors were larger in 2022 for both methods. Four Bulgarian companies (Allterco, Elana AgroCredit, Speedy, Chimimport); models developed on 1 June to 29 October 2020 and validated on 1 June to 28 October 2022.

## q42 · numeric · p16

**Q.** In Mihova et al., how did the ARIMA-based and ODE-based Markowitz portfolios do at predicting the trend for 31 October 2022?

**A.** The ARIMA portfolio predicted the trend correctly for 3 of 4 instruments (73.82% of exposure). The ODE portfolio got only one right, EAC, but it made up 74.41% of the portfolio, so returns were still significantly positive.

## q43 · method · p15

**Q.** What does the MFR-GEP algorithm add to standard gene expression programming, and how is the resulting ODE solved?

**A.** Multi-factor regularisation: normalized trading volume enters the fitness function as a regular term, weighted by an information-content coefficient and mapped through a fuzzy-rough-set subunit function K(v). The resulting higher-order ODE is reduced to a first-order system and solved with fourth-order Runge-Kutta. 10 stocks (e.g. YTO Express, Kunlun Wanwei), 118 training and 61 test points, 5-day forecasts, evaluated by mean relative error; the neural network beat it only on Taiyuan Heavy Industry.

## q44 · factoid · p09

**Q.** What is the main finding of Xie, Xia and Gao's recursive dynamic asset pricing model?

**A.** Investor sentiment is the key factor behind sustained stock price fluctuations. The model combines sentiment investors, information traders and noise traders and partly explains overreaction, bubbles and crises. Motivating example: the Shanghai Composite rose from 998.23 (6 June 2005) to 6124.04 (16 October 2007), then fell to 2470.07 (11 August 2008).

## q45 · numeric · p11

**Q.** Which UML diagram was most common in the UML systematic review, and for what purpose were UML diagrams mostly used?

**A.** Class diagrams (71 publications, 26.3%). Design and modelling was the main purpose (68.7%), then testing (18%) and implementation (13.3%). 128 of 247 retrieved papers, 2000-2019, Google Scholar.

## q46 · aggregation · p01, p02, p13, p03, p05

**Q.** Which papers model stock prices with a logistic-type differential equation whose parameters change over time?

**A.** Li et al. (TAIEX) introduced time-varying coefficients in logistic and Newton-cooling forms; the banking study and the IDX Composite study compute alpha_n and beta_n from consecutive prices; Chandra et al. update the growth rate each step in a carrying-capacity form; the 2026 velocity-acceleration paper extends this to a second-order model with dynamic a(t) and b(t).

*Note:* p12 (Lotka-Volterra) uses fixed parameters and should not be listed. Note p03 describes p02/p13 as using constant parameters, while p02/p13 themselves describe per-step alpha_n, beta_n.

## q47 · aggregation · p01

**Q.** Which paper in the corpus models stock prices with Newton's law of cooling?

**A.** Li, Chiang-Lin and Lee (2018), whose dynamic models take either a logistic-growth or a Newton's-cooling form, building on Chen et al.'s mean-reversion models.

*Note:* Directly relevant to the skripsi model.

## q48 · aggregation · p07, p14, p18

**Q.** Which papers use the Caldara and Iacoviello geopolitical risk index, and what does each conclude about oil?

**A.** Truong et al.: GPR raises oil price volatility in the short and long run. Smales: GPR raises oil volatility and oil returns, with spillovers from oil to stocks. Ozdemir et al.: GPR positively affects WTI and Brent, and negative shocks raise volatility more than positive ones.

## q49 · comparison · p07

**Q.** Is there consensus on whether geopolitical risk increases or decreases oil price volatility?

**A.** No. Truong et al. review studies finding a positive effect (Bouoiyour 2019, Li 2020, Lee 2021, Qian 2022, Wu 2023) and others finding a negative effect (Plakandaras 2019, Qin 2020, Cunado 2020, Zhang 2022), and attribute the inconsistency partly to reliance on OLS and VAR methods.

## q50 · aggregation · p02, p13, p03, p05, p04, p15, p12, p17, p06, p01

**Q.** Which numerical solution methods appear across the corpus, and in which papers?

**A.** Forward-difference discretisation of the logistic ODE (banking, IDX Composite and private-company studies) and finite differences for the second-order model (velocity-acceleration); Euler vs RK4 (Workineh et al.); RK4 inside MFR-GEP (Zhang et al.) and as the starter for Adams-Bashforth-Moulton (Lotka-Volterra study); Euler-Maruyama for Heston (Oyuna and Liu); stochastic theta Milstein with Monte Carlo (Eissa and Elsayed); dynamic integration (Li et al.).

## q51 · comparison · p02, p13, p03

**Q.** How did the logistic models' forecast accuracy behave around the onset of COVID-19?

**A.** Error spiked in all three studies: banking MAPE rose by more than 20% from December 2019 to May 2020; IDX Composite MAPE and RMSPE exceeded 10% from December 2019 to March 2020; private-company MAPE rose above 10% from January to April 2020 for all firms except UNVR, with ASII above 30%.

## q52 · comparison · p13, p02, p05

**Q.** In the Noviantri-group papers, does a shorter forecast horizon always give lower error?

**A.** Generally, but not always. Averages rise with horizon (IDX Composite experiments 4.672/5.489/6.229%; banking 3.592/4.095/4.394%; velocity-acceleration shorter windows lower). Exceptions: IDX Composite single start date, 45-day 4.204% < 30-day 4.332%; banking BBNI 45-day 4.496% < 30-day 4.706%.

*Note:* Nuance test: a correct answer must not say 'always'.

## q53 · unanswerable · —

**Q.** What MAPE did the adaptive Newton's-cooling model achieve on MEDC.JK or PGAS.JK?

**A.** Not covered by the corpus.

*Note:* This is the skripsi itself, which is not in the corpus. If you add the skripsi as p00, rewrite this as an answerable question.

## q54 · unanswerable · —

**Q.** How do transformer models such as Informer or TimesNet compare with LSTM on Indonesian stocks?

**A.** Not covered by the corpus.

## q55 · unanswerable · —

**Q.** What is BBCA's share price today?

**A.** Not covered by the corpus; it contains no live market data.

# Equation questions (added 2026-09-24, reviewed)

## q56 · equation · p06

**Q:** What stochastic differential equation defines the SP-SPDE stock price model in the pantograph-delay paper?

**A:** dS(t) = r(S(qt))S(t)dt + g(S(qt))S(t)dW(t) (Eq. 3): drift and volatility depend on the price at the proportionally delayed time qt, where q = 1 − δ ∈ (0,1) comes from the delay function h(t) = δt.

`must_include`: ['qt', 'dW']  
_Source: p06 Section 2, Eq. (3) and the text defining h(t) = δt, q = 1 − δ._

## q57 · equation · p06

**Q:** How does the pantograph (SP-SPDE) stock model differ from Black–Scholes and from the constant-delay SDDE?

**A:** Black–Scholes (Eq. 1): dS = rS dt + σS dW with constant r and σ. Constant-delay SDDE (Eq. 2): r and g depend on S(t − τ). SP-SPDE (Eq. 3): r and g depend on S(qt), a variable (proportional) delay that grows with t.

`must_include`: ['qt']  
_Comparison inside one paper; tests that Eqs. 1-3 are retrieved together._

## q58 · equation · p05

**Q:** What differential equation defines the dynamic logistic velocity–acceleration model?

**A:** dv/dt = a(t)v(t)^2 + b(t)v(t) (Eq. 1) with v(t) = dS/dt; equivalently the second-order ODE d²S/dt² = a(t)(dS/dt)^2 + b(t)dS/dt (Eq. 2). a(t) and b(t) are dynamic parameters updated at each observation.

`must_include`: ['a(t)', 'b(t)']  
_p05 Section III._

## q59 · equation · p13

**Q:** How is the logistic ODE discretised in the IDX Composite dynamic logistic study, and how are the initial parameters obtained?

**A:** Forward difference: (S_{n+1} − S_n)/Δt = α_n S_n^2 + β_n S_n (Eq. 2). With the first three prices, α_0 = α_1 = α_2 = (S_1^2 − S_0 S_2)/(S_0^2 S_1 − S_1^2) (Eq. 3) and β_0 = β_1 = β_2 = (S_0^2 (S_2 − S_1) − S_1^3 + S_0 S_1^2)/(S_0^2 S_1 − S_1^2) (Eq. 4); later α_n, β_n are updated from the most recent prices.

`must_include`: ['forward difference']  
_p13 puts the model derivation under its INTRODUCTION heading, so expected_sections lists both._

## q60 · equation · p12

**Q:** Which system of equations models the two interacting stock prices in the Lotka–Volterra paper, and what do its coefficients mean?

**A:** dx/dt = x(a_1 − b_11 x − b_12 y) and dy/dt = x(a_2 − b_21 x − b_22 y) as printed (Eqs. 1-2). a_1, a_2 are intrinsic growth rates; b_11, b_22 intraspecific competition rates; b_12, b_21 competition rates between the two stocks. The pair is chosen by minimum standard deviation; the system is solved numerically with the Adams–Bashforth–Moulton predictor-corrector.

`must_include`: ['intraspecific']  
_Eq. (2) is printed with prefactor x, not y (see q61). p12 places the model under INTRODUCTION._

## q61 · discrepancy · p12

**Q:** Is there anything unusual about the second Lotka–Volterra equation as printed in the interacting-stock-prices paper?

**A:** Yes. Eq. (2) is printed as dy/dt = x(a_2 − b_21 x − b_22 y); the standard competitive Lotka–Volterra form, and symmetry with Eq. (1), would have prefactor y. It looks like a typo in the paper.

`must_include`: — (judge only)  
_Judge-only (no reliable must_include). Verified against the PDF text layer: the paper itself prints x(...); not a parsing error._

## q62 · equation · p17

**Q:** What price and volatility dynamics does the Heston model in the crude-oil volatility paper assume?

**A:** Price: dX(t) = X(t)(μ dt + √V(t) dW_1(t)) (Eq. 1). Variance: dV(t) = β(θ − V(t))dt + σ√V(t) dW_2(t) (Eq. 4), where β is the reversion rate, θ the long-run variance and σ the volatility of volatility; 2βθ > σ² keeps V(t) positive. Simulated with Euler–Maruyama.

`must_include`: ['long-run']  
_p17 'The Heston Model' subsection._

## q63 · equation · p10

**Q:** How is daily realized volatility constructed in the U.S. oil and gas firms volatility study?

**A:** RV²_{t,5m} is the sum of squared 5-minute returns (Eq. 1); the estimate averages RV² at 5, 10, 15 and 30 minutes (Eq. 2); overnight variation OV_t² = (100(O_t − C_{t−1})/C_{t−1})² is added; RV_t = ln(RV_t² + OV_t²) (Eq. 3).

`must_include`: ['30', 'overnight']  
_p10 Section 2.2.1-2.2.2._

## q64 · equation · p01

**Q:** What are the four dynamic models in the Taiwan stock index paper, and what equation defines each?

**A:** Model A (dynamic logistic): dS/dt = α_1(t)S² + β_1(t)S (Eq. 2). Model B (dynamic transformed logistic): d²S/dt² = α_2(t)(dS/dt)² + β_2(t)dS/dt (Eq. 9). Model C (dynamic relative growth rate transformed logistic): dδ/dt = α_3(t)δ² + β_3(t)δ (Eq. 12). Model D (dynamic general Newton model): dS/dt = α_4(t)[S(t) − A(t)] (Eq. 15). Coefficients are treated as constant over very short intervals.

`must_include`: ['Newton', 'logistic']  
_p01 Section 3._

## q65 · equation · p03

**Q:** In the discrete-time logistic model for Indonesian private companies, how do α(t), β(t) and the carrying capacity relate, and what is the discrete model?

**A:** dS/dt = β(t)(1 − S/K)S (Eq. 1) is rewritten as dS/dt = β(t)S + α(t)S² with α(t) = −β(t)/K; the forward-difference discrete model is (S_{n+1} − S_n)/Δt = β_n S_n + α_n S_n² (Eq. 3).

`must_include`: ['K']  
_p03 Section II. Overlaps q59 (same research group): a good cross-paper confusion test._

## q66 · unanswerable · —

**Q:** What closed-form analytical solution does the corpus give for the Lotka–Volterra stock price system?

**A:** The corpus doesn't cover this. The Lotka–Volterra paper solves the system numerically (Adams–Bashforth–Moulton predictor-corrector with Runge–Kutta starting values) and gives no closed-form solution.

`must_include`: — (judge only)  
_Refusal test near an equation topic. Should not produce a derived formula._
