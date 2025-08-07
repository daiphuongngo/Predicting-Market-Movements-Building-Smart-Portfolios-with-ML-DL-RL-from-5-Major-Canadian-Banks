# Predicting-Market-Movements-Building-Smart-Portfolios-with-ML-DL-RL-from-5-Major-Canadian-Banks

## **Master of Liberal Arts (ALM), Data Science**

## CSCI S-278 Applied Quantitative Finance in Machine Learning

## Name: **Dai Phuong Ngo (Liam)**

Manager, Data Analyst - Canadian Corporate Tax, Tax Technology - Asset Management - KPMG Canada

## Professor: **MarcAntonio Awada, PhD**

Head of Research and Data Science, Digital Data Design Institute, Harvard University

### **1. Project Objective and Motivation**

My project **Momentum-Based Prediction and Risk Modeling of Five Big Canadian Bank Stocks Using Machine Learning** for the course **CSCI S-278 Applied Quantiative Finance in Machine Learning** for the Master, Data Science at Harvard Extension School, Harvard University focuses on evaluating the return potential and risk-adjusted performance of the **five major Canadian banks**: **Royal Bank of Canada (RY.TO), Toronto-Dominion Bank (TD.TO), Bank of Nova Scotia (BNS.TO), Bank of Montreal (BMO.TO), and Canadian Imperial Bank of Commerce (CM.TO)**. These banks are systematically important to Canada’s financial system, represent diverse operational strategies and are well-traded equities with ample historical data. Their economic significance makes them ideal for me studying equity momentum, factor modeling and machine learning-based prediction.

### **2. Dataset and Feature Engineering**

I used weekly price and return data from **January 2020 to July 2024** througout COVID-19 pandemic and post pandemic, collected using the `yfinance` Python package. For each bank, I engineered features such as:

* **5-week average return**
* **10-week rolling volatility**
* **10-week momentum**

I selected **S\&P 500 (SPY)** as the exemplary benchmark market index and the **13-week U.S. Treasury bill yield (IRX)** was used as the **risk-free rate proxy**. I computed all returns as percentage changes and aligned to consistent them weekly intervals.

### **3. Labeling Strategy for Classification**

I created binary classification labels based on the **weekly excess return over the S\&P 500**:

* **Outperformer** if the bank’s return exceeded the S\&P 500 by **0.3% or more**
* **Underperformer** if the bank underperformed the S\&P 500 by **1.0% or more**

These thresholds helped me focus the model on meaningful momentum signals and reduce noise.

### **4. Models and Architectures**

I began with **Random Forest classifiers** and expanded to an **ensemble learning architecture** that includes:

* **Random Forest**
* **Logistic Regression**
* **Support Vector Classifier (SVC)**

I then applied **voting mechanism** where a stock was classified as outperforming or underperforming if at least one model predicted positively (`>= 1` vote threshold), making the ensemble less conservative and increasing signal sensitivity.

### **5. Evaluation Metrics and Confusion Matrices**

I measured different models' performance using:

* Accuracy
* Precision
* Confusion matrix

My results later showed:

* Outperformer classifier: 42% precision, confusion matrix indicating solid true positive rate
* Underperformer classifier: 29% precision, with moderate ability to detect weak performers

### **6. Strategy Backtest Results**

I used my model predictions for these as follows:

* The Outperforming portfolio achieved a cumulative return of 10.34×, versus 4.10× for the S\&P 500, reflecting +152% relative outperformance
* The Underperforming portfolio returned 0.39×, underperforming the S\&P’s 0.68×, validating model weakness in short signals

These results suggest the long momentum signal is more reliable under this framework.

### **7. Factor Modeling with CAPM**

To evaluate market sensitivity, I implemented **CAPM regression** using OLS from `statsmodels`:

* Regressed each bank’s excess return against market excess return
* Extracted alpha (intercept), beta (market sensitivity), and R² (fit quality)

These regressions confirmed varied exposure levels of each bank to broad market movements.

### **8. Risk-Adjusted Performance Metrics**

I calculated:

* **Sharpe Ratio** – measuring return per unit of volatility
* **Maximum Drawdown (MDD)** – indicating downside risk
* **Calmar Ratio** – balancing return vs. worst drawdown

These metrics helped me contextualize raw returns by highlighting volatility and capital protection.

### **9. Visualizations and Diagrams**

I gnerated plots to include:

* **Cumulative return curves** for predicted vs. market portfolios
* **Confusion matrices** to assess classification spread
* **Feature importance** rankings from Random Forest
* **CAPM regression line fits** for selected banks

These visuals enhanced interpretability and demonstrated the model's real-world relevance.

### **10. Course Concepts Applied**

My project incorporates multiple concepts from the course CSCI S-278 **Applied Quantitative Finance and Machine Learning** course:

* **Momentum and volatility-based features**
* **Binary classification and ensemble learning**
* **CAPM factor modeling**
* **Backtesting strategy evaluation**
* **Risk-adjusted return metrics**
* **Confusion matrix and precision analysis**
* **Deep Reinforcement Learning application**

My workflow, therefore, reflects a practical investment modeling pipeline from feature generation to portfolio return analysis.


## Price Trend Analysis (2014–2024)

<img width="1389" height="590" alt="Notebook 1 - plot 1" src="https://github.com/user-attachments/assets/0168a7a1-c601-4414-ad19-d29ad256ff80" />

<img width="1389" height="590" alt="Notebook 1 - plot 2" src="https://github.com/user-attachments/assets/615cc3c4-be53-40a2-834a-7886be6a95b3" />

Based on my above reports, **Royal Bank of Canada (RY)** illustrates the strongest price growth, rising from ~$45 to over $170, clearly outperforming its peers. Meanwhile, the reports demonstrate that **Bank of Montreal (BMO)** also shows significant appreciation, especially post-2020, with prices exceeding $140 in 2024. Next, **TD Bank** follows closely behind BMO in terms of price growth, maintaining a relatively consistent upward trajectory. Lastly, I observe that **Bank of Nova Scotia (BNS)** and **CIBC (CM)** show more volatile and flat in price performance over the decade, with BNS clearly underperforming. Therefore, my preliminary analysis signals that the COVID-19 crash in early 2020 is evident across all five biggest Canadian banks, followed by a strong rebound—especially for RY and BMO.

Now I will analyze their Daily Return Behaviors. The Daily Return plots show all banks experienced sharp spikes in volatility during market shocks, notably, March 2020 COVID crash. Their volatility has remained relatively stable post-2021, but occasional high-return days with positive and negative figures still appear. I observe that the return series for all 5 banks are highly correlated, reflecting the interconnectedness of Canadian banking performance with macroeconomic trends.

---

### Summary Statistics Interpretation

| Bank | Mean Return | Std Dev   | Max Daily Gain | Max Daily Loss |
| ---- | ----------- | --------- | -------------- | -------------- |
| RY   | **0.0534%** | 1.07%     | 14.90%         | -10.54%        |
| CM   | 0.0527%     | 1.21%     | **18.96%**     | -17.13%        |
| BMO  | 0.0494%     | **1.26%** | 16.98%         | -16.41%        |
| TD   | 0.0385%     | 1.18%     | 17.88%         | -12.33%        |
| BNS  | 0.0325%     | 1.18%     | 16.84%         | -13.37%        |

Based on the given summary statistics, I can see that **RY** leads in average daily return and shows **lower volatility**, supporting its strong price performance. On the other hand, **CM** and **BMO** exhibit higher volatility (standard deviation), leading to larger swings in both directions. Meanwhile, it seems to me that **BNS** is the most conservative performer, with the lowest average return and substantial drawdowns, making it less attractive from a growth perspective. Therefore, all banks demonstrate fat-tailed behavior with high max gains and losses, which is consistent with financial time series exhibiting heavy tails and leptokurtosis.

The results clearly shows to me that there is Risk-Return Tradeoff. **RY** offers the best risk-adjusted return, with the highest average return and lowest volatility among the five banks. Besides, **CM** offers high returns but at the cost of higher volatility, suggesting greater risk for investors, especially new those with limited budget and investment frequency. **BNS** seems less rewarding despite having comparable volatility to peers-raising questions about its growth strategy or exposure to macro shocks.

In summary, **RY and BMO** shows that they are likely more attractive for long-term investors seeking growth and stability. **TD** is second best that provides a moderate return-risk profile. And **CM** could be suitable for risk-tolerant investors like me seeking higher upside. Unfortunately, **BNS** may require more of my caution due to persistent underperformance and volatile downside risk.

### Histogram of Daily Returns

<img width="1389" height="985" alt="Notebook 1 - plot 3" src="https://github.com/user-attachments/assets/224f1f96-bd55-4129-893f-2d061fdaa38b" />

My histograms of daily returns above show that all banks have bell-shaped distributions, centered around zero. They exhibit slight left-skewness, with more pronounced negative returns than positive extremes. **CM** shows the widest tail, reinforcing its higher downside risk from the summary stats. As observed from the plots, daily returns of Canadian banks are approximately normally distributed, validating my assumptions for basic modeling like CAPM and linear regression, though risk management must account for tail risk.

### Correlation Heatmap

<img width="672" height="590" alt="Notebook 1 - plot 4" src="https://github.com/user-attachments/assets/6ccb6074-10e5-4a1d-8a63-681ef3dba05b" />

All pairwise correlations are very high (ranging from 0.73 to 0.80). **BMO-CM (0.80)** and **BMO-BNS (0.79)** show the strongest relationships. **CM-TD (0.73)** is slightly weaker, possibly due to differences in risk exposures or operational scale. These insights indicate low diversification potential among these banks, including non-bank or global assets that may improve risk-adjusted returns.

### Scatter Matrix of Returns

<img width="1108" height="946" alt="Notebook 1 - plot 5" src="https://github.com/user-attachments/assets/b467d223-5749-4896-a09b-5f355343ea51" />

As I observe, all scatter plots reveal strong positive linear relationships between all pairs of banks. The density of points around the diagonal in each pairwise plot confirms high correlation. Also, even KDE plots on diagonals also support normal-like distribution. In short, I can extract an insight that Canadian banks move together in the market, suggesting exposure to common macroeconomic or industry-wide factors. Therefore, these insights are useful for my later portfolio diversification analysis.

### Boxplot of Daily Returns
<img width="989" height="590" alt="Notebook 1 - plot 6" src="https://github.com/user-attachments/assets/78a690da-48ec-4f0f-8f52-0fa8ae7c8d8a" />

Here is another outlook of analysis I figured out that boxplots confirm symmetry and reveal many outliers. RY and BMO have slightly wider interquartile ranges, reflecting more variation in their daily performance. Each bank’s distribution is tightly centered around zero, but the tails extend widely, consistent with fat-tailed distributions. The visual representation of these boxplots imposes a potential risk that despite low median returns, there are significant positive or negative deviations.

### Autocorrelation

<img width="990" height="390" alt="Notebook 1 - plot 7" src="https://github.com/user-attachments/assets/e9793ae1-fbd0-41f5-913c-5fb4eece2f03" />

As seen above, the ACF plots show significant autocorrelation at lag 0 as I expected, but very low autocorrelation at all other lags. Besides, most values beyond lag 1 fall within confidence intervals. This implies that daily returns are weakly autocorrelated or even nearly serially uncorrelated. Therefore, this supports the efficient market hypothesis (EMH), suggesting me to confirm that past daily returns do not predict future returns, at least in the short term.

## LSTM Forecasting Analysis for Five Canadian Banks (2014–2024)

<img width="831" height="374" alt="Notebook 2 - LSTM for BMO" src="https://github.com/user-attachments/assets/2a6a2ee7-249a-4b53-b7f9-4d99b4bb0654" />

<img width="822" height="374" alt="Notebook 2 - LSTM for CM" src="https://github.com/user-attachments/assets/3379b2a2-e832-4749-85f4-61a316f97fbc" />

<img width="831" height="374" alt="Notebook 2 - LSTM for RY" src="https://github.com/user-attachments/assets/d35cf46b-10ee-4785-91f0-fdd22ef6d795" />

<img width="822" height="374" alt="Notebook 2 - LSTM for TD" src="https://github.com/user-attachments/assets/3ada3896-077e-4522-b536-1576dcdc7df4" />

<img width="822" height="374" alt="Notebook 2 - LSTM for BNS" src="https://github.com/user-attachments/assets/d9c0cbd0-1b58-47d7-99dd-7a846b386d99" />

My general observations indicate that all banks exhibit strong predictive performance with LSTM, as the predicted lines closely follow the actual prices. These alignments suggest that the models effectively captured short-term temporal dependencies in stock price sequences where I trained the models using a rolling window of 60 days and tested on the final 20% of the data (of roughly 2 years).

Now I will analyze individual bank's performance:

#### Royal Bank of Canada (RY)

It shows the best performance as the model captured long-term upward momentum with very low deviation. As tracking the pattern, I found that both short-term corrections and rebounds are well modeled. This illustrates that RY’s stability and consistent trend made it ideal for LSTM forecasting.

#### Toronto-Dominion Bank (TD)

There is clearly high volatility as the model predicted direction well but underperformed in magnitude during rapid price jumps. There are also challenging zones where significant deviations occurred in later years when volatility increased.

#### Bank of Nova Scotia (BNS)

There is smooth tracking as predicted series aligns very closely with actual prices. But there is moderate variance that slight smoothing was noticed, but trend detection is reliable considerably.

#### Bank of Montreal (BMO)

There is high fit accuracy that one of the top performers alongside RY and BNS. I would consider it as strength that it captured both short-term and long-term trends with tight error margins.

#### Canadian Imperial Bank of Commerce (CM)

There is volatility capture here as the model reasonably followed the swings but under-reacted to sharp price changes. I also observe deviation as some short-term peaks were smoothed out due to the model's preference for trend continuity.

There are model strengths and limitations that I would need to put into considetaion.

Regarding strengths, when conducting non-linear modeling, I see that LSTM is adept at capturing lagged dependencies and nonlinear price movements. It has shownsequential memory which is ideal for time series with seasonality or repeating patterns like the banking cycles. Its predictions are relatively smooth which help remove noise while preserving trends.

However, there are certainly limitations. I observe over-smoothing as LSTM tends to flatten out rapid swings due to training loss minimization (MSE). When
data sensitivity, aspects of scaling, window size and test horizon significantly impact results. Therefore, LSTM does not offer interpretable coefficients like linear models, particularly CAPM.


In short, I can conclude that the LSTM models successfully demonstrated the ability to forecast stock price movement trends across five major Canadian banks. RY, BMO, and BNS had the most reliable forecasts with minimal tracking errors, making them strong candidates for future momentum or signal-based trading strategies. While TD and CM showed slightly less precision, the models still performed within acceptable error ranges.

There are some future work that I could include like comparing against ARIMA, GRU or Transformer models. Furthermore, I find that incorporating macroeconomic indicators like interest rates can increase explanability and model performance. I was recommended with testing on smaller data unit like minute or hourly data for short-term traders and consider adding rolling retraining and online learning capabilities.

### Analysis of Scaled Adjusted Close Price Distributions (MinMaxScaler)

<img width="1490" height="1025" alt="Notebook 2 - Histogram after Scaling" src="https://github.com/user-attachments/assets/926112b1-80bd-45cc-beef-b2417e1075f1" />

I applied MinMax scaling to normalize each bank's price data between 0 and 1, which is a standard preprocessing step for machine learning models like LSTM. This helps me to preserve relative volatility and trends while removing absolute price scale differences.

Here are some of my observations from the 5 banks:

#### BMO (Bank of Montreal)

BMO's distribution is multimodal, with multiple visible peaks, suggesting BMO experienced several distinct pricing regimes over the last 10 years. Its price range is fairly well spread, with substantial activity across the [0.1–0.9] interval. Therefore, this indicates non-stationarity and long-term market regime changes.

#### BNS (Bank of Nova Scotia)

BNS has a centered and moderately uniform distribution, peaking around 0.4-0.5, which suggests that BNS's price stayed within a tight range for most of the decade, but with a few movements toward higher and lower extremes. From another angle of view, the relatively balanced distribution makes BNS predictable for time series models.

#### CM (CIBC)

CIBC has a left-skewed distribution, with many prices clustered at lower values (under 0.3), which suggests longer periods of lower prices, or less recovery in later years. Besides, I found that the spread into the upper range was less frequent, indicating lower growth compared to peers.

#### RY (Royal Bank of Canada)

RBC also experiences multimodal, with visible spikes around 0.05, 0.25, 0.5, and 0.65. Like BMO, RBC's price progression was dynamic, capturing growth spurts and pullbacks over the years. Therefore, I have more confidence to sstate that RY's rich price dynamics likely helped LSTM models achieve strong forecasts.

#### TD (Toronto-Dominion Bank)

TD has a right-skewed distribution, with the majority of prices clustered in the 0.6-0.9 range. This suggests consistent upward growth or a longer period spent in higher price zones. On the other hand, unlike CM, TD spent less time in the low range, highlighting resilience and growth.

In short, I can affirm that multimodal or skewed distributions indicate market regime shifts or long-run trends that can be exploited by sequence models like LSTM. Besides, I can ore evenly distributed series (like BNS) suggest mean-reverting or stable behavior, favoring simpler models (e.g., ARIMA).
* These distributions also highlight the importance of rescaling for models sensitive to magnitude (especially RNNs).

---

I will put my summary into a table for easier tracking:

| Bank    | Skew/Modality      | Implication                                             |
| ------- | ------------------ | ------------------------------------------------------- |
| **BMO** | Multimodal         | Volatile, non-stationary pricing so my LSTM application and deployment can be effective       |
| **BNS** | Centered, moderate | Stable pricing observed which is good for mytrend or momentum strategies     |
| **CM**  | Left-skewed        | Often undervalued which I find it as potential value investing candidate |
| **RY**  | Multimodal         | Captures high growth and dips which is dynamic for modeling    |
| **TD**  | Right-skewed       | Sustained higher prices so I can see it as strong long-term signal       |

###  Analysis of Stationarity Test Results for 5 Canadian Banks (Adjusted Close Prices) with ADF and KPSS

Here are the 2 most popular stationarity tests I will implement:

* **ADF (Augmented Dickey-Fuller)**:

  * Null Hypothesis (H₀): Series has a unit root (non-stationary).
  * If **p-value < 0.05**, I reject H₀ and conclude **stationarity**.
* **KPSS (KwiatkowskiPhillipsSchmidtShin)**:

  * Null Hypothesis (H₀): Series is stationary.
  * If **p-value < 0.05**, I reject H₀ and conclude **non-stationarity**.
 
  As observed, I found that all 5 series fail the ADF test as ADF p-values are very high so I fail to reject H₀. Therefore, they are likely non-stationary.

Also, as all 5 series fail the KPSS test, shown by KPSS statistics that are far above critical values, with p-values < 0.01. So I reject H₀ as they are non-stationary.

Moreover, these Adjusted Close price series are trend-driven and non-stationary, as is common with raw stock prices. So I can affirm that both tests confirm this. I also need to proceed with modeling like ARIMA, VAR or deep learning later on so I have to double sure that stationarity must be imposed with other techniques like First-Differencing to transform prices into returns or price differences, or Log-differencing which is better if modeling percentage changes, or MinMaxScaling for common data standardization. Then I will rerun ADF and KPSS on the differenced or log-return series to check if stationarity is achieved.


### Stationarity Analysis of Scaled Adjusted Close Prices (2014-2024) after MinMaxScaling

As observed, all five banks' scaled price series are non-stationary. this is confirmed by both ADF (high p-values) and KPSS (low p-values). Therefore, the ADF test fails to reject the null hypothesis of non-stationarity. Also, the KPSS test rejects the null of stationarity, reinforcing my conclusion.

However, I find that MinMaxScaler does not induce stationarity. Scaling adjusts the range (0 to 1) but does not remove trends or unit roots. As per my visual inspection of the price trends, it previously confirmed structural growth patterns (especially in RY, BMO, TD), contributing to non-stationarity.

Hence, instead, Log transformations or differencing may be required as the data is non-stationary, techniques like First-order differencing, Log-differencing, Detrending with moving averages or regression residuals should be explored prior to training models that assume stationarity, such as ARIMA.

Another point to mention that LSTM handles non-stationary data well. It is basically Unlike statistical models, LSTM-based neural nets can capture long-term dependencies and trends even when the series is non-stationary, which I find that it aligns with the accurate forecasts observed in earlier LSTM plots.

In short, my analysis confirms that the original time series are trend-dominated and non-stationary, even after scaling. If you're using traditional statistical models (ARIMA, GARCH), consider transformation and differencing. However, for deep learning models like LSTM, retaining the original structure post-scaling is acceptable, and can even be beneficial for my later sequence modeling, if applicable.

### Seasonal Decompose


<img width="1189" height="788" alt="Notebook 2 - Seasonal Decompose - BMO" src="https://github.com/user-attachments/assets/459713bb-75af-420b-8c0a-9d92f6400d3d" />

<img width="1189" height="788" alt="Notebook 2 - Seasonal Decompose - BMS" src="https://github.com/user-attachments/assets/373c018a-d730-44dd-8cc1-a46793ef160e" />

<img width="1189" height="789" alt="Notebook 2 - Seasonal Decompose - CM" src="https://github.com/user-attachments/assets/e81c9bd7-4e65-4c69-a25b-850b64db6be5" />

<img width="1189" height="789" alt="Notebook 2 - Seasonal Decompose - RY" src="https://github.com/user-attachments/assets/8a49a1c6-f6be-4745-b522-8f2701769779" />

<img width="1189" height="789" alt="Notebook 2 - Seasonal Decompose - TD" src="https://github.com/user-attachments/assets/902a11f9-7ccf-44a0-9642-071b79baaf7f" />

Regarding the Multiplicative Decomposition, in a multiplicative model:

$$
\text{Observed} = \text{Trend} \times \text{Seasonal} \times \text{Residual}
$$

This means that trend reflects long-term growth while seasonality represents periodic fluctuations like monthly patterns. And residual captures random fluctuations not explained by trend or seasonality.

I will explain the Trend Analysis (2014-2024) as follows. RBC and BMO have the strongest upward trends, reflecting long-term price appreciation and likely investor confidence. TD and CM show moderate trend growth, especially after 2020. However, BNS displays a flatter trend, suggesting relatively slower long-term performance. Therefore, I conclude that banks like RY and BMO exhibit robust structural momentum, which aligns with their strong market capitalization and earnings history.

Now will analyze their Seasonal Component. All five banks exhibit strong and consistent seasonal patterns with a 12-month cycle. Their seasonality amplitude is narrow (between ~0.98 to ~1.02), meaning seasonal fluctuations are modest but systematic. I also observe that peaks often occur late in Q1 or early Q2, and troughs appear in late Q3 or early Q4, likely reflecting fiscal year effects, dividend payout cycles, institutional portfolio rebalancing and earnings season dynamics. These are proven as TD, RY, and BNS have sharp seasonal peaks around March-May, possibly tied to post-earnings or dividend payout strength.

Regarding the Residual Component, the residuals, which are random error not explained by trend/seasonal, are fairly small and stable esides, most residual points hover around 1.00, indicating that random fluctuations are minor compared to trend/seasonality. But there is one exception that some larger deviations in 2020-2021, likely due to COVID-19 market shocks.

At this point, I can come up with some Strategic Implications. For forecasting or portfolio modeling, I can use seasonal adjustment when forecasting short-term returns. BMO and RY may serve as my anchor investments for long-term portfolios. I will also consider de-seasonalizing data if using regression or machine learning for monthly predictions. For trading strategies, I can see that mean-reversion or pairs trading may benefit from knowing seasonal peaks and troughs. I will also have to asses the timing entries and exits I will buy in trough months and sell in peak months, as these actions could improve my tactical asset allocation.

### Fractal Differencing


<img width="989" height="390" alt="Notebook 2 - Factionally DIfferenced Series - BMO" src="https://github.com/user-attachments/assets/40ed002e-5481-4879-bc1b-01d1aeaa2f63" />


<img width="990" height="390" alt="Notebook 2 - Factionally DIfferenced Series - BNS" src="https://github.com/user-attachments/assets/f7ec1eed-8510-4ad6-8b9d-7f8d5d0a320d" />


<img width="990" height="390" alt="Notebook 2 - Factionally DIfferenced Series - CM" src="https://github.com/user-attachments/assets/284d4368-00a6-4db1-b21d-a0a4b353d0da" />


<img width="989" height="390" alt="Notebook 2 - Factionally DIfferenced Series - RY" src="https://github.com/user-attachments/assets/920b9e72-a88c-4c89-9713-8c9a0ad780c6" />


<img width="990" height="390" alt="Notebook 2 - Factionally DIfferenced Series - TD" src="https://github.com/user-attachments/assets/a69285f2-a3dc-46a2-a70a-d64524a98e94" />

```
 BMO with Fractional Differencing (d=0.92) 
ADF Statistic: -2.6826
p-value: 0.0771
Critical Values:
	1%: -3.433
	5%: -2.863
	10%: -2.567


 BNS with Fractional Differencing (d=0.92) 
ADF Statistic: -3.7046
p-value: 0.0040
Critical Values:
	1%: -3.433
	5%: -2.863
	10%: -2.567


 CM with Fractional Differencing (d=0.92) 
ADF Statistic: -2.2433
p-value: 0.1909
Critical Values:
	1%: -3.433
	5%: -2.863
	10%: -2.567


 RY with Fractional Differencing (d=0.92) 
ADF Statistic: -1.4752
p-value: 0.5458
Critical Values:
	1%: -3.433
	5%: -2.863
	10%: -2.567


 TD with Fractional Differencing (d=0.92) 
ADF Statistic: -2.8626
p-value: 0.0499
Critical Values:
	1%: -3.433
	5%: -2.863
	10%: -2.567

```

I applied fractional differencing as I aim to transform non-stationary time series into stationary ones while retaining long-term memory, which is crucial for models like ARIMA or machine learning pipelines that require stationarity. Standard differencing like `d = 1` removes valuable long-term memory. On the other hand, fractional differencing with `d < 1` preserves long-range dependencies, important for financial forecasting.


I can observe BNS is clearly stationary after fractional differencing (p < 0.01). Meanwhile, TD is marginally stationary (p ≈ 0.05), just hitting the 5% critical value threshold. Also, BMO is borderline non-stationary (p = 0.077), just outside 5% significance. Lastly, CM and RY remain non-stationary post-differencing.

Based on the plots, each displays the fractionally differenced series using a 3-term approximation and `d = 0.92`. I can notice that variance and volatility increase gradually after 2020, notably during and after the COVID-19 crisis.Thankfully, differenced series maintain the overall shape of volatility spikes but reduce longer-term trends. Fortunately enough, the differenced series have a more mean-reverting behavior, indicating improved stationarity in some banks.

I also see that, BNS is now suitable for ARIMA or GARCH modeling. TD and BMO might be acceptable depending on the tolerance level for p-value. However, RY and CM need further differencing or alternative transformation such as log-diff or power transforms.


In short, fractional differencing at `d = 0.92` proves effective in partially stabilizing financial time series. I can state that only BNS and marginally TD achieved clear statistical stationarity. Still, this technique helps me to balance the trade-off between stationarity and memory retention, making it well-suited for forecasting tasks in financial applications.

## Lasso Regression and Dendogram

### Analysis of Hierarchical Clustering for Feature Selection


<img width="998" height="990" alt="Notebook 3 - Dendogram" src="https://github.com/user-attachments/assets/58ff2850-8239-4584-84e9-6ae76c00e622" />


<img width="2490" height="989" alt="Notebook 3 - Dendogram 2" src="https://github.com/user-attachments/assets/4085bc26-e27c-4e36-84ac-3ba0c0c005fb" />

I used hierarchical clustering on standardized engineered features derived from Canadian bank OHLCV data to export a Clustermap (Heatmap with Dendrogram) to analyze Feature Correlation Structure:

* Heatmap Matrix: Shows the pairwise Pearson correlation between features.
* Hierarchical Clustering: Groups features with similar correlation profiles into clusters.
* Color Map (`Blues`): Darker blue = higher positive correlation.

There are highly correlated Cluster, including `f4`, `f6`, `f5`, `f8`, `f3` are tightly grouped and strongly correlated. These are all volume-based metrics:

    * `f4`: Volume change
    * `f5`: Volume change over 50 days
    * `f6`: Volume % change
    * `f8`: Volume vs. 200-day MA
    * `f3`: Log volume

 This suggests redundancy among these features. I can select only one or two to reduce multicollinearity.

Besides, there are moderately related pair like `f1` and `f2` (close/open return and open/previous close return) are somewhat isolated but closely clustered, indicating they share similar return behavior.

There are also unique features like `f9` (Close vs. 50-day EMA) and `f10` (z-score of Close) which form a separate sub-cluster, reflecting trend-following behaviors, and have low correlation with volume-related features.

Regarding the Dendrogram, my Hierarchical Tree of Features. My dendrogram visualizes how features cluster based on their correlation "distance". The shown shorter linkage (horizontal lines) imply more similar features. I can observe that `f4` and `f6` are the most similar pair, joined at a very short distance (volume difference and volume % change). `f5`, `f8`, and `f3` successively join that cluster, forming the volume cluster.  `f1`, `f2` and `f9` are less similar to volume features and remain in separate branches longer.
And lastly, `f10` joins later, suggesting it captures normalized trend signals orthogonal to raw volume/price data.


### Lasso Regression Pipeline for 5 Banks


<img width="590" height="390" alt="Notebook 3 - Lasso Regression RY alpha 0 01" src="https://github.com/user-attachments/assets/d0cf69a3-7536-4994-811f-6a2b872faf67" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression RY alpha 2" src="https://github.com/user-attachments/assets/20e207f4-1849-443c-b284-1a7b6bc0acee" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression RY alpha 0 08" src="https://github.com/user-attachments/assets/028a2975-dfc0-45f6-bc46-35982b378a84" />


<img width="589" height="390" alt="Notebook 3 - Lasso Regression TD alpha 0 01" src="https://github.com/user-attachments/assets/2411aae9-74e3-4021-a444-f0ff3f7fa854" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression TD alpha 2" src="https://github.com/user-attachments/assets/45c96eb9-04dc-4c5e-aa20-6fff75ef6fb5" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression TD alpha 0 08" src="https://github.com/user-attachments/assets/8e413643-40a1-476a-bbb5-cb33ffc97d24" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression BNS alpha 0 01" src="https://github.com/user-attachments/assets/8c187b1b-f49a-499a-8564-7ef3d88b186c" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression BNS alpha 2" src="https://github.com/user-attachments/assets/96cc82e8-f356-4215-b88f-9d54386c9767" />


<img width="589" height="390" alt="Notebook 3 - Lasso Regression BNS alpha 0 08" src="https://github.com/user-attachments/assets/c90c3107-e1d8-47b5-b0c4-e627cca97f63" />


<img width="589" height="390" alt="Notebook 3 - Lasso Regression BMO alpha 0 01" src="https://github.com/user-attachments/assets/83e17402-2177-412e-a63f-4d40e0c069e0" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression BMO alpha 2" src="https://github.com/user-attachments/assets/4990595e-2c9a-422b-b301-8ebe5baca9f2" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression BMO alpha 0 08" src="https://github.com/user-attachments/assets/f2450202-6976-492a-922f-f18e0236b800" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression CM alpha 0 01" src="https://github.com/user-attachments/assets/4ed9db43-92bf-4854-a88e-290048d9fd52" />


<img width="590" height="390" alt="Notebook 3 - Lasso Regression CM alpha 2" src="https://github.com/user-attachments/assets/a24408d7-c387-4ab6-9430-7932286a83bc" />


<img width="589" height="390" alt="Notebook 3 - Lasso Regression CM alpha 0 08" src="https://github.com/user-attachments/assets/38f75e6c-fd99-4bed-be83-cfebcc811237" />

The `f10` disappear for α=0.08 could be because for all 5 Canadian banks, the top coefficient `f10` (Z-score of close) appears as the dominant predictor when `y = Close` and α is low (0.01) or moderate (2). However, when predicting percentage change (`y = pct_change()`) with α = 0.08, `f10` has no significant influence, and all coefficients are shrunk toward zero.

By looking at the plots, at α = 0.01 or 2 with Close, there are xcellent linear patterns, suggesting `f10` is linearly related to `Close`. I can observe that all predictions are highly accurate (tight diagonal scatter). However, for α = 0.08 with `pct_change()`, there are wide scatter so Lasso is unable to find strong predictors. Feature importances nearly vanish, confirming the poor correlation between engineered features and next-day returns.

### PCA on Financials


<img width="790" height="490" alt="Notebook 3 - PCA on Financials" src="https://github.com/user-attachments/assets/e56a3175-ae3d-4f6c-af36-d6a613059f1c" />

Here I performed Principal Component Analysis (PCA) on normalized financial data of 5 Canadian banks (RY, TD, BNS, BMO, CM), using key financial features:

* Total Revenue
* Net Income
* EBITDA
* Interest Expense

Due to availability, the final PCA input had:

* 20 observations (rows) — 4 years × 5 banks
* 3 features (columns) — from `safe_cols` that were present for each bank


The curve shows:

* PC1 explains roughly 63% of the variance
* PC1 + PC2 explains more than 95%
* PC1 + PC2 + PC3 explains nearly 100%

I need only 2 components to retain at least 95% of the total variance in my dataset. With Dimensionality Reduction, I can reduce my 3D financial feature space to just 2 components without losing significant information. The model simplification is obvious as models using these 2 principal components can be more stable and generalize better. I can see that the first principal component likely reflects general scale of the bank (total revenue/net income dominance), while the second may capture efficiency or leverage nuances.

### Eigen-Portfolio Weights from PCA for the selected principal component


<img width="1589" height="590" alt="Notebook - Extract and visualize eigen-portfolio weights (PCA component loadings) for the selected principle component" src="https://github.com/user-attachments/assets/ff77c4df-256e-4ca7-9258-45bf3601d280" />

Here I perform PCA on the standardized financials of 5 Canadian banks using 3 features:

* Total Revenue
* Net Income
* Interest Expense

PC1 is the linear combination of the input features that explains the largest variance across banks' financials. It is often interpreted as the "general factor" of financial strength, size, or operating scale. I can think of PC1 as an "eigen-portfolio" where feature weights is asset allocations, banks are portfolios, and PC1 score for each bank is portfolio return. In other words, banks with high Total Revenue and Net Income will have higher PC1 scores.
PC1 acts as a synthetic factor capturing banks’ size-adjusted operating performance. I can analyze the feature importance as follows:

* Total Revenue (41.7%) dominates so it drives the most variation among the banks. Bigger banks with higher revenue pull PC1 scores higher.
* Net Income (33.4%) closely follows, representing profitability, it's also key in differentiating performance.
* Interest Expense (24.8%) is the least influential but still significant so it may represent differences in leverage, cost of debt or capital structure.

My bar chart clearly shows the relative influence of each feature on PC1. Total Revenue is the most influential, indicating that scale dominates variation across banks, which is a common pattern in the financial sector. PC1 explains \~63% of total variance (from earlier PCA). It is heavily driven by Total Revenue and Net Income, such as, size and profitability. Besides, Interest Expense plays a secondary role, variation in debt structure is less dominant. This eigen-portfolio insight helps interpret the latent structure in my bank financials and is useful for my future clustering, anomaly detection or factor investing.

### Ridge Regression on Lag Features for 5 Banks


<img width="990" height="390" alt="Notebook 3 - Ridge Regression for BMO" src="https://github.com/user-attachments/assets/205a569a-9c9d-4719-8a42-ba59920a186f" />


<img width="989" height="390" alt="Notebook 3 - Ridge Regression for BMS" src="https://github.com/user-attachments/assets/02e36f8e-661c-4de7-89d1-a352389e3922" />


<img width="989" height="390" alt="Notebook 3 - Ridge Regression for CM" src="https://github.com/user-attachments/assets/413cc442-6307-4b24-b3f5-d5ce64536abb" />


<img width="990" height="390" alt="Notebook 3 - Ridge Regression for RY" src="https://github.com/user-attachments/assets/c9c44267-315b-4c37-aa3f-300f42520467" />


<img width="990" height="390" alt="Notebook 3 - Ridge Regression for TD" src="https://github.com/user-attachments/assets/ebbb4e20-213c-4985-8c82-8cecb1fd8a4c" />

I used Ridge Regression with lagged price features (`lag_1` to `lag_5`) to predict the daily closing price of Canadian bank stocks: RY, TD, BNS, BMO, CM.

| Bank    | R² Score   | RMSE       | Interpretation                                                                   |
| ------- | ---------- | ---------- | -------------------------------------------------------------------------------- |
| BNS | 0.9820 | 0.6123 | Highest predictive power, very low error. Ridge regression fits BNS very well. |
| CM  | 0.9774     | 0.5953     | Very good fit, with low RMSE and strong correlation to lagged prices.          |
| BMO | 0.9733     | 1.2102     | High R², but larger RMSE due to more volatile price range.                     |
| RY  | 0.9729     | 1.1418     | Strong model fit, slightly less accurate than BNS/CM.                          |
| TD  | 0.9581     | 0.7933     | Good performance, but slightly weaker fit compared to others.                  |

As per my plots, all predicted lines follow actual prices closely, indicating Ridge is capturing the temporal dynamics well. For all banks, especially BNS, CM, and RY, the prediction curves nearly overlap with actual prices, confirming the high R² scores (~0.97-0.98). Some details are noteworthy are that TD had more short-term price fluctuations in recent years, which the model captures with slight lag, hence the lower R². Moreover, BMO and RY show minor deviations at peaks/valleys, suggesting nonlinearities that Ridge can't fully model.

I find that Ridge Regression worked well here as its Lagged Features (`lag_1` to `lag_5`) provide recent historical price context — a natural fit for financial time series. Furthermore, Ridge Regression (with L2 regularization) controls overfitting, especially when lagged features are correlated. AndTimeSeriesSplit preserves chronological integrity, avoiding data leakage.

In short, my Ridge Regression setup is effective, especially for BNS and CM. The strong R² values and tight fit in prediction plots validate my lag-based features and modeling strategy. Slightly weaker results for TD suggest potential value in modeling returns or volatility rather than raw prices.

## Single / Multiple Linear Regressions

#### TD


<img width="1189" height="590" alt="Notebook 4 - SLM, MLR for TD" src="https://github.com/user-attachments/assets/41cb8507-75c1-4013-b897-25de0a28c5f4" />


<img width="574" height="432" alt="Notebook 4 - SLM, MLR Residual for TD" src="https://github.com/user-attachments/assets/6c76670a-45ab-44a6-a3dc-b0cbff4a5c59" />

#### BNS


<img width="1189" height="590" alt="Notebook 4 - SLM, MLR for BNS" src="https://github.com/user-attachments/assets/67b619f9-ce72-4785-9c73-58e5244a25bf" />


<img width="565" height="432" alt="Notebook 4 - SLM, MLR Residual for BNS" src="https://github.com/user-attachments/assets/ab9a3bf8-828c-4a32-a1ac-2af980c2b89d" />

#### RY


<img width="1189" height="590" alt="Notebook 4 - SLM, MLR for RY" src="https://github.com/user-attachments/assets/54599498-760a-4159-a3af-a5f3bcbdedb3" />


<img width="574" height="432" alt="Notebook 4 - SLM, MLR Residual for RY" src="https://github.com/user-attachments/assets/7ab53fc0-1f6a-4646-8f12-a7b3265474ae" />

My goal of this modeling exercise is to understand and predict the adjusted closing price of TD Bank (TD.TO) using other major Canadian bank stocks as predictors, particularly RBC (RY.TO) in a simple linear regression (SLR), and RY, BNS, BMO, and CM in a multiple linear regression (MLR). My models aim to uncover the strength of comovement among Canada's Big Five banks, assess the explanatory power of peer institutions on TD's price, and evaluate whether a multivariate model improves predictive performance. Additionally, my analysis highlights violations of regression assumptions and informs the next steps for building more robust, time-aware financial models.

**SLR (Simple Linear Regression): TD ~ RY**

* R² = 0.632: 63.2% of TD's price variation is explained by RY's price.
* Significant coefficient:

  * Intercept: 32.07 (p < 0.001)
  * RY.TO coefficient: 0.336 (p < 0.001)

Based on the above results, the model suggests that for every 1 CAD increase in RY, TD increases by ~0.336 CAD. I find the fit moderate as over one-third of TD's movement is left unexplained. But there are issues to consider further. Durbin-Watson is 0.010 which is extremely low, causing severe positive autocorrelation. Residuals are not white noise, implying model misspecification.Jarque-Bera (JB) test is significant as residuals are not normally distributed. Also, SLR misses complexity by relying on a single predictor.

**MLR (Multiple Linear Regression): TD ~ RY + BNS + BMO + CM**

* R² = 0.956 → 95.6% of TD's price variance is explained by the four other banks.
* All predictors are statistically significant (p < 0.001):

  * Coefficients (impact on TD):

    * RY: +0.207
    * BNS: -1.236 (negative correlation)
    * BMO: +0.482
    * CM: +1.615

My model fits TD stock very well-almost all variation is explained. However, the sign of the BNS coefficient being negative may indicate multicollinearity between features or economic offsetting effects (e.g., TD gains when BNS drops). Therefore, I have to diagnose further if multicollinearity is true. The Condition Number is 1340, whcih shows potential multicollinearity very likely.
Also, Durbin-Watson is 0.034 which shows severe autocorrelation, a red flag in time series.

Looking at the chart of TD Actual vs. SLR & MLR Predictions, the SLR (TD ~ RY)tracks general trend but lags behind or overshoots in volatile regions. The performance underperforms in 2023-2025 range, which shows model rigidity. The MLR (TD ~ RY, BNS, BMO, CM) follows TD closely at the beginning, but diverges upward post-2023, which suggests overfitting or unaccounted nonstationarity in relationships. Another point to mention that the actual TD price flattens while MLR keeps rising.

Switching to the MLR Residual Plot (Predicted vs. Residuals), non-random scatter, suggesting strong structure and clustering. However, Residuals fan out and curve so heteroskedasticity and nonlinearity present. Also, several clusters show persistent bias which indicates systematic prediction error.

Therefore, the model iolates OLS assumptions asrrors are not homoscedastic (constant variance) and residuals show non-zero mean over intervals. They suggest that I need to use time series models (ARIMA, VAR) or nonlinear/regularized models (e.g., Ridge, Lasso, XGBoost) like I did earlier.

My objective is to predict the adjusted closing price of RBC (ticker: RY.TO) using two types of regression models. The first model is a simple linear regression (SLR), which uses only the adjusted closing price of TD Bank (TD.TO) as the explanatory variable. My second model is a multiple linear regression (MLR) that incorporates data from all of Canada's Big Five banks: TD Bank, Scotiabank (BNS.TO), Bank of Montreal (BMO.TO) and Canadian Imperial Bank of Commerce (CM.TO). These banks typically exhibit similar trends due to shared macroeconomic drivers and sectoral behavior, making them suitable predictors for RBC's stock price.

In my SLR model, the regression output shows an R-squared of approximately 0.765, an intercept of 20.12, and a TD coefficient of 1.95. This means that about 76.5 percent of the variation in RBC's stock price can be explained by TD's stock price alone. The positive coefficient indicates that RBC and TD generally move in the same direction. However, my model is limited by its univariate nature and may suffer from omitted variable bias because it excludes the effects of other major banks.

My MLR model offers an improved R-squared of approximately 0.956, with an intercept of 32.5 and individual coefficients for TD (0.2068), BNS (-1.2361), BMO (0.4824), and CM (1.6153). The significantly higher R-squared suggests that my model explains nearly 96 percent of the variation in RBC's price, highlighting the benefit of including multiple predictors. The coefficients suggest that BMO and CM have a strong positive influence on RBC's price, while TD's influence weakens once the effects of other banks are accounted for. Interestingly, BNS has a negative coefficient, which may reflect multicollinearity or structural differences in performance trends, particularly during events such as the COVID-19 pandemic. If my model's output includes a high condition number, like over 1000, it signals multicollinearity, meaning the independent variables are highly correlated, which can affect the stability of the coefficient estimates.

When comparing predicted versus actual RBC prices in a time series plot, my SLR model tends to underfit, especially during periods of high volatility or nonlinear market behavior. In contrast, my MLR model more closely follows RBC's actual price trajectory, especially during key macroeconomic phases such as the 2020 market dip, the post-COVID rally through 2022-2023, and the plateau in 2024. This indicates that my MLR model captures broader market dynamics more effectively than the SLR.

The residual plot from my MLR model reveals a distinct funnel-shaped pattern, suggesting heteroskedasticity—where the variance of residuals increases with the predicted values. This violates the homoscedasticity assumption of linear regression. Additionally, the extremely low Durbin-Watson statistic (0.034) confirms strong positive autocorrelation, indicating that residuals are serially correlated over time. These diagnostic findings imply that ordinary least squares may not be the most reliable approach for this financial time series. To address these issues, I should consider using time-series models such as ARIMA or VAR, which are better equipped to handle autocorrelation. Furthermore, applying robust standard errors or transforming the data (e.g., logarithmic scaling) could help stabilize residual variance and improve inference. Overall, while my MLR model significantly improves predictive accuracy over the SLR model, these residual diagnostics suggest that further refinement is necessary to satisfy key regression assumptions.

Overall, my MLR model significantly outperforms the SLR model. While the SLR model is simpler, it explains less of the variance in RBC's price and does not incorporate the broader banking sector. The MLR model, despite its increased complexity and potential multicollinearity, provides stronger predictive performance and more accurately tracks RBC's movements. The residual structure also improves with MLR, although some further refinements may be necessary to address autocorrelation and other issues typical in financial time series data.

My purpose of this analysis is to model and predict the adjusted closing price of RBC (RY.TO) by leveraging data from peer institutions in the Canadian banking sector. Two regression models are evaluated: a Simple Linear Regression (SLR) using only TD Bank (TD.TO) as a predictor, and a Multiple Linear Regression (MLR) incorporating four other major banks — TD, Scotiabank (BNS.TO), Bank of Montreal (BMO.TO), and Canadian Imperial Bank of Commerce (CM.TO). The objective is to assess how well my models capture RBC's price dynamics, evaluate model fit, interpret coefficients, and diagnose residual behavior to identify modeling limitations.

My simple linear regression model, which predicts RY using TD alone, produces an R² of 0.632. This indicates that approximately 63.2% of the variance in RBC's stock price is explained by TD's price, which is a reasonable starting point but clearly leaves room for improvement. The Akaike Information Criterion (AIC) for this model is 11,680, suggesting a moderate fit. By contrast, the multiple linear regression model that includes TD, BNS, BMO, and CM as predictors achieves a significantly improved R² of 0.956 and a much lower AIC of 8,741. This sharp increase in explained variance and drop in AIC confirm that the multivariate approach offers a far superior fit to the data. The improvement from 63.2% to 95.6% in R² underscores the value of incorporating additional banks that move in tandem with RBC due to common macroeconomic drivers.

Examining the MLR coefficients provides further insights into the individual contribution of each bank to RBC's price. The model intercept is 32.50, which represents the expected base price of RBC when all other predictors are zero — though this has limited real-world interpretation. The TD coefficient is 0.207, meaning that for every one-dollar increase in TD's price, RBC's price is expected to increase by about 21 cents, holding other variables constant. The BMO coefficient is 0.482, reflecting a moderately positive relationship, while CM shows the strongest positive influence on RBC with a coefficient of 1.615. Interestingly, the coefficient for BNS is negative at -1.236, suggesting that a one-dollar increase in BNS correlates with a \$1.24 decline in RBC's price, when other variables are held constant. This inverse relationship could reflect structural differences, countercyclical behavior, or more likely, multicollinearity among the predictors.

The residual plot from the MLR model reveals a noticeable funnel shape, where residual variance increases with higher predicted values. This pattern clearly indicates heteroskedasticity, which violates the OLS assumption of constant variance in residuals. Moreover, the residuals are not randomly scattered around zero, showing clusters and systematic deviations that imply potential nonlinear patterns or model misspecification. The Durbin-Watson statistic of 0.034 confirms the presence of severe positive autocorrelation, meaning that residuals are serially correlated and not independent over time. Together, these findings suggest that the MLR model, while strong in explanatory power, fails to meet several key regression assumptions, which could impact the validity of its predictions and inference.

A visual comparison of predicted versus actual RY values over time shows that my SLR model, which relies solely on TD, consistently underestimates RBC's price, particularly between 2022 and 2025. It struggles to capture shifts in dynamics across the banking sector during periods of volatility. In contrast, my MLR model closely follows the actual price of RBC, particularly during the 2023–2024 period, when interactions among the major banks become more influential. However, even my MLR model begins to slightly overshoot the actual RY price in the final year of the dataset (2025), which could be attributed to strong leverage from the CM coefficient. This points to a possible overfitting issue or a structural change in market behavior not captured by my model.

In conclusion, my multiple linear regression model clearly outperforms the simple linear regression model in terms of predictive accuracy and model fit. It captures more than 95% of the variance in RBC's price, compared to just 63% using TD alone. However, the residual diagnostics raise red flags. The presence of heteroskedasticity, serial autocorrelation, and potential multicollinearity all suggest that the current linear framework may be insufficient. RBC's price is likely influenced by additional non-linear, time-dependent, or macroeconomic factors not currently included in my model. To address these limitations, future work should consider using time-series models like ARIMA or VAR, regularized regressions like Ridge or Lasso to mitigate multicollinearity, or even nonlinear methods such as XGBoost or neural networks to better capture complex market behavior.

### Use Robust Standard Errors

### Log-transform prices to stabilize variance

```
Breusch-Pagan Test (SLR, log-transformed) 
p-value: 1.7351839221278078e-25

Breusch-Pagan Test (MLR, log-transformed) 
p-value: 1.6018719840252692e-14
```
To address the heteroskedasticity identified in my regression models, I explored two remediation strategies: applying robust standard errors using White's HC0 correction and applying logarithmic transformation to stabilize variance. The goal was to ensure valid inference and meet the assumptions required for reliable regression analysis.

In Option 1, I applied heteroskedasticity-robust standard errors using the HC0 correction, which adjusts the standard errors without altering the coefficients themselves. For my simple linear regression (SLR) model predicting RBC's price from TD's price, the R² remained at 0.632, indicating that TD alone explains about 63% of RBC's price variance. After applying robust standard errors, the t-statistic for TD rose to 67.278 with a p-value less than 0.001, confirming a statistically significant relationship. This correction is particularly important given that prior tests, such as the Breusch-Pagan test, had flagged heteroskedasticity in the residuals. With robust standard errors in place, the inference is now considered valid despite the model's violation of homoscedasticity.

For the multiple linear regression (MLR) model, which includes TD, BNS, BMO, and CM as predictors, the R² remained strong at 0.956. All predictors were statistically significant with p-values below 0.001. However, the BNS variable retained a large negative coefficient, which raises concern. This could be a result of multicollinearity, a scenario where predictors are highly correlated with each other, or a structural mean reversion effect where BNS and RY move in opposite directions during certain market conditions. The condition number for this model was 1,340, which exceeds the threshold of 1,000 commonly cited as a warning sign for multicollinearity. This reinforces the need for caution when interpreting individual coefficients in the MLR.

In Option 2, I experimented with log-transforming the response and predictor variables to stabilize variance and potentially eliminate heteroskedasticity. However, my log-transformed models did not improve the situation. The p-values for the log-transformed SLR and MLR remained extremely low, 1.73e-25 and 1.60e-14, respectively, indicating that heteroskedasticity likely persists. This suggests that variance stabilization through log transformation is not effective in this case, possibly because the underlying relationships in the data are nonlinear or influenced by macroeconomic factors not addressed by simple transformations.

In summary, robust standard errors via White's HC0 correction proved to be the most effective and appropriate remedy. They allow me to retain the original model structure while ensuring that inference remains statistically valid even in the presence of heteroskedasticity. Moving forward, I will rely on `.get_robustcov_results(cov_type='HC0')` for all coefficient significance testing and confidence intervals. This approach is both practical and statistically sound, especially when working with financial data that often violate the ideal conditions of classical linear models. While multicollinearity in the MLR model remains a potential concern, the robust standard error framework provides a reliable foundation for further modeling and diagnostics.

###  Check for multicollinearity

| Variable | VIF   | Interpretation                           |
| -------- | ----- | ---------------------------------------- |
| `const`  | 75.12 | Ignore (intercept has no meaningful VIF) |
| `TD`     | 11.90 | ❌ **High multicollinearity**             |
| `BNS`    | 6.48  | ⚠️ Moderate multicollinearity            |
| `BMO`    | 19.47 | ❌ **Very high multicollinearity**        |
| `CM`     | 5.90  | ⚠️ Moderate multicollinearity            |

### Breusch-Pagan Heteroskedasticity Test on Returns

```
Breusch-Pagan Test for SLR (RY_RET ~ TD_RET) 
p-value: 9.8360e-08
⚠️ Heteroskedastic

Breusch-Pagan Test for MLR (RY_RET ~ TD, BNS, BMO, CM) 
p-value: 5.8152e-01
✅ Homoskedastic

```

The SLR (RY ~ TD)'s p-value is 9.84e-08, indicating strong evidence of heteroskedasticity. Meanwhile, MLR (RY ~ TD, BNS, BMO, CM)'s p-value is 0.5815, suggesting no evidence of heteroskedasticity. Therefore, my MLR model on returns is statistically sound in terms of residual variance so there is no need for robust errors. However, my SLR model is heteroskedastic, even on returns. Standard errors and inference may be unreliable.

### Walk Forward

In my modeling approach, daily log returns were computed using the formula `np.log(data / data.shift(1)).dropna()`. This is a standard technique in financial modeling because log returns are additive over time and better capture the effects of compounding. Unlike simple percentage returns, log returns offer mathematical advantages that make them especially useful in regression and time-series forecasting.

Next, I split the data into two parts: a basket of explanatory variables consisting of TD.TO, BNS.TO, BMO.TO, and CM.TO, and a target variable, RY.TO. This setup frames my analysis as a predictive modeling task, where my objective is to use the returns of the other major Canadian banks to forecast RBC's returns. My structure aligns well with my portfolio modeling, risk management and predictive finance use cases.

To support my resampling and time-based rolling analysis, the log return DataFrame was enhanced by introducing a MultiIndex that includes a dummy `'symbol'` level. This enabled me with the application of `.resample('M', level='Date')` to group and aggregate returns at the monthly level. This step is both elegant and scalable, allowing for cleaner manipulation of panel-like data in a time-aware context.

A set of monthly walk-forward dates was then created using the command `df_ind_basket.resample('M', level='Date').mean().index[:-1]`. This generates a series of monthly checkpoints that serve as the basis for walk-forward modeling. By slicing off the last index using `[:-1]`, my model ensures that each training window ends just before the start of the next evaluation period, thus preventing data leakage between training and testing sets.

As a result, a total of 65 walk-forward recalibration dates were generated, ranging from January 31, 2020 to May 31, 2025. These dates are crucial for setting up a dynamic and adaptive modeling pipeline. At each checkpoint, the model will be retrained on data available up to that point and used to predict returns in the following month. This walk-forward framework forms the foundation for rolling regression, dynamic portfolio allocation or more advanced online learning models, where continual adaptation to new market conditions is critical for success.

### Modified Walk Forward


<img width="1579" height="985" alt="Notebook 4 - Modified Walk Forward" src="https://github.com/user-attachments/assets/4d95b588-9f9e-44ea-ad85-53c6f1e02c4d" />


<img width="989" height="390" alt="Notebook 4 - Walk Forward R^2 over time" src="https://github.com/user-attachments/assets/cec8d03d-6e31-49b0-b5d8-6a382b9562de" />

The above walk-forward scatter plots display the relationship between predicted and actual log returns of RBC across multiple monthly recalibration windows. In each subplot, the x-axis represents the predicted out-of-sample log returns, while the y-axis shows the corresponding actual returns. The title of each subplot includes the recalibration date and the R² score for that window, allowing for a quick visual assessment of model performance at each time point.

Upon reviewing these plots, it becomes clear to me that the model's performance varies considerably over time. R² values range from very low, around 0.02 (as seen on November 30, 2020), to as high as 0.81 (on February 28, 2022). High R² values suggest that during certain periods, the model successfully captures short-term comovements between RBC and its peer banks, indicating strong linear relationships. In contrast, low R² scores point to periods where RBC may have moved idiosyncratically, possibly due to company-specific events or where short-term market noise disrupted otherwise stable relationships. These fluctuations indicate that the predictive power of my model is highly context-dependent.

A broader view of the model's performance over time is Walk Forward R² over time, which tracks the evolution of R² scores throughout the walk-forward process. The R² values oscillate dramatically, ranging from near zero to nearly one, suggesting a lack of stability in the linear relationship between RBC and the other banks. Some periods, such as early 2020, late 2021 and the first two quarters of 2023, exhibit relatively high predictability. However, this is counterbalanced by other intervals where the model performs poorly. Such variability is likely a reflection of underlying market regime changes, including macroeconomic shifts like COVID-related volatility, central bank interest rate actions or company earnings surprises. These patterns suggest that my single linear model may not be sufficient to capture the dynamic nature of market behavior, and point toward the potential benefit of dynamic or nonlinear modeling approaches.

Overall, the walk-forward framework demonstrates several strengths and limitations. On the positive side, my model achieves high predictive power in certain market conditions, with R² values occasionally exceeding 0.7. Its simplicity makes it easy to interpret and implement, which is advantageous for practical applications. The use of monthly walk-forward validation also mirrors realistic investment strategies, such as monthly portfolio rebalancing. However, the model's limitations are notable. The predictive power is unstable across time, and the linear structure may underfit during complex or nonlinear market phases. Additionally, the small validation window of approximately ten days could introduce noise and increase the variance of performance estimates. Residual behavior, though implied in the model, warrants further inspection, as errors may not be centered or homoscedastic. Visualizing residuals or adding prediction intervals could provide valuable insights for model refinement.

In conclusion, while my linear walk-forward model provides useful insights during certain market regimes, its inconsistent performance suggests the need for more flexible approaches. Incorporating dynamic features, nonlinear algorithms or regime-switching frameworks may enhance predictive stability and robustness in future iterations.

## Cointegration

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns RY TO and TD TO (p = 0 0000)" src="https://github.com/user-attachments/assets/6d50a2b9-a388-47cf-a896-45b657f3a7c2" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns RY TO and BNS TO (p = 0 0000)" src="https://github.com/user-attachments/assets/04f31598-9d83-4604-9244-ad2b0ea0b48d" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns RY TO and BMO TO (p = 0 0000)" src="https://github.com/user-attachments/assets/53e3634c-29f5-4cdb-9ca2-bbec30321a48" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns RY TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/be2052de-f540-42bc-9fa6-269755b9fec3" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns RY TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/be2052de-f540-42bc-9fa6-269755b9fec3" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns TD TO and BMO TO (p = 0 0000)" src="https://github.com/user-attachments/assets/d2b479c0-cabc-4fde-98cb-2666063019d5" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns TD TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/15661fa9-1b65-4d4e-8b5f-9335c8a6184f" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns TD TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/15661fa9-1b65-4d4e-8b5f-9335c8a6184f" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns BNS TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/1a4e365c-ad84-47e4-9fb1-5c0980fde55d" />

<img width="989" height="389" alt="Notebook 5 - Cointegration on returns BMO TO and CM TO (p = 0 0000)" src="https://github.com/user-attachments/assets/7da1deea-a388-46c6-b217-5bfc7d4f2b39" />

The primary purpose of this code is to analyze the short-term dynamic relationships between the daily log returns of Canada's Big Five banks - RY.TO (RBC), TD.TO (TD), BNS.TO (Scotiabank), BMO.TO (BMO), and CM.TO (CIBC) - using pairwise cointegration tests.

Based on the results from the pairwise Engle-Granger cointegration tests conducted on the daily log returns of Canada's Big Five banks (RY.TO, TD.TO, BNS.TO, BMO.TO, CM.TO) from January 2022 to July 2025, I can draw several key insights. The test was applied to every unique pair of banks, assessing whether there exists a statistically significant long-run equilibrium relationship between their return series. The null hypothesis in the cointegration test is that no cointegration exists between the two series. A p-value below 0.05 indicates I reject the null in favor of the alternative, suggesting cointegration.

Remarkably, all ten bank pairs demonstrated strong cointegration in their log returns, with p-values effectively equal to zero (p = 0.0000 in all cases). This result is atypical because cointegration is generally observed in price levels rather than in returns, which are usually non-stationary and less likely to share long-term equilibrium. The consistent cointegration across return pairs implies that these Canadian bank stocks not only move together over the long run in levels but also exhibit synchronized short-term fluctuations after adjusting for differencing through log returns.

For example, RY.TO and TD.TO, as shown in the corresponding plot, clearly exhibit highly correlated movements in their daily log returns. Similarly, BMO.TO and CM.TO, two banks that frequently show close valuation and operational similarities, also display tight co-movement. Other highly cointegrated pairs include BNS.TO and BMO.TO, RY.TO and CM.TO, and TD.TO and BNS.TO. This high level of pairwise synchronization in returns suggests strong systemic ties across the sector, likely due to shared macroeconomic exposure, interest rate sensitivity, regulatory environment, and investor sentiment.

While the statistical result is strong, it also calls for deeper inspection. Cointegration in returns can sometimes indicate model misspecification, particularly if the return series are actually stationary or if the log returns include overlapping influences (e.g., identical market or sector shocks). Alternatively, it may point to overdifferencing—that is, testing log returns when level data already exhibited cointegration. Another interpretation is that the Canadian banking sector is highly integrated and efficient, with returns adjusting quickly to sector-wide information shocks.

From a practical perspective, these findings support the use of pairs trading strategies, cointegration-based portfolio construction, or statistical arbitrage techniques. The high degree of synchronization also means that predictive models for one bank (e.g., RBC) can benefit substantially from incorporating return information from its peers.

In summary, all tested pairs of Canadian bank log returns are cointegrated at a highly significant level. This suggests that even short-term return dynamics among these institutions are not independent but rather tightly linked. For portfolio managers, risk analysts, or financial modelers, this reinforces the importance of modeling these assets jointly rather than in isolation. However, due to the unusual nature of cointegration in returns, additional tests such as stationarity checks (ADF, KPSS) and residual diagnostics are warranted to validate the robustness and economic interpretability of these findings.

<img width="989" height="389" alt="Notebook 5 - Log Prices RY TO vs TD TO" src="https://github.com/user-attachments/assets/8f9c4ae5-fb61-4733-aae5-76ff3b5b9714" />

<img width="989" height="389" alt="Notebook 5 - Log Prices RY TO vs BNS TO" src="https://github.com/user-attachments/assets/1175141e-c6ca-4b2e-a473-10cf9b539eb2" />

<img width="989" height="389" alt="Notebook 5 - Log Prices RY TO vs BMO TO" src="https://github.com/user-attachments/assets/bc037154-0606-4bb3-9a50-dacb0e912473" />

<img width="989" height="389" alt="Notebook 5 - Log Prices RY TO vs CM TO" src="https://github.com/user-attachments/assets/51cd6bab-5959-4885-ba48-750d7b9d595f" />

<img width="989" height="389" alt="Notebook 5 - Log Prices TD TO vs BNS TO" src="https://github.com/user-attachments/assets/fa6000be-7c19-4d22-a779-786fedbf7e7b" />

<img width="989" height="389" alt="Notebook 5 - Log Prices TD TO vs BMO TO" src="https://github.com/user-attachments/assets/7da68acb-f562-4bc7-954f-31bc22f0ca1d" />

<img width="989" height="389" alt="Notebook 5 - Log Prices TD TO vs CM TO" src="https://github.com/user-attachments/assets/609e7e30-6a9e-4245-9258-821dd813e462" />

<img width="989" height="389" alt="Notebook 5 - Log Prices BNS TO vs BMO TO" src="https://github.com/user-attachments/assets/eb91da46-d2e0-47cb-9100-399d950f188d" />

<img width="989" height="389" alt="Notebook 5 - Log Prices BNS TO vs CM TO" src="https://github.com/user-attachments/assets/bc0e0335-f9a6-41ef-825d-4754e7dae382" />

<img width="989" height="389" alt="Notebook 5 - Log Prices BMO TO vs CM TO" src="https://github.com/user-attachments/assets/a02f421c-ca52-4041-907a-525ff3d29191" />

My above analysis involves testing for pairwise cointegration among the **log price series** of Canada's Big Five banks: RBC (RY.TO), TD Bank (TD.TO), Scotiabank (BNS.TO), BMO (BMO.TO), and CIBC (CM.TO). This approach is more conventional than testing on log returns, as cointegration theory typically applies to non-stationary series like price levels. The goal is to determine whether certain bank pairs share a long-term equilibrium relationship despite short-term fluctuations. The Engle-Granger cointegration test was applied to all ten possible unique bank pairs over the period from January 2022 to July 2025, and results were visualized via side-by-side log price plots.

In nearly every plot, there is a strong visual similarity between the two log price series. For example, the plot of **RY.TO vs TD.TO** displays closely aligned trends, both falling in early 2023 and rising sharply throughout 2024 and 2025. The cointegration test returned a p-value of 0.0000, indicating a statistically significant long-run relationship. Similarly, **BMO.TO and CM.TO** track each other quite well across the entire period, especially during recovery phases. This is consistent with expectations, as both banks often exhibit similar business models and risk profiles.

Other pairs, such as **BNS.TO and CM.TO** or **TD.TO and BMO.TO**, also revealed strong comovement. Though not identical in their short-term fluctuations, their price paths exhibit co-trending behavior, and the cointegration tests confirmed their statistical linkage. Interestingly, even pairs like **RY.TO and BNS.TO**, which occasionally diverge in amplitude or slope, still returned significant cointegration p-values, suggesting that the deviation from their long-run relationship is mean-reverting.

The plots help reinforce the interpretation of the statistical results. In some cases, the price series appear to be on parallel trajectories with consistent gaps, suggesting stable proportional pricing. In others, the gap fluctuates but ultimately returns to equilibrium, reflecting temporary dislocations driven by news, earnings, or macroeconomic variables. Across the board, there is strong empirical evidence that these banks are not only part of the same sector but are also tightly bound in price dynamics due to shared economic exposure.

From a modeling perspective, these results validate the use of multivariate time-series methods such as Vector Error Correction Models (VECM) or cointegration-based long-short strategies in pairs trading. Since the log prices of these banks are cointegrated, deviations between their values could be exploited for statistical arbitrage, assuming mean reversion holds in the future. These findings also justify constructing risk or factor models using linear combinations of bank prices, as their long-term dependencies are statistically grounded.

In conclusion, my analysis demonstrates that the major Canadian banks move in coordinated patterns not only in returns but also in log prices. The strong cointegration detected across all pairs supports the hypothesis that these institutions are governed by shared fundamentals and systemic drivers. This presents both opportunities and risks for investors, analysts and modelers who rely on the stability of inter-bank relationships to forecast behavior or design financial strategies.

### Using cointegration to build a basket

<img width="735" height="590" alt="Notebook 5 - Correlation Matrix" src="https://github.com/user-attachments/assets/05492950-8e1c-4cc5-bd11-624cbc3d6d49" />

The cointegration test results for the log prices of Canada's Big Five banks over the period from 2022 to 2025 were summarized in a matrix of p-values using the Engle-Granger methodology. According to standard interpretation, a p-value below 0.05 indicates statistical evidence of cointegration, meaning I can reject the null hypothesis that no cointegration exists between the pair. Conversely, a p-value greater than or equal to 0.05 suggests no cointegration, and the relationship between the two series is likely non-stationary in the long run.

Upon reviewing the results, none of the bank pairs show statistical cointegration at the 5 percent significance level. The lowest p-value was observed for the pair BMO.TO and BNS.TO, which yielded a p-value of 0.14. Although this is below some informal thresholds used in exploratory analysis, it is still above the conventional cutoff of 0.05. As a result, no bank pair meets the criteria for strong statistical cointegration based on this test.

That said, there are several pairs with relatively low p-values that may still suggest economic relatedness or co-movement worth investigating further. These include BMO.TO and BNS.TO with a p-value of 0.14, BMO.TO and TD.TO with 0.26, and BNS.TO and CM.TO with 0.33. While not statistically significant under strict hypothesis testing, these values may indicate some level of economic co-integration or long-term alignment in pricing behavior, particularly in the context of spread trading or mean-reverting strategies.

In such cases, analysts and traders may consider relaxing the significance threshold slightly, for instance, to 10 percent, when selecting candidates for cointegration-based trading. This approach is especially useful in financial markets where small sample sizes or market frictions can reduce statistical power. Ultimately, while formal cointegration is not established at the standard 5 percent level, some of these bank pairs may still have practical use in portfolio construction, pairs trading, or other applications that rely on stable long-term relationships.

### Canadian Banks MLR (Predicting RBC from TD + BNS)

<img width="1189" height="490" alt="Notebook 5 MLR" src="https://github.com/user-attachments/assets/ddc814ba-2a0f-45eb-b3a9-269f0aa078cb" />

My model summary outlines the performance of a multiple linear regression (MLR) model designed to predict the log price of RBC (RY.TO) using the log prices of TD Bank (TD.TO) and Scotiabank (BNS.TO) as predictors. The model was trained on data spanning from January 1, 2022, to January 31, 2024. The adjusted R-squared value is 0.4196, indicating that approximately 42 percent of the variation in RBC's log price can be explained by the combined movements of TD and BNS during this period.

An adjusted R-squared of around 42 percent reflects moderate explanatory power. This suggests that while TD and BNS prices contain meaningful information about RBC's pricing behavior, they do not capture the full complexity of RBC's dynamics. The model is capable of identifying general co-movements but falls short of delivering highly precise predictions.

My above visual analysis supports this interpretation. The predicted log price of RBC (represented by the orange line) follows the general direction of the actual log price (shown in blue) over medium-term time windows. However, my model struggles to capture sharper market movements. Notably, between the third and fourth quarters of 2022, the predicted values remain relatively flat while actual prices rise and fall. A similar pattern occurs in late 2023, when RBC's actual price recovers more quickly than the model anticipates. These discrepancies indicate that the model lacks certain dynamic inputs and may not be accounting for variables unique to RBC.

This performance gap may stem from the absence of idiosyncratic factors such as company-specific news, risk management strategies, capital structure differences or dividend policies. While TD and BNS share systemic traits with RBC, they cannot fully explain all of its price behavior on their own.

Strategically, my model is useful for capturing general sector co-movement and could serve as a basic hedging tool in a portfolio. However, it is not robust enough to function as an accurate tracking portfolio or for short-term return prediction. For those purposes, more sophisticated models incorporating additional variables or nonlinear methods may be required.

### Residual Spread and Cointegration Analysis for Predicting RY.TO Using Peer Bank Log Prices

<img width="989" height="389" alt="Notebook 5 - In sample Spread" src="https://github.com/user-attachments/assets/7e51eb82-5fc2-479b-a0be-57f4d55aaefa" />

<img width="989" height="390" alt="Notebook 5 - Out sample Spread" src="https://github.com/user-attachments/assets/e6dbf1b6-0135-4eff-8a9b-1be42c043d4e" />

<img width="1190" height="489" alt="Notebook 5 - Actual vs Predicted RY Log Price (Out of sample)" src="https://github.com/user-attachments/assets/0a2b75ab-c9df-4b67-a78e-a6a2385f1ad1" />

<img width="990" height="389" alt="Notebook 5 - In sample Spread (BNS - Predicted Basket)" src="https://github.com/user-attachments/assets/441a6e08-b240-44e5-9b0e-f54a8c8e1460" />

<img width="989" height="390" alt="Notebook 5 - Out sample Spread (BNS - Predicted Basket)" src="https://github.com/user-attachments/assets/e2ea3f40-6621-4f3c-a2f0-454157a35864" />

<img width="1190" height="489" alt="Notebook 5 - Actual vs Predicted BNS Log Price (Out of sample)" src="https://github.com/user-attachments/assets/0b6b04a6-14d7-452d-a5bb-aa1a80338064" />

My analysis focuses on modeling RBC's log price (RY.TO) using a multiple linear regression (MLR) approach, where the predictors are the log prices of TD Bank (TD.TO), Scotiabank (BNS.TO), BMO (BMO.TO), and CIBC (CM.TO). My model was trained using data from January 2022 to the end of January 2024, and tested on data from April 2024 to July 2025. My primary goal is to examine whether the predicted RBC price from a linear combination of its peer banks can produce a stable residual spread, indicating potential cointegration or mean-reverting behavior.

My in-sample regression produced a spread between actual and predicted log prices of RBC, which was then subjected to the Augmented Dickey-Fuller (ADF) test to assess stationarity. The p-value from the in-sample ADF test was approximately 0.1357, and the test statistic was -2.42. Since the p-value is above the conventional 0.05 threshold, I cannot reject the null hypothesis of a unit root, suggesting that the spread is not statistically stationary in-sample. Visually, the in-sample spread plot shows some mean-reverting patterns but with noticeable cycles and persistent deviations around the mean. Although the spread fluctuates around a flat trendline, its lack of stationarity undermines the case for reliable long-term equilibrium.

My model was then applied out-of-sample to predict RBC's log price from the same basket of bank stocks. The predicted and actual prices diverge consistently over time. The predicted line underestimates RBC's price movements, especially during upward rallies between mid-2024 and early 2025. This divergence indicates that the MLR model, while moderately effective at capturing co-movement in training, fails to generalize well to future periods. The lack of dynamic adjustment or inclusion of time-varying factors likely contributes to this underperformance.

The out-of-sample spread—the difference between actual and predicted log prices of RBC—was also tested for stationarity using the ADF test. The resulting p-value was approximately 0.1394, with a test statistic of -2.41. Like the in-sample case, this result suggests that the spread remains non-stationary and does not exhibit reliable mean-reversion behavior. Visually, the spread trends upward and remains elevated for much of the test period, further confirming the model's instability outside the training window.

These results indicate that the chosen MLR model does not capture a stable long-run relationship between RBC and the selected peer banks. Although the visual alignment in the training phase appears reasonable, the statistical tests and out-of-sample plots reveal significant drift and structural deviation. This undermines the use of my model for cointegration-based strategies, such as pairs trading or statistical arbitrage, where mean-reverting spreads are essential.

In conclusion, the spread between RBC's actual log price and the price predicted by the linear basket of TD, BNS, BMO, and CM is not stationary in either the training or testing period. My MLR model demonstrates some predictive power but lacks robustness and fails to produce a residual series that satisfies the conditions for mean reversion. Future efforts should explore more dynamic models, such as Vector Error Correction Models (VECM), regime-switching models, or machine learning techniques that can capture nonlinearities and structural shifts more effectively.

### Using Log Returns (SVR vs Random Forest on RY.TO)

<img width="1189" height="490" alt="Notebook 5 - RY TO Log Returns - SVR vs Random Forest (Out of Sample)" src="https://github.com/user-attachments/assets/2c33f702-37bc-4633-b29f-5c99e5d5a460" />

This analysis compares two machine learning models, Support Vector Regression (SVR) and Random Forest (RF), in predicting the daily log returns of RBC (RY.TO) using the log returns of four major Canadian banks: TD.TO, BNS.TO, BMO.TO, and CM.TO. My focus is on evaluating both models accuracy and their ability to generate stationary residual spreads, which is essential for applications like short-term forecasting and return-based trading strategies.

The data includes log returns calculated from daily adjusted closing prices spanning January 2022 through July 2025. The training set covers up to the end of January 2024, while the testing set begins in April 2024. A grid search was used to optimize hyperparameters for the SVR model, while the Random Forest model used 200 estimators with a maximum depth of five.

Both models performed well on the training set. My SVR model achieved an R-squared value of approximately 0.65 and a root mean squared error (RMSE) of 0.00585, while my Random Forest model outperformed it with an R-squared of 0.81 and a lower RMSE of 0.00434. Importantly, the Augmented Dickey-Fuller (ADF) test on the training residuals showed p-values close to zero for both models, indicating that the residuals were stationary and mean-reverting during the training period. This suggests that both models captured stable and predictable relationships within the in-sample data.

In the out-of-sample test period, both models maintained relatively strong performance. SVR produced an R-squared value of 0.42 and an RMSE of 0.00778, while Random Forest had an R-squared of 0.43 with a slightly lower RMSE of 0.00769. The residuals from both models remained highly stationary in the test set, with the SVR model showing an ADF test statistic of -20.07 and the Random Forest model showing -15.32. These strong negative values and extremely low p-values confirm that the residual spreads continued to exhibit stationarity, which is favorable for return-based modeling.

The prediction plot supports these findings. The actual log returns of RY.TO are captured with reasonable accuracy by both models. SVR occasionally shows sharper, more reactive movements, while the Random Forest model produces smoother and slightly more conservative forecasts. Both models align well with short-term fluctuations, although some of the larger spikes in actual returns remain difficult to predict, which is typical in financial return series.

Overall, my analysis demonstrates that return-based modeling is more effective and robust than price-level modeling for predictive tasks involving RY.TO and its peers. Both SVR and Random Forest produced stable and accurate forecasts of returns, with Random Forest showing slightly better in-sample fit and generalization. The stationarity of residuals further validates the modeling approach and supports the suitability of return forecasting in financial applications such as signal generation, short-term portfolio rebalancing, or machine learning-based trading strategies.

## Strategy Implementation

### Momentum and Defensive Performance Analysis of Canadian Banks Relative to S&P Market Direction

<img width="989" height="490" alt="Notebook 6 - Strategy Implementation - Momentum" src="https://github.com/user-attachments/assets/ca34f0ca-1660-4c89-9a66-4fd1756f3ac1" />

My analysis summarizes how each of Canada's major banks performed relative to the S&P 500 index on days when the market was either rising or falling. The goal is to identify which banks tend to outperform during bullish (up) or bearish (down) sessions, helping guide both momentum and defensive strategy construction.

During S&P up days, BMO.TO showed the strongest relative performance, outperforming the market by an average of 0.057 percent. This indicates that BMO tends to exhibit positive momentum when broader markets are rising. CM.TO and TD.TO also posted modest outperformance on bullish days, suggesting they benefit from risk-on market environments. These stocks may be favorable for momentum-based strategies that seek exposure to rising trends.

On the other hand, when the S&P 500 was in decline, RY.TO emerged as the most resilient bank. It managed to outperform the index by 0.046 percent on down days, indicating strong downside protection. CM.TO also provided a small positive edge during bearish sessions. This resilience suggests both RY.TO and CM.TO are potentially strong candidates for defensive or low-volatility portfolios that seek relative safety during market downturns.

BNS.TO stood out as the weakest performer, underperforming on both up and down days. Its returns fell slightly short of the S&P benchmark in both market conditions, offering no discernible advantage in either momentum or defensive strategy contexts. Investors may consider BNS a neutral or avoid position if seeking tactical outperformance.

Overall, my findings suggest that for momentum-oriented strategies that favor upside participation, BMO.TO and CM.TO are attractive choices. For strategies focused on capital preservation and minimizing drawdowns, RY.TO and CM.TO offer the best downside protection. BNS.TO, lacking relative strength in either scenario, does not present a clear edge and may be deprioritized in active portfolio construction.

### Relative Return Analysis of Canadian Banks vs TSX: Momentum and Defensive Strategy Insights

<img width="989" height="490" alt="Notebook 6 - Strategy Implementation - Monentum - TSX" src="https://github.com/user-attachments/assets/f02fb096-6a35-4892-9992-ed6b4d02d203" />

My analysis evaluates the daily relative return performance of Canada's five largest banks compared to the TSX index, distinguishing between up-market and down-market conditions. My objective is to identify which banks exhibit strong momentum during market rallies and which ones provide downside resilience during market declines.

On days when the TSX was rising, all five banks, RY.TO, TD.TO, BNS.TO, BMO.TO, and CM.TO - outperformed the index. Among them, BMO.TO delivered the most substantial upside, outperforming the TSX by an average of 0.155 percent. CM.TO and TD.TO followed with notable positive excess returns. This consistent outperformance on TSX up days indicates that these banks, especially BMO.TO, exhibit strong momentum characteristics and tend to amplify gains during bullish sessions.

In contrast, during TSX down days, all five banks underperformed the index, suggesting that Canadian bank stocks are more sensitive to market downturns. BMO.TO had the weakest downside performance, trailing the TSX by 0.133 percent on average. TD.TO and CM.TO also recorded significant underperformance, reinforcing their higher volatility profiles. RY.TO, while still underperforming, showed the smallest relative drawdown, making it comparatively more stable in declining markets.

These dynamics offer guidance for strategy selection. For momentum-driven strategies that aim to capture gains during market rallies, BMO.TO, TD.TO, and CM.TO are strong candidates due to their consistent outperformance when the TSX rises. On the other hand, for defensive or risk-averse strategies that prioritize capital preservation during market downturns, RY.TO is the most suitable option, given its relatively mild underperformance. CM.TO presents a balanced profile, with meaningful upside and moderate downside, making it potentially attractive for investors seeking all-weather exposure.

The accompanying visualization supports these findings. It highlights how BMO.TO leads in upside excess return relative to the TSX (blue bars), while all banks, including BMO and TD, experience steeper losses than the index on down days (red bars). This asymmetrical behavior implies that these stocks behave as high beta, momentum-driven assets. As such, investors should carefully align their bank stock selections with their market outlook and risk appetite, favoring BMO and TD in bullish environments and leaning toward RY or CM in more defensive allocations.

### Weekly Outperformance Prediction of Canadian Banks Relative to the S&P 500 Using Machine Learning

```
 keystats_canadian.csv created with daily returns, S&P500 benchmark, and engineered features.
```

### Test Outperformance and Underperformance

```

🏦 Predicting Canadian Bank Momentum vs S&P500...


✅ 0 predicted to OUTPERFORM S&P500 by > 0.05%:


🔻 0 predicted to UNDERPERFORM S&P500 by > 0.15%:

```

Daily Test of Outperformance and Underperformance failed to produce any predictions, likely due to high noise in daily data, thresholds not being frequently met or signal drowned out by short-term volatility. Small daily changes often fall within noise bands and lead to few or no label = 1 cases, which hinders training. Then I will implement larger weekly changes that are more likely to surpass thresholds, so the classifier can learn actual patterns and generate meaningful signals.

```
✅ Predicted to outperform S&P500 (>0.50% weekly):
['BNS.TO' 'BNS.TO']

🔻 Predicted to underperform S&P500 (<-1.50% weekly):
None
```
My analysis focuses on predicting which of Canada's major banks are likely to outperform or underperform the S&P 500 index on a weekly basis using historical data and engineered financial features. The models apply supervised machine learning with Random Forest classifiers and utilize data from January 2018 to the present, including tickers for Royal Bank of Canada (RY.TO), TD Bank (TD.TO), Scotiabank (BNS.TO), BMO (BMO.TO), and CIBC (CM.TO).

To prepare the dataset, weekly adjusted closing prices were collected and converted into log returns. Additional engineered features included a 5-week rolling average return (momentum proxy), a 10-week volatility (risk estimate), and a 10-week cumulative return (long-term trend strength). These rolling features were computed separately for each ticker.

Outperformance and underperformance labels were defined using relative return thresholds against the S&P 500 benchmark. A bank was labeled as an outperformer if its return exceeded the S&P 500 by at least 0.5 percent in a given week. It was labeled as an underperformer if its return lagged the S&P 500 by more than 1.5 percent. These thresholds help the model detect significant deviations in performance rather than minor fluctuations.

Random Forest classifiers were then trained separately for the outperform and underperform classifications using the engineered features. The models were applied to the most recent two weeks of data for each bank to produce forward-looking predictions.

The latest results suggest that Scotiabank (BNS.TO) is predicted to outperform the S&P 500 in both of the most recent two weeks. No banks were flagged for underperformance under the defined threshold. This prediction implies that BNS.TO currently exhibits strong recent momentum and favorable technical indicators relative to the S&P 500.

My machine learning framework offers a simple but powerful tool for weekly momentum screening and risk flagging among Canadian bank stocks. It can be used to support tactical asset allocation, hedge adjustments, or signal generation in a rules-based trading strategy. Regular retraining and validation would further enhance robustness, especially in volatile or changing market conditions.

### Feature Importance Analysis in Weekly Outperformance Model

<img width="642" height="435" alt="Notebook 6 - Feature Importanace" src="https://github.com/user-attachments/assets/a1a1ca3a-a8f9-4f1b-a0d9-ab87eda9470b" />

Now I analyze the feature importance results from the Random Forest classifier used to predict Canadian banks' weekly outperformance relative to the S&P 500. The model is based on three engineered features: 5-week average return, 10-week momentum, and 10-week volatility. The importances are calculated based on their relative contribution to model decision-making.

The 5-week average return accounts for approximately 34 percent of the model’s decision power. This suggests that recent short-term return performance is the most influential factor. A bank stock with consistently strong returns over the past five weeks is more likely to be flagged by the model as a potential outperformer in the upcoming week.

Close behind is the 10-week momentum feature, which contributes about 33 percent to the model. This feature captures the broader trend over a slightly longer time horizon. Its nearly equal importance to the 5-week average return implies that the model values both short-term and medium-term price action when identifying strong candidates.

Interestingly, the 10-week volatility feature also holds roughly 33 percent importance. Lower volatility appears to be positively associated with outperformance, meaning that the model prefers stocks with not only upward trends but also relative stability. This highlights that the model is not purely chasing high-return candidates but also considers the consistency of those returns.

Overall, my model demonstrates a balanced perspective, weighing return-based momentum signals and risk-based volatility nearly equally. This distribution indicates a well-structured and non-redundant feature set. It also suggests that the model tends to favor bank stocks that exhibit a combination of strong recent gains and controlled risk exposure, aligning with momentum strategies that avoid excessive drawdowns.

### Backtesting Random Forest Predictions on Canadian Bank Stocks vs S&P500

<img width="846" height="470" alt="Notebook 6 - Outperformer Stocks Cumulative Returns" src="https://github.com/user-attachments/assets/ad606816-14aa-4bf8-92e6-0f8e5f16a0a5" />

<img width="846" height="470" alt="Notebook 6 - Underperformer Stocks Cumulative Returns" src="https://github.com/user-attachments/assets/5d3cbc08-54fa-4bad-afe7-cdc168b978a8" />

My Random Forest backtest aimed to evaluate how well the model could distinguish between Canadian bank stocks that would outperform or underperform the S&P500 benchmark using engineered weekly features such as 5-week average returns, 10-week volatility, and 10-week momentum.

For the outperformer classification model, the results showed an accuracy of 67% and a precision of 52%. This means the model correctly identified 67% of predictions overall and when it predicted a stock to outperform, it was correct about 52% of the time. The backtest executed 99 "buy" trades based on these outperformer predictions. These trades achieved an average weekly return of 1.0%, compared to the market's 0.4% in the same periods. This led to a 0.6 percentage point outperformance over the S\&P500, indicating a modest edge in identifying upside momentum.

For underperformers, the model had a higher accuracy of 80% but a much lower precision of 27%. This reflects the challenge of reliably identifying losing trades. A total of 22 "sell" signals were triggered. These predicted underperformers had an average return of -3.1%, while the market in those same weeks averaged -1.9%. The difference of -1.2 percentage points confirms that the selected stocks underperformed even the declining market, but not dramatically so. This supports that the model is somewhat capable of identifying downside risk, albeit with limited precision.

Looking at cumulative return plots, the outperformer strategy significantly beat the benchmark, with the equity curve diverging sharply upwards. On the other hand, the underperformer portfolio performed worse than the market, confirming its negative predictive value. However, the low precision on this side of the model suggests results should be interpreted cautiously.

Overall, my backtest confirms that the feature set provides meaningful signals, particularly for identifying stocks with potential upside. The model is less reliable for catching underperformers, highlighting room for improvement in classification thresholds or feature expansion. Nonetheless, this setup could form the basis for a dual strategy: overweighting predicted outperformers while treating underperformers more as a volatility or risk flag than a short signal.

### Evaluation of Ensemble Model Performance for Weekly Canadian Bank Predictions

<img width="1390" height="690" alt="5 Canadian Bank Stock Prices (2020 - Present)" src="https://github.com/user-attachments/assets/758bebc9-38db-4ded-95ad-082a6231bc83" />

<img width="1389" height="690" alt="Canadian Benchmark Indexes (TSX, XIC TO, XIU TO)" src="https://github.com/user-attachments/assets/11372334-1038-4a56-8df7-a8c520e699e6" />

My analysis compared two classifiers, one for identifying potential outperforming stocks and another for underperformers, based on weekly financial features for Canadian banks relative to the S\&P 500. My model used an ensemble of Random Forest, Logistic Regression, and Support Vector Classifier (SVC) to improve robustness.

For the outperformers classifier, my model achieved an accuracy of 53% and a precision of 42%. The confusion matrix showed 116 true negatives, 177 false positives, 39 false negatives, and 130 true positives, from a total of 307 predictions. While the precision isn't exceptionally high, the classifier was profitable. The cumulative returns of stocks predicted to outperform more than doubled that of the market, indicating strong predictive power in identifying upward-trending stocks.

The underperformers classifier performed modestly, with an accuracy of 57% and a lower precision of 29%. The confusion matrix for this model showed 201 true negatives, 154 false positives, 45 false negatives, and 62 true positives, out of 216 total predictions. Despite the low precision, the classifier served its purpose in helping to avoid poorly performing stocks. The group of predicted underperformers underperformed the market by 43%, aligning with model expectations.

Several factors contributed to the model's effectiveness. The liberal voting mechanism (predicting positives if any of the classifiers agree) increased coverage. The dataset had roughly 35% positive class instances, offering a good balance between signal and noise. The ensemble model leveraged the strengths of each classifier type, while the three engineered features, 5-week average return, 10-week volatility, and 10-week momentum—were effective for weekly prediction horizons.

To further improve model performance, there are several promising strategies. On the feature engineering side, incorporating industry-adjusted returns using sector-specific ETFs like ZEB.TO, or adding macroeconomic indicators such as USD/CAD exchange rates and interest rates, could help contextualize each stock's behavior. Including lagged returns as inputs may also better capture momentum shifts.

From a modeling standpoint, shifting from equal voting to a weighted ensemble based on validation AUC would allow stronger models to carry more influence. Adding models like Gradient Boosting or XGBoost could also enhance non-linear pattern detection. Additionally, switching to time-series cross-validation would provide a more realistic assessment of out-of-sample generalization.

Finally, the strategy logic could be refined. Selecting only the top N highest-probability outperformers could concentrate returns and reduce noise. Filtering predictions by risk metrics like Sharpe ratio or maximum drawdown would improve robustness. Simulating transaction costs would also bring the backtest closer to real-world conditions and stress-test the strategy's profitability.

In summary, my ensemble model delivers meaningful predictive power for both long and short strategies on Canadian bank stocks, with room for enhancement through better validation, additional features, and portfolio filters.

### Canadian Financial Market Overview (2019-Present)

<img width="1389" height="690" alt="Canadian Bond ETF Prices (2020 - Present)" src="https://github.com/user-attachments/assets/6de884f8-8fe7-49e0-9f79-3a4ac5d1edbb" />

<img width="1389" height="690" alt="Notebook 6 - Canadian Benchmark" src="https://github.com/user-attachments/assets/528a0e02-9075-4d8f-9a37-eb535acc391b" />

The plots I generated  are highly relevant to what I did earlier in my machine learning-based financial modeling work. These charts help me contextualize and validate the behavior of the Canadian bank stocks I used as targets in my models.

To begin with, I previously built classification models to predict whether Canadian bank stocks, such as RY.TO, TD.TO, BNS.TO, BMO.TO, and CM.TO, would outperform or underperform the S\&P 500 on a weekly basis. The price plots I created now provide long-term context for these exact stocks, visually highlighting trends, momentum bursts, and relative performance differences. For instance, RY.TO clearly outperformed over time, which is consistent with its higher predicted outperformance rate in my model, while BNS.TO lagged behind, validating its weaker profile in the classifier output.

Next, I included Canadian equity benchmarks: GSPTSE, XIC.TO, and XIU.TO, which are more geographically aligned with Canadian banks compared to the S\&P 500. While I used the S&P 500 as my benchmark earlier, these plots suggest that incorporating domestic indexes could help me build more accurate models. Using Canadian benchmarks as the reference point would likely reduce bias and improve the explanatory power of my predictive features.

I also plotted Canadian bond ETFs, such as XBB.TO and XGB.TO, which function as lower-risk or near risk-free investment vehicles. These bond trends can act as proxies for macroeconomic conditions or interest rate environments. Including their returns or volatilities as macro features in my classification model could enhance regime detection and provide better context for the behavior of the bank stocks I'm modeling.

Moreover, the long-term stock price charts visually confirm that my engineered features, such as 5-week average return, 10-week momentum and 10-week volatility, are relevant. The charts reveal clear periods of volatility clustering and sustained price trends, supporting the use of these time-based features. This gives me greater confidence that my model is not just fitting noise but learning from meaningful market structure.

Lastly, these plots strengthen my ability to interpret model outputs and consider practical investment applications. By comparing the behavior of predicted outperformers and underperformers against actual benchmark movement, I can better assess the real-world value of my predictions. These visualizations also set a foundation for building more robust portfolio simulations or walk-forward strategies in the future.

Overall, these charts tie directly back to my earlier work by reinforcing the importance of my target assets, suggesting ways to improve benchmark selection, supporting feature relevance, and helping me visualize model-informed strategies over time.

## Efficient Frontier Analysis for 5 Canadian Banks (2019–2024)
<img width="961" height="701" alt="Notebook 8 - Efficient Frontier" src="https://github.com/user-attachments/assets/1f8010cd-dacb-498d-b14f-e58cab36fe9c" />


My analysis applies portfolio theory to assess risk-return tradeoffs among the five largest Canadian banks: RY.TO, TD.TO, BNS.TO, BMO.TO, and CM.TO. Using historical adjusted close prices from 2019 to the end of 2024, I calculated daily returns and annualized both expected returns and the covariance matrix. A Monte Carlo simulation of 10,000 random portfolios was conducted to visualize the efficient frontier.

The resulting scatter plot represents portfolios by their expected return and volatility (standard deviation), color-coded by Sharpe Ratio. Two optimal portfolios were identified. The red star marks the portfolio with the maximum Sharpe Ratio, which indicates the best risk-adjusted return. Its allocation is heavily weighted toward BMO.TO (about 59 percent) and BNS.TO (about 36 percent), with minimal exposure to the remaining banks. This portfolio achieves the highest excess return per unit of risk.

The blue cross represents the minimum volatility portfolio, which offers the lowest risk exposure. It is also significantly allocated to BMO.TO (about 62 percent) and TD.TO (about 23 percent), suggesting BMO.TO plays a central role in stabilizing return variability. This portfolio would be suitable for more conservative investors prioritizing stability over aggressive returns.

The simulation highlights the wide distribution of risk-return profiles, where some portfolios clearly dominate others in terms of Sharpe Ratio. The frontier also reinforces the importance of diversification, as the top-performing portfolios blend exposure to multiple banks rather than concentrating on a single stock.

My optimization exercise supports practical applications such as asset allocation, risk management, or wealth advisory planning. It offers a framework to quantify trade-offs between return and volatility and provides a foundation for building more advanced strategies, including dynamic portfolio rebalancing or stress-testing under various economic scenarios.

## Forecasting Canadian Bank Stock Prices Using RNNs

### RNN ver 1 (LSTM)

<img width="989" height="390" alt="RNN - plot 1" src="https://github.com/user-attachments/assets/fc7a9385-0c48-47f2-aedc-e0a95f0bd2ba" />

<img width="989" height="290" alt="RNN - plot 2" src="https://github.com/user-attachments/assets/bb063032-e7f4-466e-9bc3-c6a1acde2747" />

<img width="989" height="390" alt="RNN - plot 3" src="https://github.com/user-attachments/assets/6d4d03eb-af38-431c-a99e-d9a3efd742a1" />

<img width="989" height="290" alt="RNN - plot 4" src="https://github.com/user-attachments/assets/31c63c20-f869-4f5d-98c3-42b71c0270f7" />

<img width="989" height="390" alt="RNN - plot 5" src="https://github.com/user-attachments/assets/8fa158cc-c5c6-4f0c-a587-879878f2fa2d" />















