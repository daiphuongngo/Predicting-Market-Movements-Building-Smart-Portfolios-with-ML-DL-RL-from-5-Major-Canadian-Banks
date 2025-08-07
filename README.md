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




