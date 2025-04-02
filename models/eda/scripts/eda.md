## current thoughts

1. make video where you plot a lot of cve over time. y epss, x time
1.1 might be interesting to plot different epss behaviours groups: high epss, low epss, high variance epss, high mean variance or like many ups and downs, medium high epss, steady epss, fluctuating epss, more?
1.2 subset also epss with missing values in it's time series. plot them.
1.3 Maybe we can try to identify cve with also a steady increase: maybe we can reverse engineer why the epss model can actually have steady increase there? What is special about these cve?
1.4 maybe create a segment with cve where the epss gets above the threshold then goes below and goes above again. 
1.4.1 Maybe here also some that drop waaay below again and go up a lot again. 
1.5 maybe also segment on ones where the epss score actually captures the birth of the cve. So epss has a score from day 0 of id assignment. 
1.6 maybe also create a segment for cves that were identified as interesting by papers. 

## GPT https://chatgpt.com/share/67e9cf7b-73f8-800f-a09c-0be770167498:
Below is a comprehensive, step-by-step explanation of how you might conduct exploratory data analysis (EDA) for EPSS data (probability values in \([0, 1]\)) across hundreds of thousands of CVEs. The aim is to help you:

1. Understand how EDA typically looks for a **single time series** (the simplest case).  
2. Contrast that with your **multi-entity, multi-time-series** data (the actual context).  
3. Lay out **best practices** and **common techniques** to explore this kind of dataset without worrying (yet) about the final modeling approach (classification vs. regression, which loss function, etc.).  

The result should equip you to make better research decisions and set up any subsequent analysis, feature engineering, and modeling phases correctly.

---

## 1. EDA for a Single Time Series (One CVE, One EPSS Score Series)

Imagine you only have one CVE, with a daily (or weekly) EPSS score: \( y_t \in [0,1] \). How would you typically explore this single time series?

### 1.1 Plot the Series Over Time
- **Line Plot**: The most direct visualization. On the x-axis is time (e.g., days), on the y-axis is the EPSS value (between 0 and 1).  
  - *What to look for*: Trends (increasing/decreasing), abrupt changes, volatility, periodic patterns.  

### 1.2 Summary Statistics
- **Basic Stats**: Mean, median, min, max, standard deviation of the time series.  
- **Range & Distribution**: Because EPSS is in \([0,1]\), you can see if the values are mostly near 0, mostly near 1, or spread in between.  

### 1.3 Missing Data Analysis
- **Check Missingness**: If there are days with no reported EPSS score, how frequent is that? Are there large gaps?  

### 1.4 Stationarity and Seasonality Checks
- **Stationarity**: Although strict stationarity is less commonly assumed for purely probabilistic forecasts, you might still want to see if the series is stable or if it drifts upward/downward.  
- **Seasonality**: Does the probability vary on a weekly or monthly cycle? This might appear in:
  - Autocorrelation plots (ACF, PACF).  
  - Rolling average patterns (e.g., 7-day or 30-day rolling mean).  

### 1.5 Outlier and Change-Point Detection
- **Outliers**: Single days with extremely high or low scores (like a jump to near 1).  
- **Structural Changes**: Times when the series shifts from one regime to another.  

### 1.6 Potential Next Questions After Single-Series EDA
- “Is the data mostly near 0.0 or 0.9, etc.?”  
- “Are there spikes around certain events or announcements?”  
- “Is there a daily or weekly pattern to the probabilities?”  

**At this stage**, you have a robust sense of how one CVE’s EPSS time series behaves.  

---

## 2. Contrasting Single-Series EDA with the Actual Problem

Now your real context is not just one CVE but **hundreds of thousands of CVEs**, each with its own EPSS time series. Summarizing the differences:

1. **Scale**: You cannot manually plot 200k time series in a single figure. You need aggregated or systematic approaches.  
2. **Time Alignment**: The CVEs may start at different dates, end at different dates, and have different lengths or gaps.  
3. **Objective**: You likely need to find global patterns (e.g., how typical is it for a CVE’s EPSS to jump around?), while also spotting outliers or special subgroups.  

In other words, you’re dealing with a **multi-entity, multi-time-series dataset**:  
- Each entity = one CVE.  
- Each entity has time steps = daily EPSS scores (maybe partial or missing).  

---

## 3. How to Perform EDA for Multi-Entity, Multi-Time-Series (EPSS Only)

Below is a detailed outline of best-practice EDA steps for your scenario. These steps help you glean insights into the distribution and behavior of your data across all CVEs *without* necessarily diving into advanced modeling or external features yet.

---

### 3.1 Step 1: Organize and Inspect the Data Structure

1. **Data Format**  
   - Typically, you’ll have a table like:  
     | cve\_id | date       | epss\_score |  
     |--------|------------|-------------|  
     | A      | 2022-01-01 | 0.12        |  
     | A      | 2022-01-02 | 0.10        |  
     | B      | 2021-06-11 | 0.87        |  
     | ...    | ...        | ...         |  

2. **Identify Missingness**  
   - Check how many missing scores exist (per CVE and overall).  
   - Some CVEs might have a short time series, others a long one.  
   - Some might have sporadic data.  

3. **Initial Aggregates**  
   - Number of unique CVEs.  
   - Earliest and latest date in the dataset.  
   - Overall coverage ratio (what fraction of days for a typical CVE is covered?).  

---

### 3.2 Step 2: Examine Distributions Across CVEs

You can’t just look at one CVE at a time. Instead, do aggregated or comparative analyses.

1. **Snapshot Distributions**  
   - Pick a specific date (or day-of-year) and see how the EPSS distribution looks across all CVEs that have a value that day. This gives a sense of how spread out the scores are at a single time point.  

2. **Global Stats for Each CVE**  
   - For each CVE, compute descriptive statistics: mean EPSS, standard deviation, min, max, median, number of observations.  
   - Then you can plot the distribution of these summary stats across **all** CVEs, e.g.:
     - Distribution of mean EPSS (how many CVEs have average probabilities near 0, or near 1, or in the middle?).  
     - Distribution of standard deviations (how volatile are these time series in general?).  

3. **Density Plots**  
   - For each day, you could create or examine a distribution (e.g., a histogram or kernel density) of EPSS scores across CVEs.  
   - Or, for each CVE’s entire time series, plot that distribution. Then see how those distributions differ across all CVEs.  

4. **Time Aggregation**  
   - Maybe group by month or week to see if there’s a global trend in typical EPSS scores across *all* CVEs.  

---

### 3.3 Step 3: Explore Common Patterns Over Time

Even though you have many time series, you can still look for *aggregate* time-based patterns:

1. **Global Average Over Time**  
   - For each date (where data is available), calculate the mean (or median) EPSS across all CVEs reported that day.  
   - Plot that average over time to see if, globally, the system sees probabilities going up or down.  

2. **Segment by Clusters or Subgroups**  
   - Possibly group CVEs by some criterion (e.g., severity rating, year of disclosure) and plot their *average* or *median* EPSS over time.  
   - Helps see if certain categories of CVEs have different typical probability trajectories.  

3. **Detect Outlier CVEs**  
   - Some CVEs might have extremely high or extremely volatile EPSS scores.  
   - Could do a simple outlier detection on the time-series statistics (like mean or max).  

4. **Missing Data Over Time**  
   - Investigate if there are certain time ranges with more missing data.  

---

### 3.4 Step 4: Spot-Check a Few Random Time Series

1. **Random Subset**  
   - Select a random sample of (say) 20 CVEs, plot their time series on a small multiple of line plots.  
   - This gives you a “human” sense of how some CVEs behave.  
2. **Representative Prototypes**  
   - You could also cluster CVEs on time-series shape (using, e.g., dynamic time warping or a simpler measure) and then plot the cluster centroids.  
   - This helps you see “typical” patterns among the thousands of time series.  

**Why do this?** Because aggregated statistics can hide important peculiarities. By visually examining a few examples, you glean how the data truly looks.  

---

### 3.5 Step 5: Investigate Seasonality or Trends if Relevant

1. **Global Trend**  
   - As mentioned, plot daily average EPSS across all CVEs: is there an upward or downward drift across months/years?  
2. **Weekly/Monthly Seasonality**  
   - If relevant (maybe new disclosures tend to come on certain days?), see if EPSS systematically changes by day of the week or time of month.  
3. **Long-Term Patterns**  
   - Are older CVEs systematically lower in EPSS (maybe because exploitation is less likely once the patch is widely available)? Or do certain older CVEs remain high risk?  

---

### 3.6 Step 6: Summarize Findings and Hypotheses

The final step of EDA is to interpret your results:

- **Distribution Observations**: e.g., “Most CVEs have an average EPSS below 0.2, but about 10% are consistently above 0.5.”  
- **Trend Observations**: e.g., “There seems to be no strong global trend, but some subgroups show high volatility.”  
- **Seasonality or Events**: e.g., “High spikes typically align with major exploit kits or vendor advisories.”  
- **Data Quality**: e.g., “Missing data is high in mid-2021 for many CVEs—possible data collection issue.”  

From these observations, you can form questions or next steps like:
- “Should we group CVEs by certain classes or families for better modeling?”  
- “Do we need to address missing data in certain time periods?”  
- “Do we see stable or wildly fluctuating probabilities?”  

---

## 4. Is This EDA Approach Common? Examples & Best Practices

### 4.1 Common Problems Like This

1. **Financial Sector**  
   - Probability of default across thousands of loans over time. Each loan has a time series of probabilities that may shift based on new information.  
2. **Marketing or E-commerce**  
   - Probability of conversion for large sets of users, each day.  
3. **Weather Forecasting**  
   - Probability of rain for many geographic locations over time.  

In each of these scenarios, EDA typically involves:
- Aggregating across entities to see overall trends.  
- Examining distributions of key statistics (mean, variance) across entities.  
- Checking for missing data patterns.  
- Potentially grouping or clustering.  
- Doing some raw, random inspection of individual series.  

### 4.2 Best Practices

- **Keep Plots Interpretable**: Because you have so much data, rely on aggregated or sample-based visualizations to avoid chart clutter.  
- **Document Missing Data**: Many real-world multi-entity time series have irregularities. Knowing where and how data is missing is crucial for any subsequent analysis.  
- **Ensure Boundedness**: If you see values outside \([0, 1]\) due to data corruption, fix or remove them.  
- **Look for Potential Data Quality Issues**: Are there improbable spikes or constant 0’s for long stretches?  

---

## 5. Why This Prepares You for the Rest of the Process

A thorough EDA:
1. **Guides Data Cleaning**: You learn about missing data, outliers, anomalies.  
2. **Informs Feature Engineering**: If you see strong weekly patterns, you might add a “day-of-week” feature.  
3. **Helps Model Selection**: If you see that 80% of CVEs have near-constant low probabilities, you might consider modeling approaches that handle near-zero or near-one inflation.  
4. **Clarifies the Problem Definition**: You might realize your data distribution calls for certain evaluation metrics (like Brier Score if most EPSS are near zero).  

---

## 6. Step-by-Step Summary of an Ideal EDA Flow

1. **Gather the Data**: Multi-entity, multi-time-series (CVE, date, EPSS).  
2. **Basic Structuring**: Count total records, CVEs, date range, missing records.  
3. **Global Aggregates**:  
   - Distribution of EPSS across all CVEs for each date (if feasible).  
   - Distribution of each CVE’s summary stats (mean, std, etc.).  
4. **Time Evolution (Global)**: Plot average or median EPSS across all CVEs vs. time.  
5. **Time Evolution (Subgroups)**:  
   - Possibly group by CVE characteristics or cluster them.  
   - Visualize average EPSS for those groups over time.  
6. **Sampling**: Take a random sample of CVEs (e.g., 20–50) to see actual line plots.  
7. **Check for Seasonality & Trends**: If daily data might be cyclical, verify any weekly or monthly pattern.  
8. **Document Findings & Hypotheses**: Summarize what stands out, note anomalies or data quality issues.  

This approach is both **common** and **best practice** for large-scale time-series EDA. It sets the stage for any advanced modeling or forecasting you’ll do later on.

---

## 7. Conclusion

- **For a Single CVE**: EDA is straightforward—plot, check distribution, look for trends and stationarity, do outlier checks.  
- **For Hundreds of Thousands of CVEs**: You need an aggregate, scalable approach:
  - Summaries or distributions of statistics across CVEs.  
  - Global trends, sub-group analysis, random sampling for in-depth inspection.  
- **Why EDA Is Critical**: It reveals the data’s structure, helps identify anomalies, and shapes subsequent decisions about how to handle missing data, how to segment CVEs, and how to model the time series (if you later choose a particular architecture or method).  

Ultimately, a rigorous EDA process gives you the **clearest possible view** of how your EPSS scores behave across time and across CVEs—ensuring you can ask the right questions and make the best decisions for your forecasting or analytics application down the line.