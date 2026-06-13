# Dataset Analysis & Model Performance Report
**Amazon Delivery Time Prediction (V3)**

This report provides a detailed analysis of the Amazon Delivery dataset (`amazon_delivery.csv`), data quality issues identified, feature engineering methodologies, and the performance outcomes of our model iteration (V3 vs. baseline V2).

---

## 1. Executive Summary

By systematically addressing data quality issues and reversing the premature exclusion of key features, we achieved a massive leap in prediction accuracy. 

* **Baseline V2 Stacking Model RMSE:** `40.99`
* **V3 Gradient Boosting Model RMSE:** `22.9250` (An absolute reduction of **18.06 minutes** in average prediction error, representing a **44.1% performance improvement**).
* **V3 Random Forest Regressor RMSE:** `23.5746`

### Key Drivers of Improvement
1. **Feature Restoration:** Restoring `Category` (specifically segmenting `Grocery`), `Agent_Rating`, and `Agent_Age` which were previously dropped.
2. **Simpson's Paradox Resolution:** Identifying and adjusting for the confounding effect of delivery categories on distance metrics.
3. **Data Quality Corrections:** Cleaning invalid `(0,0)` GPS coordinate sentinels and negative latitudes that were corrupting distance metrics.
4. **Geographically Accurate Distances:** Replacing flat Euclidean distance with spherical `Haversine` and grid-based `Manhattan` distances.

---

## 2. Dataset Overview & Demographics

The dataset comprises **43,739 rows** and **16 columns**, mapping order profiles, delivery agent demographics, atmospheric/traffic conditions, and spatial coordinates to the target variable `Delivery_Time`.

### Target Variable: `Delivery_Time`
* **Mean:** 124.91 minutes
* **Median:** 125.00 minutes
* **Standard Deviation:** 51.92 minutes
* **Range:** 10 to 270 minutes
* **Distribution Distribution:** Relatively flat and symmetric between 60 and 200 minutes, with a sharp drop outside this range. It exhibits low skewness (`0.1886`) and slightly negative excess kurtosis (`-0.2608`).

| Delivery Time Bin (min) | Order Count | Percentage |
|-------------------------|-------------|------------|
| **0–30**                | 1,852       | 4.2%       |
| **31–60**               | 2,873       | 6.6%       |
| **61–90**               | 7,572       | 17.3%      |
| **91–120**              | 9,050       | 20.7%      |
| **121–150**             | 10,197      | 23.3%      |
| **151–200**             | 8,584       | 19.6%      |
| **201–300**             | 3,611       | 8.3%       |

---

## 3. Critical Data Quality Discoveries & Imputations

Several subtle anomalies in the raw data would have severely degraded model performance if left unaddressed:

### A. Invalid Coordinate Sentinels (0.0, 0.0)
* **Anomaly:** **3,505 rows (8.0% of the dataset)** contained a store latitude and longitude of exactly `(0.0, 0.0)`.
* **Impact:** Calculating distances from `(0,0)` to delivery locations in India (typically lat ~10-30, lon ~70-90) yields massive, artificial distances of over 7,000+ km, ruining the distance feature's scale and prediction signals.
* **Correction:** Masked `(0.0, 0.0)` coordinates as `NaN` and imputed them using the **median coordinates of the corresponding delivery `Area`** (e.g. Metropolitan, Urban, Rural).

### B. Negative Latitude Sign Corruptions
* **Anomaly:** **188 rows** had negative Store Latitudes (e.g. `-30.90` instead of `30.90`).
* **Impact:** Placing stores in the southern hemisphere (e.g., Indian Ocean / Antarctica) distorted distance calculations.
* **Correction:** Applied the absolute value operator to all latitude columns to flip the sign back to positive.

### C. Missing Values & Time Formats
* **Agent_Rating:** 54 missing rows (imputed with the median value of `4.7`).
* **Weather:** 91 missing rows (rows dropped during final cleaning).
* **Order_Time / Pickup_Time:** Contained literal `"NaN "` strings which caused date-time parsing failures. Coerced these using formatting robust, midnight-aware calculations to handle overnight crossing.

---

## 4. Key Analytical Insights

### A. Simpson's Paradox in Distance vs. Delivery Time

One of the most fascinating statistical discoveries in this dataset is the correlation between delivery distance and delivery time:

> [!IMPORTANT]
> * **Global Correlation (All Items):** `-0.0024` (Indicates distance has absolutely zero impact on delivery time, which is physically counter-intuitive).
> * **Grocery-Only Correlation:** `+0.3091`
> * **Non-Grocery Correlation:** `+0.3092`

#### Why does this happen?
This is a classic case of **Simpson's Paradox**, driven by a single confounding variable: **Category (specifically, Grocery vs. Non-Grocery)**.

1. **Distance Distributions are Identical:** The average delivery distance for Grocery items is **9.56 km**, and for Non-Grocery items it is **9.67 km**. Both groups have a median distance of **9.17 km**.
2. **Delivery Times are Radically Shifted:** 
   * **Grocery:** Mean delivery time of **26.54 minutes** (median = 26.0 min).
   * **Non-Grocery:** Mean delivery time of **131.38 minutes** (median = 130.0 min).
3. **The Confounding Shift:** Because Grocery orders are prioritized (instant delivery) and delivered ~105 minutes faster than standard packages for the same distance, mixing these two populations hides the positive correlation. In a scatter plot, they appear as two distinct horizontal bands. When combined, the flat global correlation mask the strong positive relationship that exists *within* each class.

```
Delivery Time (min)
  ^
  |   ====================  Non-Grocery (Mean: 131m, Corr: +0.31)
  |
  |   --------------------  Combined (Corr: -0.00)
  |
  |   ====================  Grocery (Mean: 26m, Corr: +0.31)
  +---------------------------> Distance (km)
```

---

### B. Agent Demographic Influences
* **Agent_Rating (Correlation: -0.2926):** Strong negative correlation. Highly rated agents complete deliveries significantly faster.
* **Agent_Age (Correlation: +0.2555):** Strong positive correlation. Younger agents complete deliveries faster on average.

---

## 5. Feature Engineering Highlights

To maximize the predictive capacity of tree-based ensembles and Deep Neural Networks, we engineered several high-utility features:

1. **Haversine Distance:** Calculates the spherical great-circle distance between store and drop points (in kilometers), correcting the geometric distortion of V2's Euclidean calculations.
2. **Manhattan Distance:** Approximates grid-routing distance on urban street grids:
   $$D_{\text{manhattan}} = R \cdot |\Delta \text{lat}| + R \cdot \cos(\text{mean\_lat}) \cdot |\Delta \text{lon}|$$
3. **Time To Pickup:** Computes the delta in minutes between order placement and agent pickup. Corrected for overnight orders where the pickup occurs past midnight (negative delta resolved by adding 1440 minutes).
4. **Temporal Flags:** Created binary indicators to capture localized congestion patterns:
   * `Is_Rush_Hour`: Hour of day falls in `[8-10 AM, 5-8 PM]`.
   * `Is_Late_Night`: Hour of day falls in `[11 PM - 4 AM]`.
   * `Is_Weekend`: Day of week is Saturday or Sunday.

---

## 6. Model Evaluation & Comparison

Restoring dropped columns and correcting coordinate data dramatically boosted model power. The Gradient Boosting model, which natively handles non-linear relationships and feature interactions, emerged as the superior model.

### Performance Summary Table

| Model | RMSE (min) | MAE (min) | $R^2$ Score | Status / Notes |
|-------|------------|-----------|-------------|----------------|
| **V1 Baseline Model** | 44.9004 | - | - | Simple single-tree regressor |
| **V2 Stacking Ensemble** | 40.9932 | - | - | DNN + XGBoost (Dropped key cols, Euclidean distance) |
| **V3 RandomForest** | 23.5746 | 18.0396 | 0.7953 | Baseline tree ensemble |
| **V3 Gradient Boosting** | **22.9250** | **17.8151** | **0.8065** | **Optimal Single Model** (Hyperparameter Tuned) |
| **V3 Stacking (RF + GB + DNN)** | *Pending* | *Pending* | *Pending* | Currently training in the cloud |

### Feature Importance (V3 Gradient Boosting)
The relative importance of the top predictors in our Gradient Boosting model aligns perfectly with our analytical findings:

1. **Category_Encoded (`29.87%`):** By far the most crucial feature, distinguishing the fast-delivery Grocery segment from standard deliveries.
2. **Agent_Rating (`18.05%`):** Agent performance score.
3. **Traffic_Low (`9.82%`):** Presence of low traffic conditions.
4. **Agent_Age (`8.50%`):** Agent demographic profile.
5. **Weather_Sunny (`6.03%`):** Favorable weather indicator.
6. **Manhattan_Distance (`5.22%`):** Grid distance in km.
7. **Haversine_Distance (`5.08%`):** Great-circle distance in km.
8. **Weather_Cloudy (`4.09%`):** Moderate weather indicator.
9. **Weather_Fog (`4.00%`):** Unfavorable weather condition.
10. **Traffic_Jam (`3.06%`):** Extreme traffic delay indicator.

---

## 7. Conclusions & Strategic Recommendations

1. **Keep Segmenting Categories:** Since `Grocery` behaves so differently from items like `Cosmetics` or `Electronics`, we should explore training two separate sub-models (a dedicated Grocery model and a standard delivery model) if we want to squeeze out even higher accuracy.
2. **Prioritize Agent Performance Data:** Agent Rating and Age carry substantial predictive weight. Operational pipelines should ensure real-time delivery estimations are closely tied to the assigned agent's historic stats.
3. **Capture Traffic Dynamics:** Distance features are useful, but Traffic conditions (`Traffic_Low`, `Traffic_Jam`) combined account for more than 12% of feature importance, suggesting routing duration is heavily traffic-constrained rather than purely distance-constrained.
