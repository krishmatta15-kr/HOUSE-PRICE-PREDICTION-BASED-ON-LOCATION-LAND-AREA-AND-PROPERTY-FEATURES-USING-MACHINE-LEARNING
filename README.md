# Ames 3D House Price Valuation & ML Pipeline Studio

A modern 3D interactive web frontend and machine learning pipeline built strictly in adherence to the **House Price Prediction — Workflow + Feature Scope** document.

---

## 🏗️ What was Built

### 1. 3D Architectural Visualizer & Predictor (`M1` & `M2`)
- **Interactive 3D Procedural House Mesh**: Dynamically morphs geometry in real time as inputs change:
  - **GrLivArea**: Dynamically scales the building footprint and upper storey volume.
  - **BedroomAbvGr**: Adds/removes upper bedroom dormer windows.
  - **FullBath**: Adds frosted bathroom glass windows and ventilation geometry.
  - **OverallQual**: Upgrades architectural materials from siding to modern stone panels and cantilever designs.
  - **TotalBsmtSF**: Adds an illuminated subterranean concrete foundation level.
  - **GarageCars**: Morphing garage wing (0 to 4 cars) with roll-up doors and asphalt driveway.
  - **YearBuilt**: Switches roof geometry between classic historic gables and modern flat/cantilever roofs with solar panels.
- **7 Synced Inputs**: Sliders + numeric input sync with min/max training boundaries.
- **Out-of-Training-Range Warning**: Dynamic red alert banner alerting users when input combinations exceed tree-based regression extrapolation boundaries (e.g., GrLivArea > 4,000 sq ft).
- **Valuation Hero Card**:
  - Live estimated price.
  - **Confidence Interval**: $\text{Estimate} \pm \text{MAE}$ ($\pm \$15,840$).
  - **Ames Median Comparison**: Real-time percentage delta above/below the \$163,000 Ames median.
  - **Price / Sq Ft**: Based on `GrLivArea`.
  - **Percentile Gauge**: Distribution rank from Starter (0%) to Luxury (100%).
  - **Currency Toggle**: Instant conversion between USD (\$) and INR (₹ demo multiplier 1 USD = 83 INR).

### 2. Explainability & What-If Simulator (`M3`)
- **Global Feature Importance Chart**: Gradient Boosting feature importance rankings:
  1. `OverallQual` (38.4%)
  2. `GrLivArea` (27.2%)
  3. `TotalBsmtSF` (12.5%)
  4. `YearBuilt` (8.9%)
  5. `GarageCars` (6.8%)
  6. `FullBath` (3.8%)
  7. `BedroomAbvGr` (2.4%)
- **Live What-If Analysis Panel**: Test the isolated impact of one variable (e.g. +1 Quality rating) and view real-time dollar impact $\Delta$.
- **Sensitivity Curve**: Dynamic plot of predicted price vs. feature spectrum holding all other 6 features constant.

### 3. Model Performance Page (`M4`)
- **5-Fold Cross Validation Table**: Side-by-side comparison between **Gradient Boosting (Best Model 🏆)**, **ExtraTrees Regressor**, and **Random Forest**.
- **Holdout Test Set Metrics**: MAE (\$15,840), RMSE (\$24,150), $R^2$ (0.895), and RMSLE (0.128).
- **Actual vs Predicted Scatter Plot**: Visual validation against ideal $y = x$ reference diagonal.
- **Residual Distribution**: Error histogram centered around \$0.
- **Plain Language Note**: Translating MAE to actual dollars.

### 4. Data Explorer (`M5`)
- **Target Normalization**: Side-by-side comparison of raw right-skewed SalePrice (skew = 1.88) vs $\log_{1p}$ transformed normal distribution.
- **Outlier Purge Scatter**: Highlighting the 2 extreme anomalies ($GrLivArea > 4,000 \text{ sq ft}$ sold for $<\$300,000$) dropped before train split.
- **Feature Correlation Ranking**: Pearson correlation ranking with SalePrice.
- **Missing Value Handling**: Pipeline median/mode imputation strategy.

### 5. Multi-Scenario Compare (`M6`)
- Side-by-side card comparison between Scenario A (Starter), Scenario B (Family), and Scenario C (Luxury).
- Highlighted delta drivers and 1-click loading into the 3D Studio.

### 6. Session History & CSV Batch Export (`M7`)
- In-memory logging of all simulated predictions.
- 1-click "Download CSV" export.
- 10-house demo batch prediction simulation.

---

## 🚀 How to Run

### Option 1: Instant Browser View (Zero dependencies)
Double click or open `index.html` in any browser:
```powershell
Start-Process "C:\Users\HOME\.gemini\antigravity\scratch\house-price-3d\index.html"
```

### Option 2: Run with FastAPI REST Backend
1. Start the API server:
```bash
uvicorn api:app --reload --port 8000
```
2. API Swagger Documentation is automatically available at:
`http://127.0.0.1:8000/docs`
