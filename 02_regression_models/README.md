# Predicting Used Car Prices with Linear & Non-Linear Regression

**Techniques:** Multiple linear regression, VIF/multicollinearity diagnostics, polynomial regression, interaction terms, qualitative predictors (dummy variables)
**Datasets:** UsedCars.csv (Toyota Corolla listings, 9 features), UsedCars2_NonLinear.csv (extended with Color)
**Full Code:** [02_regression_models.Rmd](02_regression_models.Rmd)

---

## The Business Problem

A used car dealership needs a pricing model. Which features drive price the most? How much does a car depreciate per year? Per 10,000 kilometers? Does car color affect resale value? And is the relationship between mileage and price actually a straight line, or is something more complex going on?

These are the kinds of questions that regression analysis is built for. The project starts with a standard multiple linear regression and progressively adds complexity — qualitative predictors, interaction terms, and polynomial regression — to capture patterns that a simple linear model misses.

---

## Part 1: Multiple Linear Regression

### Building the Full Model

The first model includes all 9 available predictors: Age (months), KM (kilometers driven), HP (horsepower), Metallic (paint type), Automatic (transmission), CC (engine displacement), Doors, Gears, and Weight.

```r
lm.res <- lm(Price ~ Age + KM + HP + Metallic + Automatic + CC + Doors + Gears + Weight,
              data = mydata)
```

### Understanding What the Numbers Mean

Rather than just reading the significance stars from the summary output, the analysis walks through the mechanics of how t-statistics, p-values, and R-squared are actually computed:

- **t-statistic** = coefficient estimate / standard error. A large absolute t-value means the estimate is far from zero relative to its uncertainty.
- **p-value** = the probability of observing a t-statistic this extreme if the true coefficient were zero. Small p-values are evidence that the variable matters.
- **R-squared** = the share of price variation explained by the model. It can be computed three different ways, all yielding the same result.

```r
bhat  <- lm.sum$coefficients[, 1]
se    <- lm.sum$coefficients[, 2]
tstat <- bhat / se
```

This manual reproduction confirms the model explains a large share of the variation in used car prices.

### Detecting Multicollinearity with VIF

When predictor variables are correlated with each other, regression coefficients become unstable. The Variance Inflation Factor (VIF) quantifies this. A VIF above 5-10 is a red flag.

```r
library(car)
vif(lm.res)
```

Weight shows elevated VIF values, which makes sense — heavier cars tend to have larger engines (CC) and more doors. To confirm, Weight is regressed on all other predictors. The high R-squared from that regression confirms the redundancy. The individual coefficient on Weight should be interpreted with caution, even though the overall model fit looks fine.

### Full Model vs. Reduced Model

Three predictors (Metallic, CC, Doors) were not statistically significant. Dropping them produces a 6-predictor model:

```r
lm.res3 <- lm(Price ~ Age + KM + HP + Automatic + Gears + Weight, data = mydata)
```

The reduced model has nearly the same R-squared, and the adjusted R-squared actually ticks up slightly. The dropped variables were adding noise, not signal. A simpler model that explains the same variation is always preferred — this is the principle of parsimony.

### Answering the Business Questions

**How much does price drop per year of age?**
Age in the dataset is measured in months, so the coefficient is multiplied by 12. Holding everything else constant, each additional year reduces the car's price by a specific dollar amount that a dealership can plug directly into pricing decisions.

**How much does price drop per 10,000 additional kilometers?**
The KM coefficient, multiplied by 10,000, gives the depreciation per 10,000 km driven. These are the kinds of actionable numbers a dealership pricing team or a consumer doing comparison shopping would need.

---

## Part 2: Non-Linear Regression

### Why Go Beyond Linear?

A linear model assumes a straight-line relationship between each predictor and price. In reality, some relationships curve. The first 50,000 km on a car might depreciate its value much faster than going from 150,000 to 200,000 km — the steep part of the depreciation curve has already passed. A straight-line model would miss this entirely.

### Does Color Affect Resale Value?

The extended dataset adds car Color as a categorical variable. When a factor is included in a regression, R automatically creates dummy variables — one for each color level minus a reference category:

```r
lm1 <- lm(Price ~ Age + KM + HP + Automatic + Gears + Weight + Color, data = mydata)
contrasts(Color)
```

Several colors show statistically significant coefficients. This aligns with market reality: black and silver cars tend to hold value better because they have broader buyer appeal.

### The Age-Mileage Interaction

An interaction term between Age and KM tests whether the effect of mileage on price changes depending on the car's age:

```r
lm2 <- lm(Price ~ Age * KM + HP + Automatic + Gears + Weight, data = mydata)
```

A significant Age:KM interaction means the marginal effect of mileage is not constant. A high-mileage car that is relatively new is evaluated differently than a high-mileage car that is already old. Without this interaction, the model would treat the KM effect as identical regardless of age — an oversimplification.

### Polynomial Regression: Capturing the Curve

A scatter plot of KM vs. Price shows the relationship is not linear. There is a steep drop at low mileage that flattens out at higher mileage. A 4th-degree polynomial captures this shape:

```r
fit1 <- lm(Price ~ poly(KM, 4, raw = TRUE) + Automatic, data = mydata)
```

The resulting curve tracks the steep initial depreciation and the flattening at higher mileage. A dealership relying on a linear model would systematically overprice low-mileage cars and underprice mid-range ones.

---

## Key Takeaways

| Concept | What It Means in Practice |
|---------|--------------------------|
| Start full, then simplify | Build with all predictors, drop the non-significant ones, and confirm the simpler model performs just as well |
| VIF catches hidden redundancy | Even when overall R-squared looks fine, correlated predictors make individual coefficients unreliable |
| Interaction terms matter | The effect of mileage on price depends on the car's age — a model without that interaction oversimplifies |
| Polynomial regression captures curves | Price-mileage depreciation is steep early and flattens later. A linear model misprices cars at both ends |
| Qualitative predictors have real effects | Car color affects resale value. R handles the dummy encoding, but understanding the reference category is necessary for correct interpretation |

---

## R Packages

- **Base R** — data manipulation, plotting, linear regression, polynomial regression
- **`car`** — Variance Inflation Factor (VIF) analysis for multicollinearity diagnostics
