# Machine Learning for Business Analytics (R)

**Author:** Nna-dozie Odinaka
**Program:** MBA, Georgia Institute of Technology (Georgia Tech)
**Language:** R

---

## What This Repository Is About

This repository is a collection of applied machine learning projects built during my MBA program at Georgia Institute of Technology. Each project takes a real business problem, applies a machine learning technique in R, and interprets the results in a way that business decision-makers can act on.

The work covers both **supervised learning** (regression, classification, decision trees, random forests) and **unsupervised learning** (clustering, recommendation systems). Every project includes the R code, the dataset, detailed explanations of the methodology, and a discussion of what the results mean for the business.

This portfolio demonstrates two things:

1. Building and evaluating machine learning models in R.
2. Translating technical output into actionable business insights.
3. Applying machine learning techniques to real-world business problems.

---

## Projects

### 1. Unsupervised Learning: Customer Segmentation & Recommendation Engines

**File:** [01_unsupervised_learning.Rmd](01_unsupervised_learning.Rmd)
**Datasets:** ShoppingVisits.csv, MovieLens (built into the `recommenderlab` package)

**Business problem:** A retailer wants to segment its customers into meaningful groups based on visit frequency and spending behavior. Separately, a recommendation engine is built that predicts which movies a new user would enjoy based on the preferences of similar users.

**What was done:**

- K-means clustering on raw vs. scaled data, demonstrating why variable scaling is critical for distance-based algorithms
- Tested k = 2 through 5 and used the Elbow Chart to determine the optimal number of clusters (k = 3)
- Hierarchical clustering with four different linkage methods (average, complete, single, centroid) and compared results via dendrograms
- Side-by-side comparison of K-means vs. hierarchical clustering to validate that the segments are consistent across methods
- User-based collaborative filtering (UBCF) on the MovieLens dataset using the `recommenderlab` package, generating a personalized top-10 movie recommendation list from just 15 user ratings

**Techniques:** K-means, hierarchical clustering, elbow method, dendrogram analysis, user-based collaborative filtering

---

### 2. Regression Analysis: Predicting Used Car Prices

**File:** [02_regression_models.Rmd](02_regression_models.Rmd)
**Datasets:** UsedCars.csv, UsedCars2_NonLinear.csv

**Business problem:** A used car dealership needs a pricing model. Which features drive price the most? How much does a car depreciate per year? Per 10,000 kilometers? Does car color affect resale value? Is the relationship between mileage and price actually linear?

**What was done:**

- Multiple linear regression model with 9 predictors (Age, KM, HP, Metallic, Automatic, CC, Doors, Gears, Weight)
- Manually reproduced t-statistics, p-values, and R-squared to demonstrate understanding of what these metrics compute under the hood
- VIF (Variance Inflation Factor) analysis to detect multicollinearity, confirmed by regressing Weight on all other predictors
- Compared a full model (9 predictors) against a reduced model (6 significant predictors) using R-squared and adjusted R-squared
- Answered specific business questions: price change per year of age, price change per 10,000 km
- Extended the analysis with qualitative predictors (car Color as dummy variables), interaction terms (Age * KM), and polynomial regression (degree 4) to capture the non-linear relationship between mileage and price

**Techniques:** Multiple linear regression, t-tests, VIF/multicollinearity diagnostics, model comparison, qualitative predictors, interaction terms, polynomial regression

---

### 3. Classification Models: From Loan Approvals to Auction Predictions

**File:** [03_classification_models.Rmd](03_classification_models.Rmd)
**Datasets:** LoanData.csv, CommuteMode.csv, Auctions.csv

**Business problem:** Four classification tasks in one project:

1. Predicting which bank customers will accept a personal loan offer
2. Understanding what drives commuters' choice between different transportation modes
3. Predicting whether an eBay auction will attract competitive bidding
4. Comparing ensemble methods (random forest, bagging) against a single decision tree

**What was done:**

- **Logistic Regression:** Compared a Linear Probability Model (LPM) against logistic regression on loan data, showing why the LPM produces impossible probabilities (below 0, above 1). Built a confusion matrix, computed accuracy/sensitivity/specificity, predicted for a specific customer, and computed marginal effects to compare the two models.
- **Multinomial Logit:** Modeled commuter transportation mode choice using the `mlogit` package. Computed predicted choice probabilities and marginal effects showing how a one-minute increase in travel time for one mode shifts commuters toward other modes (cross-elasticities for policy analysis).
- **Decision Trees:** Built a default tree, then grew an unpruned tree (cp=0) and used 10-fold cross-validation to find the optimal pruning point. Compared test-set accuracy between the large and pruned trees to demonstrate the bias-variance tradeoff.
- **Random Forests:** Built both a bagging model (mtry=5, all predictors) and a random forest model (mtry=2, random subset) with 1,000 trees each. Plotted OOB and test error convergence curves and generated variable importance plots to identify which listing features matter most.

**Techniques:** Linear probability model, logistic regression, confusion matrix, marginal effects, multinomial logit, cross-elasticities, decision trees, pruning, cross-validation, bagging, random forests, OOB error, variable importance

---

## Repository Structure

```
ml-business-analytics-R/
├── README.md                          <- You are here
├── .gitignore
├── data/
│   ├── ShoppingVisits.csv             <- Retail customer visit/spend data (100 obs)
│   ├── UsedCars.csv                   <- Toyota Corolla pricing data (9 features)
│   ├── UsedCars2_NonLinear.csv        <- Extended car data with Color variable
│   ├── LoanData.csv                   <- Bank customer loan acceptance data
│   ├── CommuteMode.csv                <- Commuter transportation mode choice data
│   └── Auctions.csv                   <- eBay auction competitiveness data
├── 01_unsupervised_learning.Rmd       <- Clustering + collaborative filtering
├── 02_regression_models.Rmd           <- Linear + non-linear regression
└── 03_classification_models.Rmd       <- Logistic, multinomial, trees, forests
```

---

## Techniques Covered

| Category | Technique | Project |
|----------|-----------|---------|
| Unsupervised | K-means clustering | 01 |
| Unsupervised | Hierarchical clustering (4 linkage methods) | 01 |
| Unsupervised | Elbow method for optimal k | 01 |
| Unsupervised | User-based collaborative filtering (UBCF) | 01 |
| Regression | Multiple linear regression | 02 |
| Regression | VIF / multicollinearity diagnostics | 02 |
| Regression | Polynomial regression | 02 |
| Regression | Interaction terms | 02 |
| Regression | Qualitative predictors (dummy variables) | 02 |
| Classification | Linear probability model | 03 |
| Classification | Logistic regression | 03 |
| Classification | Confusion matrix / accuracy metrics | 03 |
| Classification | Marginal effects | 03 |
| Classification | Multinomial logit | 03 |
| Classification | Cross-elasticities | 03 |
| Classification | Decision trees (CART) | 03 |
| Classification | Pruning via cross-validation | 03 |
| Classification | Bagging | 03 |
| Classification | Random forests | 03 |
| Classification | Variable importance (Gini) | 03 |

---

## R Packages Used

- **Base R** for data manipulation, plotting, and linear/logistic regression
- **`car`** for Variance Inflation Factor (VIF) analysis
- **`recommenderlab`** for user-based collaborative filtering on MovieLens data
- **`mlogit`** for multinomial logit modeling of discrete choices
- **`rpart`** and **`rpart.plot`** for decision tree construction and visualization
- **`randomForest`** for random forest and bagging ensemble models

---

## How to Run

1. Clone this repo: `git clone https://github.com/YOUR_USERNAME/ml-business-analytics-R.git`
2. Open any `.Rmd` file in VS Code or other IDE of choice(with the R extension) or RStudio
3. Install required packages if needed (each .Rmd file includes install checks)
4. Knit to HTML from the terminal:
   ```r
   rmarkdown::render("01_unsupervised_learning.Rmd")
   rmarkdown::render("02_regression_models.Rmd")
   rmarkdown::render("03_classification_models.Rmd")
   ```
5. All data files are in the `data/` folder and relative paths are used throughout, so no `setwd()` calls are needed

---

## About Me

MBA candidate at Georgia Institute of Technology with a focus on analytics and machine learning. This portfolio demonstrates the application of machine learning techniques to business problems using R, from data preparation through model building, evaluation, and interpretation.

Feel free to reach out: nodinaka3@gatech.edu
