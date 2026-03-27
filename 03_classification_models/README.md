# Classification Models: From Loan Approvals to Auction Predictions

**Techniques:** Logistic regression, linear probability model, confusion matrix, marginal effects, multinomial logit, cross-elasticities, decision trees, pruning, random forests, bagging, variable importance
**Datasets:** LoanData.csv (bank customers), CommuteMode.csv (transportation choices), Auctions.csv (eBay listings)
**Full Code:** [03_classification_models.Rmd](03_classification_models.Rmd)

---

## The Business Problems

This project tackles four classification tasks, each using a different technique and a different business context:

1. **Loan approvals** — A bank wants to predict which customers will accept a personal loan offer, so marketing dollars can be targeted at high-probability customers rather than spent on blanket campaigns.
2. **Commuter mode choice** — A city transportation authority wants to understand what drives commuters to choose car vs. bus vs. train, and how changes in one mode's travel time shift demand to other modes.
3. **eBay auction competitiveness** — A seller wants to predict whether an auction will attract competitive bidding based on listing characteristics like opening price, duration, and category.
4. **Ensemble methods** — The single decision tree from problem 3 is compared against random forests and bagging to measure the accuracy improvement from combining many trees.

---

## Part 1: Logistic Regression for Loan Prediction

### Why Not Just Use Linear Regression?

A natural first instinct is to run ordinary linear regression on a binary outcome (1 = accepted loan, 0 = declined). This is called the **Linear Probability Model (LPM)**. The coefficients are easy to interpret — "a one-unit increase in Income increases the probability of acceptance by X."

The problem shows up when the predicted values are examined:

```r
head(sort(predict(lpm.res)), 10)  # some predictions below 0
tail(sort(predict(lpm.res)), 10)  # some predictions above 1
```

The LPM produces predicted probabilities below 0 and above 1, which are meaningless. This is the core motivation for logistic regression, which constrains all predictions to the valid [0, 1] range using the logistic function.

### Evaluating the Model

A confusion matrix compares predicted classifications against actual outcomes. Three metrics are computed: overall accuracy, specificity (correctly identifying non-acceptors), and sensitivity (correctly identifying acceptors).

```r
confusion <- table(yhat, mydata$Loan)
sum(diag(confusion)) / sum(confusion)   # accuracy
confusion[1, 1] / sum(confusion[, 1])   # specificity
confusion[2, 2] / sum(confusion[, 2])   # sensitivity
```

### Making It Actionable: Marginal Effects

Raw logistic regression coefficients are in log-odds, which are not directly useful to a business stakeholder. Marginal effects translate them to the probability scale: "a one-unit change in Income changes the predicted probability of loan acceptance by Y percentage points."

The key difference from the LPM: in logistic regression, marginal effects are not constant. They depend on where the customer sits on the probability curve. At the extremes (very high or very low probability), changes in predictors have less impact than in the middle.

```r
PE.logit <- dlogis(xb) * (coef(logit.res)[-1])
```

---

## Part 2: Multinomial Logit for Transportation Mode Choice

### Beyond Binary: Multiple Unordered Choices

When the outcome has more than two categories (car, bus, train, etc.), binary logistic regression does not apply. The **multinomial logit** model extends logistic regression to handle multiple unordered choices.

```r
m1.res <- mlogit(choice ~ cost + time, Commutemode)
```

Both cost and time have negative coefficients — higher cost or longer travel time makes a mode less attractive. This is consistent with how commuters actually behave.

### Cross-Elasticities: The Policy Tool

The most useful output is the cross-elasticity matrix. Each column shows: if that mode's travel time increases by one minute, how does the probability of choosing *every* mode change?

```r
effects(m1.res, covariate = "time", data = xval)
```

The diagonal elements are negative (a mode's own demand drops when its travel time increases). The off-diagonal elements are positive (competing modes pick up the displaced commuters). A transit authority could use this matrix to estimate the ridership impact of reducing bus travel time by 5 minutes before committing resources to the project.

---

## Part 3: Decision Trees for eBay Auctions

### Transparent, Rule-Based Predictions

Decision trees split the data at each node using the feature and threshold that best separates the classes. The result is a flowchart that any stakeholder can follow from top to bottom to understand exactly why the model predicts competitive or non-competitive.

```r
tree.default <- rpart(Competitive ~ ., data = train.df, method = "class")
rpart.plot(tree.default, extra = 1)
```

### The Overfitting Problem and How Pruning Solves It

A tree with no constraints will keep splitting until every training observation is perfectly classified. This memorizes the noise in the training data and performs poorly on new observations. The fix:

1. **Grow a large tree** with no complexity penalty (`cp = 0`)
2. **Run 10-fold cross-validation** to measure error at each complexity level
3. **Prune back** to the cp value that minimizes cross-validated error

```r
tree.large  <- rpart(Competitive ~ ., data = train.df, method = "class",
                     cp = 0, minsplit = 20, xval = 10)
tree.pruned <- prune(tree.large, cp = 0.006)
```

The pruned tree achieves comparable (or better) test-set accuracy with far fewer splits. It generalizes better because it is not fitting to noise. This is the bias-variance tradeoff in action: a simpler model has slightly more bias but substantially less variance.

---

## Part 4: Random Forests & Bagging

### From One Tree to a Thousand

Single decision trees are interpretable but unstable — small changes in the training data can produce a completely different tree. Ensemble methods fix this by building many trees and averaging their predictions.

Two approaches are compared, each with 1,000 trees:

- **Bagging** (`mtry = 5`): Every split considers all 5 predictors. The only source of diversity between trees is the bootstrapped training samples.
- **Random Forest** (`mtry = 2`): Each split considers only 2 randomly chosen predictors. This decorrelates the trees from each other, reducing the variance of the combined ensemble.

```r
bag.res <- randomForest(Competitive ~ ., data = train.df, xtest = x_test,
                        ytest = y_test, mtry = 5, ntree = 1000, importance = TRUE)
rf.res  <- randomForest(Competitive ~ ., data = train.df, xtest = x_test,
                        ytest = y_test, mtry = 2, ntree = 1000, importance = TRUE)
```

### Error Convergence

The error curves (OOB and test-set, for both models) are plotted together. Both models converge after roughly 200-300 trees. The OOB error tracks the test error closely, confirming that the out-of-bag estimate is a reliable proxy for generalization performance even without a separate validation set.

### Which Features Matter Most?

The variable importance plot ranks predictors by their Mean Decrease in Gini index — how much each variable contributes to reducing classification impurity across all 1,000 trees. This provides clear guidance on which listing attributes have the largest impact on whether an auction becomes competitive.

```r
varImpPlot(rf.res, type = 2)
```

---

## Key Takeaways

| Concept | What It Means in Practice |
|---------|--------------------------|
| LPM vs. logistic regression | The LPM is a quick baseline, but logistic regression is the right tool — it keeps predicted probabilities between 0 and 1 |
| Marginal effects | Translate log-odds coefficients into probability-scale changes that business stakeholders can act on |
| Multinomial logit | Handles multi-class choices; the cross-elasticity matrix is directly useful for policy analysis |
| Decision trees | Transparent and interpretable, but prone to overfitting without pruning |
| Pruning via cross-validation | Finds the right model complexity that balances training fit and generalization |
| Random forests vs. bagging | Random predictor subsets at each split decorrelate the trees, reducing ensemble variance |
| Variable importance | Compensates for the loss of interpretability in ensemble methods by ranking which features drive predictions |

---

## R Packages

- **Base R** — data manipulation, plotting, logistic regression
- **`mlogit`** — multinomial logit modeling of discrete choices
- **`rpart`** and **`rpart.plot`** — decision tree construction and visualization
- **`randomForest`** — random forest and bagging ensemble models
