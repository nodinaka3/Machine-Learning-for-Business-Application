# Customer Segmentation & Recommendation Engines

**Techniques:** K-means clustering, hierarchical clustering, elbow method, user-based collaborative filtering (UBCF)
**Datasets:** ShoppingVisits.csv (100 retail customers), MovieLens (100K ratings, built into `recommenderlab`)
**Full Code:** [01_unsupervised_learning.Rmd](01_unsupervised_learning.Rmd)

---

## The Business Problem

A retailer collects two pieces of information about each customer: how often they visit and how much they spend per visit. The question is whether these customers fall into natural groups that the marketing team can target differently. Separately, a movie streaming platform wants to recommend films to a new user who has only rated 15 movies so far.

Both problems fall under **unsupervised learning** — there are no labels, no "right answers" in the data. The algorithms have to discover structure on their own.

---

## Part 1: Customer Segmentation with Clustering

### Why Scaling Matters

K-means clustering assigns customers to groups by minimizing the distance between each point and its cluster center. The catch is that distance depends on scale. If one variable ranges from 0 to 500 (dollars spent) and another ranges from 1 to 20 (visits), the dollar variable will dominate every distance calculation. Visit frequency barely registers.

The analysis runs K-means twice — once on raw data and once on standardized data (mean = 0, standard deviation = 1) — and compares the results side by side:

```r
km3_raw <- kmeans(mydata, 3, nstart = 20)
km3_sc  <- kmeans(mydata_sc, 3, nstart = 20)
```

The difference is significant. Without scaling, the clusters split customers almost entirely by spending. With scaling, both variables contribute equally, and the segments reflect differences in *both* shopping frequency and spending behavior. This is a general rule: distance-based algorithms should always run on standardized data.

### Choosing the Right Number of Clusters

How many segments actually exist? The analysis tests k = 2, 3, 4, and 5, then uses the **elbow chart** to make the call. The elbow chart plots total within-cluster sum of squares against the number of clusters. The "elbow" — the point where adding more clusters stops producing meaningful improvement — lands at **k = 3**.

```r
plot(ss.vec, type = "b", xlab = "Number of Clusters", ylab = "Total within-Cluster ss",
     main = "Elbow cluster chart")
```

### Validating with Hierarchical Clustering

K-means requires picking k upfront. Hierarchical clustering takes a different approach: it builds a tree (dendrogram) showing how observations merge step by step, and the tree can be cut at any level to produce clusters.

The analysis runs hierarchical clustering with four different linkage methods — average, complete, single, and centroid — and compares the results:

- **Average and complete linkage** produce balanced, interpretable segments
- **Single linkage** exhibits "chaining," where it creates one massive cluster and a few stragglers
- **Centroid linkage** falls somewhere in between

When K-means and hierarchical clustering (average linkage) both produce nearly identical 3-cluster segments, that agreement strengthens the case that these groups represent real patterns in the data, not artifacts of one particular algorithm.

### What the Segments Mean for the Business

With three validated customer segments, the retailer can design targeted strategies:

- **Low-frequency, low-spend:** Re-engagement campaigns, introductory discounts, win-back emails
- **Moderate-frequency, moderate-spend:** The core base. Loyalty programs and bundle deals could push this group toward higher spend per visit
- **High-frequency, high-spend:** The VIP segment. Exclusive offers, early access, and premium service to retain these high-value customers

---

## Part 2: Movie Recommendation Engine

### How User-Based Collaborative Filtering Works

Collaborative filtering powers platforms like Netflix, Amazon, and Spotify. The user-based variant (UBCF) works in four steps:

1. **Similarity calculation** — Compare the new user's ratings against every existing user using cosine similarity or Pearson correlation
2. **Neighbor selection** — Identify the k most similar users
3. **Prediction** — For each unrated movie, compute a weighted average of the neighbors' ratings
4. **Ranking** — Sort by predicted rating and return the top-N recommendations

### Building the Recommender

A new user profile is created with just 15 movie ratings (scale of 1-5). The recommender trains on the full MovieLens dataset (943 users, 1,664 movies, 100K ratings), finds the most similar users, and generates a personalized top-10 recommendation list:

```r
rec.ub  <- Recommender(MovieLense, "UBCF")
pred.ub <- predict(rec.ub, myrating, n = 10, type = "topNList")
as(pred.ub, "list")
```

Even with only 15 ratings from the new user, the algorithm leverages the full 100K-rating dataset to find meaningful patterns. The sparsity of the new user's profile is compensated by the richness of the existing data.

---

## Key Takeaways

| Concept | What It Means in Practice |
|---------|--------------------------|
| Always scale before K-means | Without it, the variable with the largest range dominates the clustering |
| Use multiple methods and compare | When K-means and hierarchical clustering agree, the segments are more credible |
| The elbow chart picks k | Look for the bend where additional clusters stop adding value |
| Linkage method matters | Average and complete linkage produce balanced clusters; single linkage chains |
| UBCF works with sparse data | 15 ratings from a new user can drive a personalized recommendation list when the existing dataset is rich enough |

---

## R Packages

- **Base R** — data manipulation, plotting, K-means, hierarchical clustering
- **`recommenderlab`** — user-based collaborative filtering on MovieLens data
