# Models Overview

### xgboost

XGBoost uses gradient boosted trees which naturally account for non-linear relationships between features and the target variable, as well as accommodating complex interactions between features.

### KNN 

* Scale features because in K-nearest neighbors (KNN) with a Euclidean distance measure, the model is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.

### K-Means

* K-Means uses the Euclidean distance measure here so feature scaling matters.

### PCA 

* Scaling is critical while performing Principal Component Analysis(PCA). PCA tries to get the features with maximum variance, and the variance is high for high magnitude features and skews the PCA towards high magnitude features.

### Gradient Descent

* We can speed up gradient descent by scaling because θ descends quickly on small ranges and slowly on large ranges, and oscillates inefficiently down to the optimum when the variables are very uneven.