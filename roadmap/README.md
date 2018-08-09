

**Exploratory Data Analysis (EDA)**

Check list:

- Many search terms / products appeared several times -> collections.Counter()
- Text similarities are great features.
- Many products don’t have attributes features. Would this be a problem?
- Product ID seems to have strong predictive power. However the overlap of product ID between the training set and the testing set is not very high. Would this contribute to overfitting?
- Check the distribution of features with continuous data.

Approaches:

- Features are linear related -> heat map plot -> Pearson correlation coefficient
- Outliers:
  - scatter plot -> manual remove outliers
  - Use OneClassSVM, EllipticEnvelope, IsolationForest, LocalOutlierFactor from sklearn to identify outliers and remove them.

- Classification -> scatter plot with colour labels
- Feature Selection
- Filters
  -  Correlation Selection -> select high correlation features
  -  Information Gain -> compute node's gini -> gini(parent) - gini(child)
  -  Variance Selection -> sklearn.feature_selection.VarianceThreshold
  -  Chi-squared Test -> select the best features -> sklearn.feature_selection.SelectKBest +
sklearn.feature_selection.chi2 
  - Wrapper -> individually add / delete features based on the precision -> sklearn.feature_selection.SelectFromModel
  - Embedded -> decision tree filter out feature importances -> clf.feature_importances_ + sklearn.feature_selection.SelectFromModel(clf, prefit=True)

**Feature Engineering**

- Stack train & test -> do feature transformation together
- Distribution of features -> box plot -> box-cox transformation.
- Missing data
  - Replace with mean / medium / common / regression / zero
  - Skip all rows with missing data / Skip features with many missing values
  - Anaylze -> missing data get added to one branch of split (ex: in credit feature, add unknown to poor)
  - Use classification error to determine where missing data go.
- Categorical variables -> one-hot encoded
- Noise -> less regularized, more iterations or depth of trees or deeper networks
- Linear Combination -> add, minus, multiply, divide by, ...etc
- Count attributes. Find those frequent and easily exploited ones.
- Unbalanced data:
  - Data augmentation (generate new data, such as rotation and shift in image data)
  - Give different weights to different classes.
  - Upsampling - increase the sampling rate
  - Downsampling - decrease the sampling rate
- Clustering -> k-means -> same cluster consider as same label
- Dimension Reduction -> PCA, Autoencoder
- Standardize data - mean = 0, std = 1
(Note: Tree-based models don't depend on standardization, but boosting, neural networks do care.)
- Split data into training and testing data. Shuffle the training data.

**Modeling**

Use machine learning as baseline to optimize deep learning.

Deep learning is not suitable for solving:
- Data is too small
- Data doesn't have local related features.

(1) Select Models

(a) Regression
- Ridge Regression: Linear Regression + L2 Regularization
- Lasso: Linear Regression + L1 Regularization
- Polynomial Regression
- Gradient Boost Machine (GBM)
- XGBoost
- Neural Networks
- Ensembles

(b) Classification
- Decision Tree
- Random Forest
- Logistic Regression
- Support Vector Machines (SVMs)
- k-nearest neighbors
- Naive Bayes
- Gradient Boost Machine (GBM)
- XGBoost
- Neural Networks
- Ensembles

(c) Clustering
- k-means
- Gaussian Mixture Model (GMM)
- Mean-shift
- DBSCAN
- Agglomerative Clustering

(d) Dimension Reduction
- Autoencoder
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Isometric Feature Mapping (Isomap)

**Tune Parameters**

- Search for papers to know the approximate values
- Grid Search
- Bayesian Optimization

(3) XGBoost Tunning Tips

(a) eta: Step size used in updating weights. Lower eta means slower training but better convergence.

(b) num_round: Total number of iterations.

(c) subsample: The ratio of training data used in each iteration. This is to combat overfitting.

(d) colsample_bytree: The ratio of features used in each iteration. This is like max_features in RandomForestClassifier.

(e) max_depth: The maximum depth of each tree. Unlike random forest, gradient boosting would eventually overfit if we do not limit its depth.

(f) early_stopping_rounds: If we don’t see an increase of validation score for a given number of iterations, the algorithm will stop early. This is to combat overfitting, too.

(h) Steps:

Step 1. Reserve a portion of training set as the validation set.

Step 2. Set eta to a relatively high value (e.g. 0.05 ~ 0.1), num_round to 300 ~ 500.

Step 3. Use grid search to find the best combination of other parameters.

Step 4. Gradually lower eta until we reach the optimum.

Step 5. Use the validation set as watch_list to re-train the model with the best parameters. Observe how score changes on validation set in each iteration. Find the optimal value for early_stopping_rounds.



(4) Deep Learning Tunning Tips
- Try mini-batch gradient descent.
- Try small learning rate at first.
- Try ReLU activation function and Adam optimizer at first.
(a) Underfitting (high bias):
- Deeper Neural Network (more neurons and more layers)
- Decrease L2 Regularization
- More Features

(b) Overfitting (high variance):
- Reason 1: model is too complex
- Reason 2: too many noises
- Reason 3: less training data

Try:
- L2 Regularization (restrict weigths)
- Dropout
- Batch Normalization
- Data Augmentation
- Pruning
- Gradient Clipping / Early Stopping
- Esemble Models
- Reduce Features
- More Data
- Check model's coefficient, overfitting often associated with large estimated coefficient.

(5) Cross Validation (CV)
- (Kaggle) Public leader board scores are not consistent with local CV scores due to noise or non ideal distribution. Local CV > public leader board.
- 5-fold CV is good enough.
- Implement stratified cross validation instead of basic cross validation on large number of classes or imbalance distribution for each classes.

(6) Evaluation Metric
Use the correct metric to evaluate the scores.

(7) Ensemble Models -- NO FREE LUNCH THEOREM

It reduces both bias and variance of the final model. Base models should be as unrelated as possibly. This is why we tend to include non-tree-based models in the ensemble even though they don’t perform as well. The math says that the greater the diversity, and less bias in the final ensemble. Also, performance of base models shouldn’t differ to much.

-  Bagging: 

Use different random subsets of training data to train each base model. Then all the base models vote to generate the final predictions. This is how random forest works.

-  Boosting: 

Train base models iteratively, modify the weights of training samples according to the last iteration. This is how gradient boosted trees work. (Actually it’s not the whole story. Apart from boosting, GBTs try to learn the residuals of earlier iterations.) It performs better than bagging but is more prone to overfitting.

-  Blending: 

Use non-overlapping data to train different base models and take a weighted average of them to obtain the final predictions. This is easy to implement but uses less data.

-  Stacking:

Take 5-fold stacking as an example. First we split the training data into 5 folds. Next we will do 5 iterations. In each iteration, train every base model on 4 folds and predict on the hold-out fold. You have to keep the predictions on the testing data as well. This way, in each iteration every base model will make predictions on 1 fold of the training data and all of the testing data. After 5 iterations we will obtain a matrix of shape #(samples in training data) X #(base models). This matrix is then fed to the stacker (it’s just another model) in the second level. After the stacker is fitted, use the predictions on testing data by base models (each base model is trained 5 times, therefore we have to take an average to obtain a matrix of the same shape) as the input for the stacker and obtain our final predictions.



**Pipline** -> create a highly automated pipeline
- Automated grid search / bayesian optimization
- Automated ensemble selection.


