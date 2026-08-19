Logistic Regression (Cell 564d919f)
What it is: Despite its name, Logistic Regression is a fundamental algorithm for classification tasks, not regression. It models the probability of a binary outcome (0 or 1), but can be extended to handle multi-class classification problems, as seen here. It does this by fitting a logistic function (sigmoid) to the data, which squashes any real-valued number into a probability between 0 and 1.
How it works (Multi-class): For multi-class problems like predicting diseases, Logistic Regression typically uses one of two strategies:
One-vs-Rest (OvR) or One-vs-All (OvA): This trains a separate binary classifier for each class, where each classifier distinguishes one class from all other classes. The class with the highest predicted probability from these classifiers is chosen as the final prediction.
Multinomial Logistic Regression: This is a direct extension that models the probability distribution over all classes simultaneously using a softmax function.
In this code: The LogisticRegression from sklearn.linear_model automatically handles the multi-class nature using one of these strategies (by default, it's 'ovr' for binary problems, and 'auto' for multi-class which often defaults to 'lbfgs' solver for multinomial logistic regression, or 'ovr' for other solvers if specified). The parameters used are:
solver='saga': A good choice for large datasets and can handle both L1 and L2 regularization. It's often preferred for its speed and ability to handle various types of optimization problems.
max_iter=1000: The maximum number of iterations for the solver to converge. This was increased to ensure the model has enough steps to find an optimal solution.
random_state=42: Ensures reproducibility of the results.
penalty='l2': This is a regularization term that helps prevent overfitting by penalizing large coefficients. L2 regularization (Ridge) adds the sum of the squared magnitudes of coefficients to the loss function.
Evaluation: The model's performance is measured using accuracy_score, precision, recall, and f1_score. A confusion_matrix is also generated, though due to the large number of classes (754), it's visually dense and primarily intended for MLflow tracking rather than direct interpretation in the notebook.
MLflow Integration: The entire process (parameters, metrics, and the trained model itself) is logged using MLflow, enabling easy tracking and comparison of different runs.
Random Forest Classifier (Cell 003a0932)
What it is: Random Forest is an ensemble learning method for classification and regression. It operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
How it works: It introduces randomness in two main ways:
Bootstrap Aggregating (Bagging): Each tree is trained on a different random subset of the training data (with replacement).
Feature Randomness: When splitting a node in a decision tree, only a random subset of features is considered. This decorrelates the trees, making the ensemble's prediction more robust and less prone to overfitting.
Advantages: Random Forests are highly robust, can handle a large number of features, are less susceptible to overfitting than individual decision trees, and often provide high accuracy.
In this code: The RandomForestClassifier from sklearn.ensemble is used with the following parameters:
n_estimators=100: This specifies the number of decision trees in the forest. More trees generally lead to better performance but increase computational cost.
max_depth=None: Allows nodes to expand until all leaves are pure or until all leaves contain less than min_samples_split samples. This means the trees can grow quite deep.
min_samples_split=2: The minimum number of samples required to split an internal node.
min_samples_leaf=1: The minimum number of samples required to be at a leaf node.
random_state=42: For reproducibility.
n_jobs=-1: Utilizes all available CPU cores for parallel processing, speeding up training.
Evaluation: Similar to Logistic Regression, the Random Forest model is evaluated using accuracy_score, precision, recall, and f1_score, and its confusion_matrix is generated.
MLflow Integration: Parameters, metrics, and the trained Random Forest model are also logged with MLflow for comparison.
Both models are fundamental to machine learning and were chosen to provide a baseline comparison for this multi-class disease classification task. The MLflow integration allows for easy tracking of their performance and artifacts.
