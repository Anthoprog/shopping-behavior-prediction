import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from imblearn.over_sampling import SMOTE

# Test size for splitting data
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data and split into training and testing sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )

    # Apply SMOTE for balancing classes
    if len(set(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model and evaluate performance
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print evaluation results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate (Sensitivity): {100 * sensitivity:.2f}%")
    print(f"True Negative Rate (Specificity): {100 * specificity:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, predictions, sensitivity, specificity, "Confusion Matrix")

def load_data(filename):
    """
    Load shopping data from a CSV file and convert into evidence and labels.
    """
    evidence = []
    labels = []

    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                evidence.append([
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    month_to_index(row["Month"]),
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    1 if row["VisitorType"] == "Returning_Visitor" else 0,
                    1 if row["Weekend"] == "TRUE" else 0
                ])
                labels.append(1 if row["Revenue"] == "TRUE" else 0)
            except ValueError:
                warnings.warn("Skipping invalid data row.")
    return (evidence, labels)

def month_to_index(month):
    """
    Convert month name to an index (January = 0, December = 11).
    """
    months = {
        "January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
        "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
    }
    return months.get(month, -1)

def train_model(evidence, labels):
    """
    Train a k-nearest neighbor model with hyperparameter tuning.
    """
    param_grid = {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        scoring=make_scorer(roc_auc_score, needs_proba=True),
        cv=stratified_kfold
    )
    grid_search.fit(evidence, labels)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate(labels, predictions):
    """
    Calculate sensitivity and specificity.
    """
    true_positive = sum(1 for actual, pred in zip(labels, predictions) if actual == 1 and pred == 1)
    true_negative = sum(1 for actual, pred in zip(labels, predictions) if actual == 0 and pred == 0)
    false_positive = sum(1 for actual, pred in zip(labels, predictions) if actual == 0 and pred == 1)
    false_negative = sum(1 for actual, pred in zip(labels, predictions) if actual == 1 and pred == 0)

    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0

    return sensitivity, specificity

def plot_confusion_matrix(y_true, y_pred, sensitivity, specificity, title):
    """
    Plot a confusion matrix with sensitivity and specificity annotations.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
    plt.title(f"{title}\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()
