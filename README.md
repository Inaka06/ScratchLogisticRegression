# Heart Disease Prediction — Logistic Regression From Scratch

## Disclaimer

This project was originally developed some time ago and has only recently been uploaded for documentation and portfolio purposes. The implementation reflects my understanding and coding practices at the time it was created.

---

## Overview

This project implements Logistic Regression from scratch using NumPy to predict 10-Year Coronary Heart Disease (CHD) risk from the `heart.csv` dataset.

The objective of this project was to understand the internal mechanics of logistic regression rather than relying on pre-built machine learning libraries. All core components of the model — including gradient descent optimization and loss computation — were implemented manually.

---

## Objectives

* Implement binary classification without using `sklearn`’s LogisticRegression model
* Understand gradient descent optimization
* Implement numerically stable sigmoid activation
* Apply weighted loss to address class imbalance
* Build a complete evaluation pipeline

---

## Model Implementation

### Logistic Regression

The model includes:

* Custom sigmoid activation with numerical stability
* Gradient descent optimization
* Weighted binary cross-entropy loss
* Manual train/test split
* Proper feature standardization (training data statistics only)

### Numerical Stability

To prevent overflow in exponential calculations:

```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
```

### Class Weighting

To reduce the effect of class imbalance, the minority class is weighted during training:

```python
weights_array = np.where(y == 1, 2.0, 1.0)
```

### Loss Function

Weighted Binary Cross-Entropy:

$$
L = -\frac{1}{m} \sum w_i \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]
$$

---

## Data Preparation

* Missing numeric values are filled with the median of each column.
* The dataset is split into training (80%) and testing (20%) sets.
* Feature standardization is performed using the training set mean and standard deviation to prevent data leakage.

---

## Evaluation

The model outputs:

* Accuracy
* Precision, Recall, and F1-score
* Confusion Matrix visualization

Example output:

```
Accuracy: 0.84
```

(Results may vary depending on random train/test split.)

---

## Project Structure

```
.
├── heart.csv
├── logisticRegression.py
└── README.md
```

---

## Potential Improvements

If further developed, the following enhancements could be added:

* L2 regularization
* Early stopping
* Loss curve visualization
* ROC-AUC evaluation
* Comparison with sklearn’s LogisticRegression implementation

---

## Requirements

* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
