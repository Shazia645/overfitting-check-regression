# Overfitting Check: Linear Regression vs Decision Tree vs Random Forest

A small machine learning project that shows **what overfitting looks like** and **why cross-validation matters**, using the 50 Startups dataset.

I compare three regression models and check whether each one actually learns the pattern or just memorizes the training data.

## Project Goal

- Detect overfitting by comparing **Train R²** and **Test R²**
- See how `max_depth` affects a Decision Tree
- Learn why a tiny test set gives unreliable scores
- Use **K-Fold cross-validation** to compare models fairly

## Dataset

**50_Startups.csv**: 50 startups with these columns:

| Column | Description |
|---|---|
| R&D Spend | Money spent on research and development |
| Administration | Administration cost |
| Marketing Spend | Money spent on marketing |
| State | Location of the startup |
| Profit | **Target** variable to predict |

## Tools Used

- Python 3.11
- Pandas, NumPy
- scikit-learn
- Jupyter Notebook (VS Code)

## Step 1: Train/Test Split Results

| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| Linear Regression | 0.954 | 0.899 | ~0.05 |
| Decision Tree (no limit) | **1.000** | 0.836 | ~0.16 |

**Observation:** The Decision Tree scored a perfect 1.0 on training data but dropped on test data. This is the classic sign of **overfitting**: the tree memorized the training rows instead of learning the pattern.

## Step 2: Controlling Tree Depth

| max_depth | Train R² | Test R² |
|---|---|---|
| 2 | 0.893 | 0.723 |
| 3 | 0.969 | 0.800 |
| 4 | 0.988 | 0.715 |
| 5 | 0.999 | 0.801 |
| None | 1.000 | 0.836 |

**Observation:** The test score jumps up and down and does not improve steadily as depth is reduced. With only 50 rows, the test set has about 10 rows, so one or two predictions can change R² a lot. A single split is **not reliable** for choosing the best model.

## Step 3: Cross-Validation

### Mistake I ran into

My first cross-validation with `cv=5` gave very bad results (Linear Regression Mean R² of **-0.26**). The reason: the dataset is **sorted by Profit**, so without shuffling, each fold contained a different profit range. The model trained on one range and was tested on a completely different one.

### Fix

Use `KFold` with `shuffle=True`:

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, x, y, cv=kf, scoring="r2")
```

### Final Results (5-fold, shuffled)

| Model | Mean R² | Std |
|---|---|---|
| Linear Regression | **0.9279** | 0.0438 |
| Random Forest (100 trees) | 0.9277 | 0.0419 |
| Decision Tree (no limit) | 0.8719 | 0.0717 |
| Decision Tree (max_depth=3) | 0.8509 | 0.0617 |

## Key Learnings

1. **Train R² = 1.0 is a warning sign**, not a success. It usually means the model memorized the data.
2. **A small test set gives unstable scores.** Do not pick a model from one split of 10 rows.
3. **Cross-validation is more reliable** because every row is used for testing once.
4. **Always check if data is sorted.** If it is, use `shuffle=True` in `KFold`.
5. **Random Forest reduces overfitting** compared to a single tree (0.872 → 0.928) because it averages 100 trees.
6. **Simple models can win.** Linear Regression and Random Forest tied here, so the simpler and more explainable Linear Regression is the better choice for this data.

## Project Structure

```
overfitting-check/
├── overfitting_check.ipynb
├── 50_Startups.csv
└── README.md
```

## How to Run

```bash
git clone https://github.com/Shazia645/overfitting-check.git
cd overfitting-check
pip install pandas numpy scikit-learn jupyter
jupyter notebook overfitting_check.ipynb
```

## Author

**Shazia Batool**
AI student at COMSATS University
GitHub: [Shazia645](https://github.com/Shazia645)