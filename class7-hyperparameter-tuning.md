# 超參數調整(Hyperparameter Tuning)

在本節中，將介紹如何對機器學習模型進行超參數調整，以提升模型性能。

- 超參數調整是指在訓練模型之前設定的參數，這些參數不能從數據中學習到。
- 使用 **Scikit-learn** 的乳腺癌數據集示範超參數調整。

## 導入工具包
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

## 載入資料集
```python
# 載入乳腺癌數據集
data = load_breast_cancer()
X = data.data
y = data.target

# 拆分數據為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 初始化模型
- 使用隨機森林分類器作為範例

```python
# 初始化隨機森林分類器
baseline_model = RandomForestClassifier(random_state=42)

# 訓練基線模型
baseline_model.fit(X_train, y_train)

# 預測測試集
y_pred_baseline = baseline_model.predict(X_test)

# 計算基線模型分數
baseline_f1 = f1_score(y_test, y_pred_baseline)
print(f"Baseline Macro F1: {baseline_f1:.5f}")

# 輸出模型的參數值
print("Baseline Model Parameters:")
print(baseline_model.get_params())
```

## 定義超參數範圍
```python
# 定義超參數範圍
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## 網格搜索 (Grid Search)
```python
# 初始化網格搜索
grid_search = GridSearchCV(estimator=baseline_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# 進行網格搜索
grid_search.fit(X_train, y_train)

# 輸出最佳參數和最佳分數
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
```

## 隨機搜索 (Random Search)
```python
# 初始化隨機搜索
random_search = RandomizedSearchCV(estimator=baseline_model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)

# 進行隨機搜索
random_search.fit(X_train, y_train)

# 輸出最佳參數和最佳分數
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_}")
```

## 評估最佳模型
```python
# 使用最佳模型進行預測
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 計算分數
best_f1 = f1_score(y_test, y_pred)

print(f"Baseline Macro F1: {baseline_f1:.5f}")
print(f"Best Macro F1: {best_f1:.5f}")

```
