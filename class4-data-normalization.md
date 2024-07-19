# 數據標準化(Data Normalization)

**數據標準化**將數據轉換到相同的尺度，以提高模型的穩定性和收斂速度。

# 導入工具包
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
```

## Min-Max Scaling（最小-最大縮放）
- 將資料縮放到 [0, 1] 的範圍內。
```python
# 假設 'target' 是目標欄位的名稱
target_column = 'target'

# 分離特徵和目標欄位
features = df.drop(columns=[target_column])
target = df[target_column]
```

```python
# 進行 Min-Max Scaling 只對特徵欄位
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# 將目標欄位添加回去
normalized_df_min_max = pd.concat([normalized_features, target.reset_index(drop=True)], axis=1)
normalized_df_min_max
```

## Z-Score Standardization（標準化）
- 將資料縮放為均值為0，標準差為1。
```python
normalized_z_score_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# 將目標欄位添加回去
normalized_df_z_score = pd.concat([normalized_z_score_features, target.reset_index(drop=True)], axis=1)
normalized_df_z_score
```

## Robust Scaler（穩健縮放器）
- 基於中位數和四分位數範圍進行縮放，對於含有離群值的資料集更為穩健。
```python
# 3. Robust Scaler（穩健縮放器）
scaler = RobustScaler()
normalized_robust_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# 將目標欄位添加回去
normalized_df_robust = pd.concat([normalized_robust_features, target.reset_index(drop=True)], axis=1)
normalized_df_robust
```

## Max Abs Scaler（最大絕對值縮放器）
- 將資料縮放到 [-1, 1] 的範圍內，適合稀疏資料集。
```python
# 4. Max Abs Scaler（最大絕對值縮放器）
scaler = MaxAbsScaler()
normalized_max_abs_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# 將目標欄位添加回去
normalized_df_max_abs = pd.concat([normalized_max_abs_features, target.reset_index(drop=True)], axis=1)
normalized_df_max_abs
```
