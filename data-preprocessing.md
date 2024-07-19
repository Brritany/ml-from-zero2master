# 資料前處理(Data Preprocessing)
- 沒有優質的資料，就沒有優質的成果！
  - 決策必須基於高品質資料
    - 例如，重複或遺失的資料可能會導致不正確甚至誤導性的統計資料。
- 資料清理往往十分耗時

# 加載數據集
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 加載數據集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head(5)
```

## 檢查每個欄位是否存在缺失值
```python
missing_values = df.isnull().sum()
print(missing_values)
```

## 手動建立缺失值，用於後續教學
```python
import numpy as np
# 設置隨機種子以便結果可重現
np.random.seed(0)

# 計算需要設置為缺失值的總數量
nan_fraction = 0.1  # 10% 的數據設置為 NaN
num_nans = int(np.floor(nan_fraction * df.size))

# 確保 target 欄位不會被選中
columns_to_select = df.columns[:-1]  # 排除 'target' 欄位

# 隨機選擇行和列
for _ in range(num_nans):
    row = np.random.choice(df.index)
    col = np.random.choice(columns_to_select)
    df.at[row, col] = np.nan

# 再次檢查每個欄位的缺失值
missing_values = df.isnull().sum()
print(missing_values)
```

# 處理缺失值
## 導入工具包
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
```

## 簡單插補法 `SimpleImputer`
```python
# 1. 用均值填充部分欄位
mean_fill_columns = ['mean radius', 'mean texture', 'mean perimeter']
imputer_mean = SimpleImputer(strategy='mean')
df[mean_fill_columns] = imputer_mean.fit_transform(df[mean_fill_columns])
```

```python
# 2. 用中位數填充部分欄位
median_fill_columns = ['mean area', 'mean smoothness']
imputer_median = SimpleImputer(strategy='median')
df[median_fill_columns] = imputer_median.fit_transform(df[median_fill_columns])
```

```python
# 3. 用眾數填充部分欄位
mode_fill_columns = ['mean compactness', 'mean concavity']
imputer_mode = SimpleImputer(strategy='most_frequent')
df[mode_fill_columns] = imputer_mode.fit_transform(df[mode_fill_columns])
```

## 線性插補法 `interpolate`
- 一種用於填充缺失值的方法
- 使用已知資料點之間的直線來估計缺失值
- 適用於時間序列數據或具有連續性特徵的數據

``` python
Before
   mean concave points  mean symmetry
0                  0.1            0.2
1                  NaN            0.3
2                  0.2            NaN
3                  NaN            0.5
4                  0.4            0.6

After
   mean concave points  mean symmetry
0                  0.1            0.2
1                  0.15           0.3
2                  0.2            0.4
3                  0.3            0.5
4                  0.4            0.6
```

線性插補會在Index 0 和Index 2 的已知資料之間畫一條直線。這條直線的公式是 y = mx + b，其中 m 是斜率，b 是截距。斜率 m 是 (0.2 - 0.1) / (2 - 0) = 0.1 / 2 = 0.05
```python
# 4. 用線性插值填充部分欄位
interpolate_fill_columns = ['mean concave points', 'mean symmetry']
df[interpolate_fill_columns] = df[interpolate_fill_columns].interpolate()
```

### KNN插補法 `KNNImputer`
- 使用最近鄰居算法（K-Nearest Neighbors, KNN）來填補缺失值的一種方法
- 基本思想是用資料集中與缺失值最近的K個數據點的值來估計該缺失值
- 適用於資料間具有局部相似性或聚類特性的情況
- `KNNImputer` 中的 `n_neighbors` 表示指定用於填補缺失值的最近鄰居數量，不包含本身的缺失值
- 在填補缺失值中，`KNNImputer` 會找到資料集中與缺失值最近的 `n_neighbors` 個有效資料點，然後使用這些鄰居的值（通常是均值）來填補缺失值


``` python
Before
   mean fractal dimension  radius error
0                     0.10           1.1
1                     0.15           NaN
2                      NaN           1.3
3                     0.20           1.4
4                     0.25           NaN

After
   mean fractal dimension  radius error
0                    0.10      1.100000
1                    0.15      1.266667
2                    0.175     1.300000
3                    0.20      1.400000
4                    0.25      1.266667
```
使用 KNN 算法計算出的鄰居值如下：

對於 mean fractal dimension 欄位的索引 2 的缺失值：

最近的鄰居是索引 0, 1, 3, 4，平均值是 (0.1 + 0.15 + 0.2 + 0.25) / 4 = 0.175

對於 radius error 欄位的索引 1 的缺失值：

最近的鄰居是索引 0, 2, 3，平均值是 (1.1 + 1.3 + 1.4) / 3 = 1.2667

對於 radius error 欄位的索引 4 的缺失值：

最近的鄰居是索引 0, 1, 2, 3，平均值是 (1.1 + 1.2667 + 1.3 + 1.4) / 4 = 1.2667
```python
# 5. 用KNN填充部分欄位
knn_fill_columns = ['mean fractal dimension', 'radius error']
knn_imputer = KNNImputer(n_neighbors=5)
df[knn_fill_columns] = knn_imputer.fit_transform(df[knn_fill_columns])
```

## 插補特定數值
```python
# 6. 用特定值填充部分欄位
specific_value_fill_columns = ['texture error', 'perimeter error']
df[specific_value_fill_columns] = df[specific_value_fill_columns].fillna(0)
```

## 再次確認每個欄位的缺失值
```python
# 檢查處理後每個欄位的缺失值
missing_values_after = df.isnull().sum()
print(missing_values_after)
```

## 多重插補法 `IterativeImputer`
- 用於填補資料集中的缺失值
- 迭代地使用其他特徵來估計缺失值
- 優點在於可充分利用數據集中所有可用的信息來進行插補，通常比簡單的插補方法（如均值插補或中位數插補）更加精確

常用參數

`max_iter`默認值：10

說明：最大迭代次數。插補過程將重複這麼多次，直到結果收斂或達到最大迭代次數。

`n_nearest_features`默認值：None

說明：用於插補的最近鄰居特徵的數量。如果設置為 None，則使用所有特徵。

`initial_strategy`默認值：'mean'

說明：初始插補策略，用於第一次填補缺失值。選項有 'mean'、'median'、'most_frequent'。

`imputation_order`默認值：'ascending'

說明：插補順序。選項有 'ascending'（從缺失值最少的特徵開始）、'descending'（從缺失值最多的特徵開始）、'roman'（按列順序）、'arabic'（按列逆序）和 'random'（隨機順序）。

`random_state`默認值：None

說明：控制隨機數生成器的種子，以保證結果的可重現性。
```python
# 剩餘欄位的缺失值處理
remaining_columns = ['area error', 'smoothness error', 'compactness error',
                     'concavity error', 'concave points error', 'symmetry error',
                     'fractal dimension error', 'worst radius', 'worst texture',
                     'worst perimeter', 'worst area', 'worst smoothness',
                     'worst compactness', 'worst concavity', 'worst concave points',
                     'worst symmetry', 'worst fractal dimension']

# 7. 使用IterativeImputer填充
iterative_imputer = IterativeImputer(random_state=0,
                                     max_iter=50,
                                     n_nearest_features=None,
                                     imputation_order='ascending',
                                     initial_strategy='median',
)

df[remaining_columns] = iterative_imputer.fit_transform(df[remaining_columns])
```

## 最後應該會的到一份沒有缺失值的資料集
```python
# 檢查處理後每個欄位的缺失值
missing_values_final = df.isnull().sum()
print(missing_values_final)
```
