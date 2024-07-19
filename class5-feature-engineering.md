# 特徵工程（Feature Engineering）
特徵工程是構建更好模型的關鍵部分。以下是使用 **scikit-learn** 進行特徵工程的一些常用技術，例如主成分分析（PCA）、特徵選擇和抽樣等。

# 導入工具包
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.utils import resample
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
```

# 資料集
使用 load_breast_cancer 資料集，這是一個常用的二元分類問題的資料集。
```python
# 加載乳腺癌資料集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 分離特徵和目標欄位
target_column = 'target'
features = df.drop(columns=[target_column])
target = df[target_column]
```

## 主成分分析（PCA）
PCA 是一種降維技術，通過線性變換將數據轉換到一個新的坐標系中，其中新的坐標系中的軸（主成分）是根據數據的變異性排序的。
```python
# 進行 PCA
pca = PCA(n_components=2)  # 將維度降到2維
principal_components = pca.fit_transform(features)
pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])

# 將目標欄位添加回去
pca_df = pd.concat([pca_df, target.reset_index(drop=True)], axis=1)
pca_df.head()

```

## 特徵選擇（Feature Selection）
特徵選擇是從數據中選擇最重要的特徵，以提高模型的性能。

### 使用卡方檢驗（Chi-Square Test）
卡方檢驗用於測試兩個分類變量之間的獨立性。這裡我們用它來選擇與目標變量最相關的特徵。
```python
# 使用卡方檢驗進行特徵選擇
select_k_best = SelectKBest(chi2, k=2)
selected_features = select_k_best.fit_transform(features, target)
selected_df = pd.DataFrame(data=selected_features, columns=['selected_feature_1', 'selected_feature_2'])

# 將目標欄位添加回去
selected_df = pd.concat([selected_df, target.reset_index(drop=True)], axis=1)
selected_df.head()
```

## 使用遞歸特徵消除（RFE）
遞歸特徵消除（RFE）是通過遞歸地訓練模型，並每次消除表現最差的特徵來選擇特徵。
```python
from sklearn.linear_model import LogisticRegression

# 使用遞歸特徵消除進行特徵選擇
model = LogisticRegression(max_iter=10000)
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(features, target)

selected_features_rfe = features.loc[:, fit.support_]
selected_df_rfe = pd.concat([selected_features_rfe, target.reset_index(drop=True)], axis=1)
selected_df_rfe.head()
```

## 數據抽樣（Sampling）
處理不平衡數據集的一種方法是通過過採樣或下採樣來平衡類別分佈。

### 上採樣（Over-sampling）
上採樣是通過增加少數類別的樣本數量來平衡數據集。
```python
# 將數據集分為多數類別和少數類別
majority = df[df[target_column] == df[target_column].value_counts().idxmax()]
minority = df[df[target_column] == df[target_column].value_counts().idxmin()]

# 上採樣少數類別
minority_upsampled = resample(minority, 
                              replace=True,     # 允許放回抽樣
                              n_samples=len(majority),    # 使兩個類別的數量相同
                              random_state=42)  # 設定隨機種子

# 合併數據集
upsampled_df = pd.concat([majority, minority_upsampled])
upsampled_df['target'].value_counts()
```

### 下採樣（Under-sampling）
下採樣是通過減少多數類別的樣本數量來平衡數據集。
```python
# 將數據集分為多數類別和少數類別
majority = df[df[target_column] == df[target_column].value_counts().idxmax()]
minority = df[df[target_column] == df[target_column].value_counts().idxmin()]

# 下採樣多數類別
majority_downsampled = resample(majority, 
                                replace=False,    # 不允許放回抽樣
                                n_samples=len(minority),  # 使兩個類別的數量相同
                                random_state=42)  # 設定隨機種子

# 合併數據集
downsampled_df = pd.concat([majority_downsampled, minority])
downsampled_df['target'].value_counts()
```
