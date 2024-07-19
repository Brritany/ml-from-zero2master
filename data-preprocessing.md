## 2. 特徵工程

### 2.1 特徵提取

**特徵提取**是從原始數據中提取有用特徵的過程。好的特徵能夠顯著提高模型的性能。

```python
import pandas as pd

# 數據集
df = pd.DataFrame(data.data, columns=data.feature_names)

# 特徵提取 - 這裡我們假設創建一個新的特徵
df['mean area per mean smoothness'] = df['mean area'] / df['mean smoothness']
print(df.head())
```

### 2.2 數據標準化

**數據標準化**將數據轉換到相同的尺度，以提高模型的穩定性和收斂速度。

```python
from sklearn.preprocessing import StandardScaler

# 數據標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2.3 特徵選擇

**特徵選擇**是選擇對模型性能有最大影響的特徵，從而提高模型的準確性和效率。

```python
from sklearn.feature_selection import SelectFromModel

# 建立隨機森林模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 特徵選擇
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

這些範例和步驟說明應該能夠幫助您開始構建和應用各種類型的機器學習模型。如果有任何需要修改或擴充的部分，請隨時告訴我！
