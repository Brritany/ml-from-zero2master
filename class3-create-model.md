# 模型建構

在本節中，我將介紹如何從頭開始構建機器學習模型，涵蓋監督學習、非監督學習和集成學習等不同類型的模型。每個步驟都將提供詳細的說明和實際代碼示例，幫助您在實際應用中實現這些模型。我們將使用**Scikit-learn**的乳腺癌數據集進行二元分類任務。

## 數據集介紹

我們使用的數據集是**Scikit-learn**提供的乳腺癌數據集（Breast Cancer dataset）。這個數據集包含了569個樣本，每個樣本有30個特徵，用於預測腫瘤是良性還是惡性。

### 數據集結構

- **樣本數量**：569
- **特徵數量**：30
- **類別**：二元分類（良性或惡性）
- **特徵名稱**：
  - mean radius
  - mean texture
  - mean perimeter
  - mean area
  - mean smoothness
  - mean compactness
  - mean concavity
  - mean concave points
  - mean symmetry
  - mean fractal dimension
  - radius error
  - texture error
  - perimeter error
  - area error
  - smoothness error
  - compactness error
  - concavity error
  - concave points error
  - symmetry error
  - fractal dimension error
  - worst radius
  - worst texture
  - worst perimeter
  - worst area
  - worst smoothness
  - worst compactness
  - worst concavity
  - worst concave points
  - worst symmetry
  - worst fractal dimension

### 加載數據集

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 加載數據集
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head(5)
```

## 1. 監督學習模型

### 1.1 朴素貝葉斯(Naive Bayes, NB)

**Naive Bayes** 是一種基於貝葉斯定理的分類算法，假設特徵之間是條件獨立的。它通常用於文本分類和垃圾郵件檢測等問題。

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 加載數據集
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = GaussianNB()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.2 線性回歸（Linear Regression, LR）

**線性回歸**是一種用於預測連續目標變量的基本模型。這裡我們將使用邏輯回歸來進行二元分類。

```python
from sklearn.linear_model import LogisticRegression

# 建立模型
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.3 決策樹（Decision Tree, DT）

**決策樹**是一種基於樹狀結構進行分類和回歸的模型。每個節點表示一個特徵，分支表示特徵的值，葉子節點表示預測結果。

```python
from sklearn.tree import DecisionTreeClassifier

# 建立模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.4 隨機森林（Random Forest, RF）

**隨機森林**是由多棵決策樹組成的集成學習模型，通過對多個決策樹的預測結果進行平均或投票來提高預測準確性和穩定性。

```python
from sklearn.ensemble import RandomForestClassifier

# 建立模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.5 支持向量機（Support Vector Machine, SVM）

**支持向量機**是一種監督學習算法，通過尋找最佳分隔超平面來進行分類或回歸。SVM 旨在最大化分類邊界，從而提高模型的泛化能力。

```python
from sklearn import svm

# 建立模型
model = svm.SVC()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.6 多層感知機（Multilayer Perceptron, MLP）

**多層感知機**是一種前饋神經網絡，包含一個或多個隱藏層，適用於處理非線性問題和複雜的模式識別任務。

```python
from sklearn.neural_network import MLPClassifier

# 建立模型
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000)
mlp.fit(X_train, y_train)

# 預測
predictions = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.7 XGBoost

**XGBoost**是一種基於梯度提升框架的集成學習算法，通過集成多個弱學習器來提高預測性能，並具有高效的運算和優秀的預測能力。

```python
import xgboost as xgb

# 建立模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 1.8 LightGBM

**LightGBM**是一種基於梯度提升的算法，旨在提高訓練速度和降低內存使用。它能夠處理大規模數據和高維特徵。

```python
import lightgbm as lgb

# 建立模型
train_data = lgb.Dataset(X_train, label=y_train)
param = {'num_leaves': 31, 'objective': 'binary', 'metric': 'binary_logloss'}
model = lgb.train(param, train_data, 100)

# 預測
predictions = model.predict(X_test)
# LightGBM的預測結果是概率，需要轉為二元分類結果
predictions = [1 if pred > 0.5 else 0 for pred in predictions]
print("Accuracy:", accuracy_score(y_test, predictions))
```

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
