# Pandas 資料分析和常用工具

<img src="figure/pandas.jpg" alt="pandas" width="1200">

## 引言
`pandas` 是一個強大的資料分析與處理工具，廣泛應用於數據科學、統計分析和機器學習等領域。本教學手冊旨在介紹 `pandas` 的基本用法，涵蓋資料讀取、清理、操作、分析和視覺化。
[pandas官網](https://pandas.pydata.org/pandas-docs/stable/index.html)

## 導入套件
```python
import pandas as pd

```

## 創建 DataFrame
```python
data = {
    'Name': ['John Doe', 'Anna Smith', 'Peter Johnson', 'Linda Lee', 'Sara Ku', 'David Huang', 'Laura Le', 'James J', 'Emily Wang', 'Donald Trump'],
    'Age': [28, 24, 35, 32, 40, 22, 29, 31, 27, 25],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Income': [50000, 60000, 55000, 65000, 70000, 48000, 52000, 61000, 59000, 53000],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married'],
    'Children': [0, 1, 2, 3, 1, 0, 2, 1, 0, 1],
    'Employment_Status': ['Employed', 'Employed', 'Unemployed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Self-employed', 'Employed', 'Employed'],
    'Education_Level': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'Years_of_Experience': [5, 3, 10, 8, 12, 2, 6, 9, 4, 3]
}

df = pd.DataFrame(data)

df.head()
```

## 資料讀取與輸出
### 輸出文件
```pytohn
# 將 DataFrame 輸出為 CSV 文件
df.to_csv('output.csv', index=False)

# 將 DataFrame 輸出為 Excel 文件
df.to_excel('output.xlsx', index=False)

# 將 DataFrame 輸出為 Pickle 文件
df.to_pickle('output.pickle')
```

### 讀取 CSV 文件
```python
df = pd.read_csv('data.csv')

# 顯示前五行
df.head()
```

### 讀取 Excel 文件
```pytohn
df = pd.read_excel('data.xlsx')

# 顯示前五行
df.head()
```

### 讀取 Pickle 文件
```pytohn
df = pd.read_pickle('data.pickle')

# 顯示前五行
df.head()
```

## 資料分析
### 描述統計
```python
# 計算基本統計量
df.describe()
```

### 計算相關係數矩陣
```python
correlation_matrix = df.corr()
print(correlation_matrix)
```

### 分組統計
- mean: is average of a data set is found by adding all numbers in the data set and then dividing by the number of values in the set.
- median is the middle value when a data set is ordered from least to greatest.
- mode is the number that occurs most often in a data set.
```python
# 依性別分組並計算平均年齡
grouped_df = df.groupby('Gender')['Age'].mean()
print(grouped_df)
```

### 數值計數
```python
# 計算每個性別的出現次數
gender_counts = df['Gender'].value_counts()
print(gender_counts)
```

### 資料篩選
```python
# 篩選出年齡大於30的資料
filtered_df = df[df['Age'] > 35]
filtered_df
```

## 資料清理
### 處理缺失值
```python
# 檢查缺失值
print(df.isnull().sum())
```

### 製作缺失值
```python
import numpy as np

# 設置隨機種子以便結果可重現
np.random.seed(0)

# 計算需要設置為缺失值的總數量
nan_fraction = 0.1  # 10% 的數據設置為 NaN
num_nans = int(np.floor(nan_fraction * df.size))

# 獲取 DataFrame 的所有欄位
columns_to_select = df.columns

# 隨機選擇行和列
for _ in range(num_nans):
    row = np.random.choice(df.index)
    col = np.random.choice(columns_to_select)
    df.at[row, col] = np.nan

# 檢查每個欄位的缺失值
missing_values = df.isnull().sum()
print(missing_values)
```

### 填補缺失值 `mean` & `median` & `mode`
```python
df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Age'].fillna(df['Age'].mode(), inplace=True)
```

### 刪除含有缺失值的行
```python
df.dropna(inplace=True)
```

### 改變資料類型 **Age** 欄位從`float`轉換為`int`
```python
df['Age'] = df['Age'].astype(int)
```

### 處理連續型數值
標準化公式

$$Z=\frac{X-\mu}{\sigma}$$

$$\begin{array}{l}\bullet\quad Z\text{是標準化後的數值}\\ \bullet\quad X\text{是原始數值}\\ \bullet\quad\mu\text{是數值的平均值}\\ \bullet\quad\sigma\text{是數值的標準差}\end{array}$$
```python
# 標準化連續型數值
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
```

### 處理類別數值
```python
unique_genders = df['Gender'].unique()
print(unique_genders)
```

### 使用 replace 方法轉換類別數值
```python
df['Gender1'] = df['Gender'].replace({'Male': 0, 'Female': 1})
```

### 使用 one-hot 編碼
```python
df = pd.get_dummies(df, columns=['Gender'], prefix='Gender', dtype=int)
```

### 刪除重複資料
```python
df.drop_duplicates(inplace=True)
```

### 重新命名欄位
```python
df.rename(columns={'Name': 'Full Name', 'Children': 'Children (count)'}, inplace=True)
```

### 刪除特定欄位
```python
df = df.drop(['Gender1'], axis=1)
```

### 拆分欄位
```python
# 使用 str.split 方法拆分 Full Name 列
name_split = df['Full Name'].str.split(' ', n=1, expand=True)
name_split.columns = ['First Name', 'Last Name']

name_split
```

### 拆分欄位 Methods 2
```python
df[['First Name', 'Last Name']] = df['Full Name'].str.split(' ', n=1, expand=True)
df.head()
```

### 合併欄位
```python
# 使用 concat 方法合併 DataFrame
df = pd.concat([df, name_split], axis=1)
df.head()
```
