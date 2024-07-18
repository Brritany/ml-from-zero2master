# Pandas 資料分析和處理教學手冊

## 引言
`pandas` 是一個強大的資料分析與處理工具，廣泛應用於數據科學、統計分析和機器學習等領域。本教學手冊旨在介紹 `pandas` 的基本用法，涵蓋資料讀取、清理、操作、分析和視覺化。
[pandas官網](https://pandas.pydata.org/pandas-docs/stable/index.html)

## 導入套件
```python
import pandas as pd

```

## 資料讀取與輸出
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

### 讀取 JSON 文件
```pytohn
df = pd.read_json('data.json')

# 顯示前五行
df.head()
```

### 創建 DataFrame
```python
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 24, 35, 32],
    'Gender': ['Male', 'Female', 'Male', 'Female']
}
df = pd.DataFrame(data)

# 顯示前五行
df.head()
```

### 輸出文件
```pytohn
# 將 DataFrame 輸出為 CSV 文件
df.to_csv('output.csv', index=False)

# 將 DataFrame 輸出為 Excel 文件
df.to_excel('output.xlsx', index=False)

# 將 DataFrame 輸出為 Pickle 文件
df.to_pickle('output.pickle')
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
```python
# 依性別分組並計算平均年齡
grouped_df = df.groupby('Gender')['Age'].mean()
print(grouped_df)
```





