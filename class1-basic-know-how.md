# 機器學習(Machine Learning)

<img src="figure/ML.png" alt="ML" width="1200">

- 人工智慧（Artificial Intelligence, AI）的一個重要分支。
- 它使計算機能夠通過數據進行學習並改進性能，而不需要明確的編程。

# But In Fact

<img src="figure/ML_joke.jpg" alt="ML_joke" width="900" height="600">

## 1. 機器學習概念

### 1.1 機器學習定義

- 一種使計算機系統能夠通過數據來學習和做出預測或決策的技術。
- 通過數據驅動模型來解決複雜問題。
- 目標是讓系統能夠自動從經驗中學習，並在處理新數據時做出準確的預測或判斷。

### 1.2 監督學習與非監督學習

- **監督學習**（Supervised Learning）：在這種學習模式下，模型在訓練過程中使用帶有標籤的數據。每個訓練樣本都有一個對應的正確答案，模型的目標是學習從輸入到輸出的映射。

<img src="figure/Supervised Learning.png" alt="Supervised Learning" width="1200">

  **例子**：假設您要預測房屋價格。您可以使用包含房屋特徵（如面積、位置、房間數等）和實際價格的數據集進行訓練。模型學會將房屋特徵與價格對應起來，從而對新房屋進行價格預測。

- **非監督學習**（Unsupervised Learning）：與監督學習不同，非監督學習使用的數據沒有標籤。模型的目標是從數據中找到隱藏的模式或結構。

<img src="figure/Unsupervised Learning.png" alt="Unsupervised Learning" width="1200">

  **例子**：假設您有一組顧客數據，但不知道這些顧客可以被劃分成多少類。您可以使用聚類算法（如 K-means）來自動將顧客分成不同的群體，從而發現不同的顧客類型。

### 1.3 強化學習

**強化學習**（Reinforcement Learning）是一種通過與環境交互來學習的機器學習方法。它基於獎勵機制，模型或智能體(Agent)在每一步行動後獲得回報，通過試錯法來最大化累積回報。

<img src="figure/Reinforcement Learning.webp" alt="Reinforcement Learning" width="1200">

**例子**：在遊戲中，強化學習可以用來訓練智能體自動玩遊戲。智能體在每一步行動後會根據得分（獎勵）來調整策略，最終學會如何玩得更好以獲得更高的得分。

## 2. 數據處理

### 2.1 數據清理(Data Cleaning & Data Preprocessing)

數據清理是數據預處理的第一步，包括處理缺失值、去除異常值和修正數據錯誤。

<img src="figure/Data Cleaning.png" alt="Data Cleaning" width="1200">

**例子**：如果您的數據集中有某些樣本的房屋價格缺失，您可能需要選擇填補缺失值（如使用平均值）或刪除這些樣本，以確保數據的完整性和準確性。

### 2.2 特徵工程

**特徵工程**（Feature Engineering）是從原始數據中提取和構建有用特徵的過程。好的特徵能夠顯著提高模型的性能。

**例子**：在預測房價的問題中，除了使用原始的面積和位置等特徵，您還可以創造新的特徵，如“每平方米價格”或“距離市中心的距離”，這些新特徵可能對預測結果有更好的影響。

### 2.3 數據標準化與正則化

- **數據標準化**（Data Normalization）：將數據轉換到相同的尺度，以提高模型的穩定性和收斂速度。

  **例子**：如果房屋的面積範圍是 50 到 500 平方米，而價格範圍是 50,000 到 500,000 美元，對這些特徵進行標準化（如 Z-score 標準化）可以讓它們在相同的尺度下進行比較和學習。

- **正則化**（Regularization）：防止模型過擬合的一種技術。通過在損失函數中加入懲罰項，限制模型的複雜度。

  **例子**：在回歸模型中使用 L2 正則化（Ridge Regression）可以避免模型過於複雜，從而提高對新數據的泛化能力。這樣可以防止模型僅僅記住訓練數據，而對新的數據表現不佳。

## 3. 模型評估

### 3.1 評估指標

模型的性能通常通過各種指標來評估。
- 分類問題中，常用的指標有準確率、精確率、召回率和 F1 分數。
- 回歸問題中，常用的指標有均方誤差（MSE）和決定係數（R²）。

**例子**：在疾病預測模型中，您可能會關心模型的召回率，即模型能夠正確識別出多少實際患病的患者。高召回率表示模型能夠捕捉到大部分真正的陽性案例。

### 3.2 交叉驗證

**交叉驗證**（Cross-Validation）是一種評估模型性能的方法。通過將數據集分為多個子集，進行多次訓練和驗證，以獲得更可靠的性能估計。

<img src="figure/Cross-Validation.png" alt="Cross-Validation" width="1200">

**例子**：使用 10 折交叉驗證時，將數據集分為 10 個子集，每次選擇其中 9 個子集進行訓練，剩餘的 1 個子集用於驗證，這樣的過程重複 10 次，每個子集都被用作驗證集一次。最終的性能指標是這 10 次驗證結果的平均值。

## 4. 常見機器學習算法

### 4.1 Naive Bayes（朴素貝葉斯）

**Naive Bayes** 是一種基於貝葉斯定理的分類算法，假設特徵之間是條件獨立的。它通常用於文本分類和垃圾郵件檢測等問題。

<img src="figure/Naive Bayes.jpg" alt="Naive Bayes" width="1200">

**例子**：在垃圾郵件過濾系統中，Naive Bayes 可以根據郵件中的單詞頻率來預測郵件是否是垃圾郵件。

### 4.2 線性回歸（Linear Regression, LR）

**線性回歸**（Linear Regression）是一種基本的回歸算法，用於預測連續值。它通過擬合一條最佳的直線來最小化預測值和實際值之間的誤差。

<img src="figure/Linear Regression.png" alt="Linear Regression" width="1200">

**例子**：預測一個學生的考試分數，可以使用線性回歸模型，通過學習學生的學習時間與實際考試成績之間的關係來進行預測。

### 4.3 決策樹（Decision Tree, DT）

**決策樹**（Decision Tree）是一種以樹狀結構進行分類和回歸的算法。每個節點表示一個特徵，分支表示特徵的值，葉子節點表示預測結果。

<img src="figure/Decision Tree.png" alt="Decision Tree" width="1200">

**例子**：在判斷是否批准貸款時，決策樹可以根據申請人的信用分數、收入和負債等特徵進行分類，以決定是否批准貸款。

### 4.4 隨機森林（Random Forest, RF）

**隨機森林**（Random Forest）是由多棵決策樹組成的集成學習算法。它通過對多個決策樹的預測結果進行平均或投票來提高預測準確性和穩定性。

<img src="figure/Random Forest.jpg" alt="Random Forest" width="1200">

**例子**：在預測顧客流失的問題中，隨機森林可以結合多棵決策樹的預測結果來提供更準確的預測，從而幫助企業制定保留顧客的策略。

### 4.5 支持向量機（Support Vector Machine, SVM）

**支持向量機**（Support Vector Machine, SVM）是一種監督學習算法，通過尋找最佳分隔超平面來進行分類或回歸。SVM 旨在最大化分類邊界，從而提高模型的泛化能力。

<img src="figure/Support Vector Machine.png" alt="Support Vector Machine" width="1200">

**例子**：在手寫數字識別問題中，SVM 可以用來區分不同數字的圖像，通過尋找最佳邊界來分隔不同的數字類別。

### 4.6 多層感知機（Multilayer Perceptron, MLP）

**多層感知機**（Multilayer Perceptron, MLP）是一種前饋神經網絡，包含一個或多個隱藏層。它適用於處理非線性問題和複雜的模式識別任務。

<img src="figure/Multilayer Perceptron.png" alt="Multilayer Perceptron" width="1200">

**例子**：在圖像識別中，MLP 可以用來識別圖像中的物體或特徵。通過多層隱藏層的學習，MLP 可以捕捉到圖像的複雜模式。

### 4.7 XGBoost

**XGBoost**（Extreme Gradient Boosting）是一種集成學習算法，基於梯度提升（Gradient Boosting）框架。它通過集成多個弱學習器來提高預測性能，並具有高效的運算和優秀的預測能力。

<img src="figure/XGBoost.png" alt="XGBoost" width="1200">

**例子**：在房價預測中，XGBoost 可以利用訓練數據中多個特徵的交互作用，從而提供更準確的預測結果。

### 4.8 LightGBM

**LightGBM**（Light Gradient Boosting Machine）是一種基於梯度提升的算法，旨在提高訓練速度和降低內存使用。它能夠處理大規模數據和高維特徵。

<img src="figure/LightGBM.png" alt="LightGBM" width="1200">

**例子**：在大型電商平台的推薦系統中，LightGBM 可以用來處理大量用戶行為數據，提供即時的推薦結果。

---

希望這份基礎知識文檔及其例子能夠幫助您進一步了解機器學習的基本概念。如果有任何需要修改或擴充的部分，請隨時告訴我！
