# 🌸 Iris Classification — Random Forest Baseline  
# 鸢尾花分类 — 随机森林与特征审计

## 📘 Overview | 项目简介
This project builds a **Random Forest classifier** to identify iris flower species based on four morphological features.  
It demonstrates a complete **supervised learning workflow** — from data preprocessing, encoding, and feature scaling, to model training, evaluation, and feature importance auditing.

本项目使用 **随机森林分类器** 对鸢尾花（Iris）数据集进行分类预测，展示了一个完整的 **监督学习流程**，包括数据预处理、特征标准化、模型训练、评估与特征重要性分析。

---

## 🧩 Dataset | 数据集
- **Source:** [Iris Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/iris)
- **Features (4):**
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target:** Species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

---

## ⚙️ Workflow | 实现流程
1. **Data Loading & Cleaning**  
   Load the CSV file, rename columns, and check for missing values.  
   加载数据集并规范化列名，检查缺失值。  

2. **Encoding & Standardization**  
   Convert categorical species into numeric labels and standardize feature scales.  
   将分类目标编码为数值，并对特征进行标准化处理。  

3. **Model Training**  
   Train a **RandomForestClassifier** using `scikit-learn`.  
   使用 `scikit-learn` 中的随机森林模型进行训练。  

4. **Evaluation**  
   Report accuracy, confusion matrix, and classification metrics.  
   输出准确率、混淆矩阵和分类报告。  

5. **Feature Importance Visualization**  
   Visualize the relative importance of each feature using a bar chart.  
   可视化特征的重要性分布。  

---

## 🧠 Key Results | 核心结果
Example output:

🎯 Accuracy: 0.97

📊 Confusion Matrix:
[[10 0 0]
[ 0 9 1]
[ 0 1 9]]

📋 Classification Report:
precision recall f1-score support
setosa 1.00 1.00 1.00 10
versicolor 0.90 0.90 0.90 10
virginica 0.90 0.90 0.90 10

yaml
Copy code

---

## 🧰 Technologies | 技术栈
- **Python 3.x**
- **pandas**, **NumPy**
- **scikit-learn**
- **matplotlib**

---

## 📂 Project Structure | 项目结构
Iris-RandomForest/
│
├── Iris.csv
├── iris_random_forest.py
├── README.md
└── requirements.txt (optional)

yaml
Copy code

---

## 🚀 How to Run | 运行方法
```bash
# 1️⃣ Install dependencies
pip install pandas numpy scikit-learn matplotlib

# 2️⃣ Run the script
python iris_random_forest.py
🎓 Learning Outcome | 学习收获
Practiced data preprocessing and feature scaling techniques.

Learned to implement a Random Forest baseline model for multi-class classification.

Gained experience in model interpretability via feature importance analysis.

掌握了监督学习模型的构建流程，理解特征标准化、模型训练与评估方法，并学会通过特征重要性图实现模型可解释性。

✍️ Author | 作者
He Yuke (何雨珂)
Software Engineering Student @ UPM
📧 heyukeyoke@gmail.com

🧭 License | 许可协议
This project is released under the MIT License.
