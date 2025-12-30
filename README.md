# TIC (Trauma-Induced Coagulopathy) Prediction Model

## 📋 Project Overview

This project is a machine learning-based prediction system for Trauma-Induced Coagulopathy (TIC). It analyzes patient clinical data using multiple machine learning algorithms to predict the risk of coagulopathy in trauma patients, providing support for clinical decision-making.

**Author:** Peng Hu  
**Last Updated:** December 30, 2025  
**Python Version:** 3.8+

---

## 🎯 Key Features

### 1. Data Processing
- ✅ Multi-source data integration (eICU, MIMIC databases)
- ✅ Missing value imputation (KNN Imputation)
- ✅ Feature engineering and variable selection
- ✅ Data standardization and outlier handling

### 2. Statistical Analysis
- ✅ **T-Test Analysis** - Between-group difference testing for continuous variables
- ✅ **Univariate Logistic Regression** - Risk factor analysis
- ✅ **Forest Plot Visualization** - OR values with 95% confidence intervals
- ✅ **Kolmogorov-Smirnov Test** - Feature distribution consistency testing

### 3. Machine Learning Models
- 🔹 **Logistic Regression**
- 🔹 **Random Forest**
- 🔹 **Gradient Boosting**
- 🔹 **Naive Bayes**
- 🔹 **AdaBoost**

### 4. Model Evaluation
- ✅ **ROC Curves and AUC** - Model discrimination ability
- ✅ **Precision-Recall Curves** - Precision-recall trade-off
- ✅ **Calibration Curves** - Prediction probability calibration
- ✅ **MCC Curves** - Matthews Correlation Coefficient
- ✅ **Bootstrap 95% Confidence Intervals** - Reliability assessment of performance metrics
- ✅ **Cross-Validation** - 5-fold cross-validation

### 5. Model Interpretability
- ✅ **SHAP Value Analysis** - Feature importance and impact direction
- ✅ **Feature Importance Ranking** - Identifying key predictors

---

## 📊 Evaluation Metrics

The project calculates the following performance metrics with 95% confidence intervals:

- **AUROC** - Area Under ROC Curve
- **PR-AUC** - Precision-Recall AUC
- **Accuracy** - Overall accuracy
- **Sensitivity** - Sensitivity (Recall)
- **Specificity** - Specificity
- **Precision** - Precision
- **F-Score** - F1 Score
- **Cohen's Kappa** - Cohen's Kappa coefficient
- **Brier Score** - Brier score (calibration)
- **MCC** - Matthews Correlation Coefficient

---

## 🏗️ Project Structure

```
TIC/
├── TIC.py                    # Main program file
├── README.md                        # Project documentation
├── CODE_REFACTORING_SUMMARY.md     # Code refactoring notes
├── Data/
│   └── output/
│       └── TIC/
│           ├── filled_data1.csv                      # Imputed data
│           ├── feature_info_before_imputation_*.csv  # Feature information
│           ├── ttest_continuous_variables_*.csv      # T-test results
│           ├── univariate_logistic_regression_*.csv  # Univariate regression results
│           ├── forest_plot_*.png                     # Forest plots
│           ├── train_95CI_results_*.csv              # Training set 95%CI
│           ├── insidetest_95CI_results_*.csv         # Internal test set 95%CI
│           ├── outside_95CI_results_*.csv            # External test set 95%CI
│           ├── cross_validation_results_*.csv        # Cross-validation results
│           ├── bootstrap_results_*.csv               # Bootstrap results
│           ├── important_features_*.csv              # Important features
│           ├── *_roc_curves_*.png                    # ROC curves
│           ├── *_precision_recall_curve_*.png        # PR curves
│           ├── *_calibration_curves_*.png            # Calibration curves
│           ├── *_mcc_curve_*.png                     # MCC curves
│           ├── pca_visualization_*.png               # PCA visualization
│           └── shap/
│               └── *-shap_plot-*.png                 # SHAP visualizations
└── HyperTIC/
    ├── Merged file1.csv             # Raw merged data
    └── 特征变量 - 脑部TIC.csv       # Feature variable list
```

---

## 🚀 Quick Start

### Requirements

```bash
Python >= 3.8
numpy >= 1.19.0
pandas >= 1.2.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
xgboost >= 1.3.0
shap >= 0.39.0
imbalanced-learn >= 0.8.0
scipy >= 1.6.0
```

### Installation

```bash
pip install numpy pandas scikit-learn matplotlib xgboost shap imbalanced-learn scipy
```

### Run the Program

```bash
python "TIC.py"
```

---

## 📖 Workflow

### STEP 1: Load Data
- Load raw or preprocessed data
- Calculate DIC scores
- Data cleaning and filtering

### STEP 1.5: T-Test Analysis
- Identify continuous variables
- Perform independent sample t-tests between TIC positive and negative groups
- Generate statistical reports

### STEP 1.6: Univariate Logistic Regression
- Perform univariate analysis on specified variables
- Calculate OR values and 95% confidence intervals
- Generate forest plots

### STEP 2: Split Data
- Split training and test sets
- Separate internal and external validation sets
- Save feature information

### STEP 3: Handle Class Imbalance
- SMOTE oversampling (if needed)
- Random undersampling (if needed)
- Balance training data

### STEP 4: Feature Selection
- Kolmogorov-Smirnov test
- Pearson correlation analysis
- LASSO regression feature selection
- Select optimal feature subset

### STEP 5: Standardization
- StandardScaler standardization
- Z-score outlier handling
- PCA dimensionality reduction visualization
- Outlier detection and processing

### STEP 6: Define Models
- Define multiple machine learning models
- Set regularization parameters to prevent overfitting

### STEP 7: Hyperparameter Optimization
- GridSearchCV grid search
- 5-fold cross-validation
- Select optimal parameter combinations

### STEP 8: Model Evaluation
- Train models
- Calculate 95% confidence intervals
- Generate various evaluation curves
- SHAP value analysis
- Save results to CSV and images

---

## 📈 Output Results

### 1. Statistical Analysis Results
- `ttest_continuous_variables_*.csv` - Complete t-test results
- `univariate_logistic_regression_*.csv` - Univariate regression OR values
- `forest_plot_*.png` - Forest plot visualization

### 2. Model Performance Metrics
- `train_95CI_results_*.csv` - Training set performance
- `insidetest_95CI_results_*.csv` - Internal test set performance
- `outside_95CI_results_*.csv` - External test set performance
- `cross_validation_results_*.csv` - Cross-validation results
- `bootstrap_results_*.csv` - Bootstrap confidence intervals

### 3. Visualization Charts
- **ROC Curves** - Comparison of model discrimination ability
- **Precision-Recall Curves** - Precision-recall trade-offs
- **Calibration Curves** - Prediction probability calibration assessment
- **MCC Curves** - MCC values at different thresholds
- **SHAP Plots** - Feature contribution to predictions
- **PCA Visualization** - Distribution of training and test sets

### 4. Feature Importance
- `important_features_*.csv` - Feature importance ranking for each model

---

## 🔬 Core Algorithms

### 1. DIC Score Calculation

Coagulopathy definition (one or more of the following):
- **PLT** < 100×10⁹/L
- **INR** > 1.25
- **PT** > 14 seconds
- **APTT** > 36 seconds
- **FIB** < 2.0 g/L

### 2. Feature Selection Strategy

1. **Correlation Screening** - Select top 16 features with highest correlation to target
2. **LASSO Regularization** - Further refine features through L1 regularization
3. **Cross-Validation** - LassoCV automatically selects optimal alpha value

### 3. Overfitting Prevention Strategies

- **Regularization** - All models use regularization parameters
- **Cross-Validation** - 5-fold cross-validation assessment
- **Depth Limitation** - Limit maximum depth of tree models
- **Minimum Samples** - Increase minimum samples for splitting and leaf nodes
- **Feature Sampling** - Random Forest uses sqrt(n) features

---

## 📊 Data Sources

This project integrates multiple clinical databases:

- **Source 0** - MIMIC Database
- **Source 1** - eICU Database
- **Source 2** - Other clinical data
- **Source 3** - Local clinical data

Brain surgery patients (Brain surgery = 1)

---

## ⚙️ Configuration

### Key Parameters

```python
# Label column
labelCol = 'TIC'

# Load processed data or reprocess
loadedData = 0  # 0: Reprocess, 1: Load processed data

# LASSO feature count
top_features_count = 16  # Select top 16 features with highest correlation

# Model parameters (Example - Random Forest)
n_estimators = 50        # Number of trees
max_depth = 3            # Maximum depth
min_samples_split = 30   # Minimum samples for splitting
min_samples_leaf = 15    # Minimum samples in leaf nodes
```

### Font Configuration

```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese font
plt.rcParams['axes.unicode_minus'] = False     # Minus sign display
```

---

## 📝 Variable Descriptions

### Input Features

See `特征变量 - 脑部TIC.csv` file, including:

- **Demographic Features** - Age, gender, etc.
- **Vital Signs** - Blood pressure, heart rate, etc.
- **Laboratory Tests** - Complete blood count, coagulation function, biochemical markers, etc.
- **Injury-Related** - Injury mechanism, GCS score, etc.
- **Complications** - ARDS, kidney injury, sepsis, etc.

### Target Variable

- **TIC** - Trauma-Induced Coagulopathy (0: Negative, 1: Positive)

---

## 🔍 Model Interpretability

### SHAP (SHapley Additive exPlanations)

The project uses SHAP values to analyze each feature's contribution to model predictions:

- **TreeExplainer** - For tree models (Random Forest, Gradient Boosting)
- **LinearExplainer** - For linear models (Logistic Regression)
- **KernelExplainer** - For other models

SHAP plots display:
- Feature importance ranking
- Positive/negative impact of feature values on predictions
- Feature interaction effects

---

## ⚠️ Important Notes

1. **Data Privacy** - Ensure all patient data is de-identified
2. **Computational Resources** - Hyperparameter optimization may take considerable time
3. **Random Seed** - All random processes use a fixed seed (random_state=42)
4. **Clinical Validation** - Model results require validation by clinical experts
5. **Scope of Application** - Only applicable to brain surgery trauma patients

---

## 📚 References

1. Trauma-Induced Coagulopathy (TIC) - Definition and Clinical Significance
2. Machine Learning in Critical Care Medicine
3. SHAP: A Unified Approach to Interpreting Model Predictions
4. Handling Imbalanced Datasets in Medical Research

---

## 🤝 Contributing

Contributions and suggestions are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is for research and educational purposes only. Please consult medical professionals before clinical application.

---

## 📧 Contact

**Author:** Peng Hu  
**Email:** [Your Email]  
**Institution:** [Your Institution]

---

## 🔄 Changelog

### Version 2.0 (2025-12-30)
- ✅ Code refactoring and optimization
- ✅ Added T-test analysis
- ✅ Added univariate logistic regression and forest plots
- ✅ Improved feature selection workflow
- ✅ Optimized outlier handling
- ✅ Enhanced model evaluation metrics
- ✅ Improved documentation and comments

### Version 1.0
- Initial release
- Basic machine learning prediction functionality

---

## 🎓 Acknowledgments

Special thanks to all medical staff involved in data collection and annotation.

---

**⚕️ Disclaimer: This tool is for research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.**

---
---

<br><br>

# TIC (创伤性凝血功能障碍) 预测模型

## 📋 项目概述

本项目是一个基于机器学习的创伤性凝血功能障碍（TIC）预测系统。通过分析患者的临床数据，使用多种机器学习算法预测创伤患者发生凝血功能障碍的风险，为临床决策提供支持。

**作者：** 胡鹏  
**最后更新：** 2025年12月30日  
**Python版本：** 3.8+

---

## 🎯 主要功能

### 1. 数据处理
- ✅ 多源数据整合（eICU、MIMIC等数据库）
- ✅ 缺失值填充（KNN填补）
- ✅ 特征工程和变量筛选
- ✅ 数据标准化和异常值处理

### 2. 统计分析
- ✅ **T检验分析** - 连续变量的组间差异检验
- ✅ **单因素逻辑回归** - 风险因素分析
- ✅ **森林图可视化** - OR值及95%置信区间展示
- ✅ **Kolmogorov-Smirnov检验** - 特征分布一致性检验

### 3. 机器学习模型
- 🔹 **逻辑回归** (Logistic Regression)
- 🔹 **随机森林** (Random Forest)
- 🔹 **梯度提升** (Gradient Boosting)
- 🔹 **朴素贝叶斯** (Naive Bayes)
- 🔹 **自适应提升** (AdaBoost)

### 4. 模型评估
- ✅ **ROC曲线和AUC值** - 模型判别能力
- ✅ **Precision-Recall曲线** - 精确率-召回率权衡
- ✅ **校准曲线** - 预测概率校准度
- ✅ **MCC曲线** - Matthews相关系数
- ✅ **Bootstrap 95%置信区间** - 性能指标的可靠性评估
- ✅ **交叉验证** - 5折交叉验证

### 5. 模型解释
- ✅ **SHAP值分析** - 特征重要性和影响方向
- ✅ **特征重要性排序** - 识别关键预测因子

---

## 📊 评估指标

本项目计算以下性能指标及其95%置信区间：

- **AUROC** - ROC曲线下面积
- **PR-AUC** - Precision-Recall曲线下面积
- **准确率** (Accuracy)
- **灵敏度** (Sensitivity/Recall)
- **特异度** (Specificity)
- **精确率** (Precision)
- **F1分数** (F-Score)
- **Cohen's Kappa系数**
- **Brier评分** - 校准度评估
- **MCC** - Matthews相关系数

---

## 📈 输出结果

### 1. 统计分析结果
- `ttest_continuous_variables_*.csv` - T检验完整结果
- `univariate_logistic_regression_*.csv` - 单因素回归OR值
- `forest_plot_*.png` - 森林图可视化

### 2. 模型性能指标
- `train_95CI_results_*.csv` - 训练集性能
- `insidetest_95CI_results_*.csv` - 内部测试集性能
- `outside_95CI_results_*.csv` - 外部测试集性能
- `cross_validation_results_*.csv` - 交叉验证结果
- `bootstrap_results_*.csv` - Bootstrap置信区间

### 3. 可视化图表
- **ROC曲线** - 各模型的判别能力比较
- **Precision-Recall曲线** - 精确率-召回率权衡
- **校准曲线** - 预测概率校准度评估
- **MCC曲线** - 不同阈值的MCC值
- **SHAP图** - 特征对预测的贡献度
- **PCA可视化** - 训练集和测试集分布

### 4. 特征重要性
- `important_features_*.csv` - 各模型的特征重要性排序

---

## 🔬 核心算法

### 1. DIC评分计算

凝血功能障碍定义（满足以下一项或多项）：
- **血小板** < 100×10⁹/L
- **INR** > 1.25
- **PT** > 14秒
- **APTT** > 36秒
- **纤维蛋白原** < 2.0 g/L

### 2. 特征选择策略

1. **相关性筛选** - 选择与目标变量相关性最高的16个特征
2. **LASSO正则化** - 通过L1正则化进一步精简特征
3. **交叉验证** - LassoCV自动选择最优alpha值

### 3. 模型防过拟合策略

- **正则化** - 所有模型使用正则化参数
- **交叉验证** - 5折交叉验证评估
- **深度限制** - 限制树模型的最大深度
- **最小样本数** - 增加分裂和叶节点的最小样本数
- **特征采样** - 随机森林使用sqrt(n)个特征

---

## 📊 数据源

本项目整合多个临床数据库：

- **数据源0** - MIMIC数据库
- **数据源1** - eICU数据库
- **数据源2** - 其他临床数据
- **数据源3** - 本地临床数据

研究对象：脑外科手术患者（Brain surgery = 1）

---

## ⚙️ 配置说明

### 关键参数

```python
# 标签列
labelCol = 'TIC'

# 加载已处理数据或重新处理
loadedData = 0  # 0: 重新处理, 1: 加载已处理数据

# LASSO特征数量
top_features_count = 16  # 选择相关性最高的前16个特征

# 模型参数（示例 - 随机森林）
n_estimators = 50        # 树的数量
max_depth = 3            # 最大深度
min_samples_split = 30   # 分裂最小样本数
min_samples_leaf = 15    # 叶节点最小样本数
```

---

## ⚠️ 注意事项

1. **数据隐私** - 确保所有患者数据已去标识化
2. **计算资源** - 超参数优化可能需要较长时间
3. **随机种子** - 所有随机过程使用固定种子（random_state=42）
4. **临床验证** - 模型结果需要临床专家验证
5. **适用范围** - 仅适用于脑外科创伤患者

---

## 🎓 致谢

感谢所有参与数据收集和标注的医护人员。

---

**⚕️ 免责声明：本工具仅用于研究目的，不应作为专业医疗建议、诊断或治疗的替代品。**
