# -*- coding: gbk -*-
"""
TIC (Trauma-Induced Coagulopathy) Prediction Model
Author: Peng Hu
Date: 2025-12-30
Description: Machine learning models for predicting trauma-induced coagulopathy
"""

# Standard library imports
import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy import stats

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error, roc_curve, auc,
                              matthews_corrcoef, brier_score_loss, make_scorer,
                              precision_recall_curve, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix, metrics)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.decomposition import PCA

# XGBoost import
from xgboost import XGBClassifier

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def bootstrap_confidence_interval(y_true, y_scores, y_pred, metric_func, n_bootstrap=1000):
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        y_true_resample, y_scores_resample, y_pred_resample = resample(y_true, y_scores, y_pred)
        bootstrap_sample = metric_func(y_true_resample, y_scores_resample, y_pred_resample)
        bootstrap_samples.append(bootstrap_sample)

    bootstrap_samples = np.array(bootstrap_samples)
    lower = np.percentile(bootstrap_samples, 2.5)
    upper = np.percentile(bootstrap_samples, 97.5)

    return lower, upper


def calculate_metrics(model, X_Data, y_Data, name):
    # 初始化一个字典来保存每个指标的所有样本值
    metrics_samples = {
        'auroc': [],
        'pr_auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f_score': [],
        'cohen_kappa': []  # 新增 Cohen's Kappa
    }

    # 进行1000次自助法抽样
    for _ in range(10):
        # 对数据进行自助法抽样
        X_resample, y_resample = resample(X_Data, y_Data)

        # 预测概率和标签
        y_scores = model.predict_proba(X_resample)[:, 1]
        y_pred = model.predict(X_resample)

        # 计算指标
        precision, recall, _ = metrics.precision_recall_curve(y_resample, y_scores)
        sort_idx = np.argsort(recall)
        precision = precision[sort_idx]
        recall = recall[sort_idx]
        if len(set(y_resample)) > 1:
            metrics_samples['auroc'].append(metrics.roc_auc_score(y_resample, y_scores))
        else: 
            metrics_samples['auroc'].append(0)
        metrics_samples['pr_auc'].append(metrics.auc(recall, precision))
        metrics_samples['accuracy'].append(metrics.accuracy_score(y_resample, y_pred))
        metrics_samples['sensitivity'].append(metrics.recall_score(y_resample, y_pred))
        metrics_samples['specificity'].append(metrics.recall_score(1-y_resample, 1-y_pred))
        metrics_samples['precision'].append(metrics.precision_score(y_resample, y_pred, zero_division=1))
        metrics_samples['f_score'].append(metrics.f1_score(y_resample, y_pred))
        metrics_samples['cohen_kappa'].append(metrics.cohen_kappa_score(y_resample, y_pred))  # 计算 Cohen's Kappa

    # 计算每个指标的95%置信区间
    metrics_dict = {}
    for metric, samples in metrics_samples.items():
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)
        metrics_dict[f'{metric}'] = np.mean(samples)
        metrics_dict[f'{metric}_lower'] = lower
        metrics_dict[f'{metric}_upper'] = upper

    df = pd.DataFrame(metrics_dict, index=[name])
    return df


def calculate_mcc_curve(y_val_prob, X_val, y_val):
    """Calculate MCC curve for threshold selection"""
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    mcc_scores = [matthews_corrcoef(y_val, y_val_prob >= t) for t in thresholds]
    return thresholds, mcc_scores


def plot_mcc_curves(results, save_path):
    """Plot MCC curves for different models"""
    plt.figure(figsize=(10, 6))
    
    for thresholds, mcc_scores, label in results:
        max_mcc = max(mcc_scores)
        plt.plot(thresholds, mcc_scores, label=f'{label} (Max MCC: {max_mcc:.2f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title('MCC Curves for Different Models')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curves(pr_curves, title, save_path):
    """Plot Precision-Recall curves for different models"""
    plt.figure(figsize=(8, 6))
    for precision, recall, name, score in pr_curves:
        plt.plot(recall, precision, label=f'{name} ({score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curves(calibration_data, title, save_path):
    """Plot calibration curves for different models"""
    plt.figure(figsize=(8, 6))
    for fraction_of_positives, mean_predicted_value, name, score in calibration_data:
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=f'{name} (Brier: {score:.2f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def calculate_dic_scores(row):
    """
    Calculate DIC (Disseminated Intravascular Coagulation) scores
    Coagulopathy defined as one or more of:
    - PLT <100*10^9/L
    - INR > 1.25
    - PT > 14 sec
    - APTT > 36 sec
    - FIB < 2.0 g/L
    """
    pt_score = 1 if pd.notna(row['Post-PT']) and row['Post-PT'] > 14 else 0
    inr_score = 1 if pd.notna(row['Post-INR']) and row['Post-INR'] > 1.25 else 0
    aptt_score = 1 if pd.notna(row['Post-APTT']) and row['Post-APTT'] > 36 else 0
    tt_score = 0  # Currently not used
    fib_score = 0  # Currently not used
    d_dimer_score = 0  # Currently not used
    plt_score = 1 if pd.notna(row['Post-PLT']) and row['Post-PLT'] < 100 else 0
def calculate_dic(row):
    # 计算各个评分
    scores = calculate_dic_scores(row)
    
    # 计算 TIC 列值
    dic_score = scores.sum()
    if dic_score >= 1:
        return 1
    else:
        return 0

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号
labelCol='TIC'

# 1.加载数据
#region  load data

loadedData=0
if loadedData == 1:
    # Load pre-processed data
    path = r"./DIC/Data/output/TIC/filled_data1.csv"
    allData = pd.read_csv(path, encoding='UTF-8')
    filtered_data = allData[allData['source'].isin([0, 1, 2, 3])]
    
    # Calculate TIC positive rate for each source
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['TIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'Source {source} TIC positive rate: {positive_rate:.2%}, count: {positive_count}')
else:
    path=str(r"./DIC/HyperTIC/Merged file1.csv")
    data = pd.read_csv(path,encoding='UTF-8')
    

    # Calculate DIC scores and add to dataframe
    score_columns = data.apply(calculate_dic_scores, axis=1)
    data = pd.concat([data, score_columns], axis=1)
    
    # Add TIC classification column
    data['TIC'] = data.apply(lambda row: calculate_dic(row), axis=1)


# ============================================================================
# CONFIGURATION
# ============================================================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
labelCol = 'TIC'

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 1: LOAD DATA')
print(f'{"="*80}\n')

loadedData = 0

if loadedData == 1:
    # Load pre-processed data
    path = r"./DIC/Data/output/TIC/filled_data1.csv"
    allData = pd.read_csv(path, encoding='UTF-8')
    filtered_data = allData[allData['source'].isin([0, 1, 2, 3])]
    
    # Calculate TIC positive rate for each source
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['TIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'Source {source} TIC positive rate: {positive_rate:.2%}, count: {positive_count}')
else:
    # Load raw data and process
    path = r"./DIC/HyperTIC/Merged file1.csv"
    data = pd.read_csv(path, encoding='UTF-8')
    
    # Calculate DIC scores and add to dataframe
    score_columns = data.apply(calculate_dic_scores, axis=1)
    data = pd.concat([data, score_columns], axis=1)
    
    # Add TIC classification column
    data['TIC'] = data.apply(lambda row: calculate_dic(row), axis=1)

    # Data filtering - remove rows with missing critical values
    data = data.dropna(subset=['Pre-PLT', 'Pre-APTT', 'Post-APTT'])
    data = data[data['Brain surgery'] == 1]
    
    # Filter data by sources
    filtered_data = data
    
    # Analyze positive rates for each score by source
    score_columns = ['PT_score', 'INR_score', 'APTT_score', 'TT_score', 
                     'FIB_score', 'DD_score', 'PLT_score', 'TIC']
    
    print("Positive rates for each score by source:")
    print("="*80)
    
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        total_count = source_data.shape[0]
        
        print(f'\nSource {source} (Total: {total_count}):')
        print("-" * 40)
        
        for score_col in score_columns:
            if score_col in source_data.columns:
                positive_count = source_data[score_col].sum()
                positive_rate = positive_count / total_count if total_count > 0 else 0
                print(f'  {score_col:15}: {positive_rate:.2%} ({positive_count}/{total_count})')
            else:
                print(f'  {score_col:15}: Column not found')
    
    print("\n" + "="*80)
    
    
    # 删除 source 列为 0 且 Date of injury 列为空的行
    #data = data[~((data['source'] == 0) & (data['Date of injury'].isna()))]
    
    # drop_ratios = {
    #     0: 0.0,  # 去掉 20% 的非阳性数据
    #     1: 0.5,  # 去掉 30% 的非阳性数据
    #     2: 0.4,  # 去掉 40% 的非阳性数据
    #     3: 0.0   # 去掉 50% 的非阳性数据
    # }
    # # 遍历每个 source
    # for source in [0, 1, 2, 3]:
    #     source_data = filtered_data[filtered_data['source'] == source]
    #     non_positive_data = source_data[source_data['TIC'] == 0]
    #     positive_data = source_data[source_data['TIC'] == 1]
    
    #     # 计算需要去掉的非阳性数据数量
    #     drop_count = int(len(non_positive_data) * drop_ratios[source])
    
    #     # 随机抽样去掉非阳性数据
    #     non_positive_data_sampled = non_positive_data.sample(frac=1-drop_ratios[source], random_state=42)
    
    #     # 合并保留的非阳性数据和阳性数据
    #     source_data_sampled = pd.concat([non_positive_data_sampled, positive_data])
    
    #     # 更新 filtered_data
    #     if source == 0:
    #         filtered_data_sampled = source_data_sampled
    #     else:
    #         filtered_data_sampled = pd.concat([filtered_data_sampled, source_data_sampled])


    # 更新 filtered_data
    #data = filtered_data_sampled
    
    # 筛选出 sources 为 0、1、2、3 的数据
    filtered_data = data
    # 计算每个 source 中 TIC 的阳性率
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['TIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'筛选后 Source {source} 的 TIC 阳性率: {positive_rate:.2%}, 阳性总数: {positive_count}')
    feature_vars=pd.read_csv(str(r"./DIC/HyperTIC/特征变量 - 脑部TIC.csv"),encoding='UTF-8')
    # 筛选出类型不是0的行
    print(f'筛选出类型不是0的行')
    selected_rows = feature_vars[feature_vars['类型'] == 1]
    # 获取这些行的列名
    selected_columns = selected_rows['列名'].str.strip()
    usedColumns=selected_columns.tolist()
    # if 'Surgery start time' not in usedColumns:
    #     usedColumns.append('Surgery start time')
    if 'source' not in usedColumns:
        usedColumns.append('source')
    if 'year_group' not in usedColumns:
        usedColumns.append('year_group')
    
    if 'Admission time' not in usedColumns:
        usedColumns.append('Admission time')
    if 'TIC' not in usedColumns:
        usedColumns.append('TIC')
    data=data[usedColumns]
    
    # 检查重复的列名
    duplicated_columns = data.columns[data.columns.duplicated()]

    # 打印出重复的列名
    print(f"Duplicated columns: {duplicated_columns.tolist()}")

    # 删除重复的列
    data = data.loc[:, ~data.columns.duplicated()]

    for col in data.columns:
        if (col!='Surgery start time') & (col!='Admission time'):
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 计算每列的缺失值比例
    print(f'计算每列的缺失值比例')        
    missing_ratio = data.isnull().sum() / len(data)
    # 找出缺失值比例大于0.3的列
    columns_to_drop = missing_ratio[(missing_ratio > 0.9)& (missing_ratio.index != 'Admission time') & (missing_ratio.index != 'year_group')].index
    # 删除这些列
    print(f'删除这些缺失值比例大于0.42的列: {len(columns_to_drop)}')
    
    columns_to_dropRate = missing_ratio[missing_ratio > 0.9]
    for column, ratio in columns_to_dropRate.items():
        print(f'列名: {column}, 缺失比例: {ratio:.2%}')
    #data = data.drop(columns_to_drop, axis=1)

    # ============== 在填充缺失值之前，保存各特征变量的信息 ==============
    print(f'保存填充前的特征变量信息...')
    
    # 初始化变量分类列表
    continuous_vars_info = []
    categorical_vars_info = []
    binary_vars_info = []
    
    # 创建特征信息DataFrame
    feature_info_list = []
    
    for col in usedColumns:
        if (col == 'Surgery start time') or (col == 'Admission time'):
            continue
            
        # 统计信息
        total_count = len(data[col])
        missing_count = data[col].isnull().sum()
        missing_rate = missing_count / total_count if total_count > 0 else 0
        unique_count = data[col].nunique()
        
        # 判断变量类型和编码方式
        if unique_count < 4:
            var_type = '二分类变量'
            coding_process = '无需编码（0/1）'
            binary_vars_info.append(col)
        elif unique_count < 10:
            var_type = '分类变量'
            coding_process = '序数编码(OrdinalEncoder)'
            categorical_vars_info.append(col)
        else:
            var_type = '连续变量'
            coding_process = '标准化(StandardScaler)'
            continuous_vars_info.append(col)
        
        # 保存特征信息
        feature_info_list.append({
            '变量名': col,
            '类型': var_type,
            '编码处理': coding_process,
            '总数': total_count,
            '缺失数': missing_count,
            '缺失率': f'{missing_rate:.4f}',
            '缺失率(%)': f'{missing_rate*100:.2f}%',
            '唯一值数量': unique_count
        })
    
    # 转换为DataFrame并保存
    feature_info_df = pd.DataFrame(feature_info_list)
    
    # 获取当前时间戳
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    
    # 保存到CSV文件
    output_path = f'./DIC/Data/output/TIC/feature_info_before_imputation_{now_str}.csv'
    feature_info_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'特征变量信息已保存到: {output_path}')
    
    # 打印统计摘要
    print(f'\n特征变量统计摘要:')
    print(f'  二分类变量数量: {len(binary_vars_info)}')
    print(f'  分类变量数量: {len(categorical_vars_info)}')
    print(f'  连续变量数量: {len(continuous_vars_info)}')
    print(f'  总特征数量: {len(feature_info_list)}')
    print(f'  平均缺失率: {feature_info_df["缺失率"].astype(float).mean():.4f}')
    # ============== 特征信息保存完成 ==============

    #缺失值填充    
    continuous_vars = []
    categorical_vars = []
    binary_vars = []
    # 遍历每一列
    for col in data.columns:
        if (col=='Surgery start time') or (col=='Admission time') :
            continue
        if data[col].nunique() < 4:
            binary_vars.append(col)
        # 如果唯一值数量小于10，我们假设它是分类的
        else:
            if data[col].nunique() < 10:
                categorical_vars.append(col)
            # 否则，我们假设它是连续的
            else:
                continuous_vars.append(col)
        
    df=data
    df[binary_vars] = df[binary_vars].fillna(0)
    # 对于连续变量，我们可以直接使用KNNImputer
    print(f'缺失值填充：对于连续变量，我们可以直接使用KNNImputer')
    imputer = KNNImputer(n_neighbors=5)
    df[continuous_vars] = imputer.fit_transform(df[continuous_vars])
    if len(categorical_vars)>0:
        # 对于分类变量，我们需要先将其转换为数值形式
        encoder = OrdinalEncoder()
        df[categorical_vars] = encoder.fit_transform(df[categorical_vars])

        # 然后我们可以使用KNNImputer
        df[categorical_vars] = imputer.fit_transform(df[categorical_vars])

        # 最后，我们可以将数值转换回原来的类别
        df[categorical_vars] = encoder.inverse_transform(df[categorical_vars])

    allData = df
    allData.to_csv(r'./DIC/Data/output/TIC/filled_data1.csv', index=False,encoding='utf-8')


# ============================================================================
# STEP 1.5: T-TEST ANALYSIS FOR CONTINUOUS VARIABLES
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 1.5: T-TEST ANALYSIS FOR CONTINUOUS VARIABLES')
print(f'{"="*80}\n')

from scipy import stats

# 识别连续变量（排除二值变量和分类变量，以及时间列）
continuous_vars_for_ttest = []
for col in allData.columns:
    if col in ['TIC', 'source', 'Surgery start time', 'Admission time', 'year_group']:
        continue
    if col in continuous_vars:  # 使用之前定义的continuous_vars列表
        continuous_vars_for_ttest.append(col)

print(f'连续变量数量: {len(continuous_vars_for_ttest)}')

# 准备结果列表
ttest_results = []

# 对每个连续变量进行t检验
for var in continuous_vars_for_ttest:
    # 分组：TIC=1和TIC=0
    group_tic_1 = allData[allData['TIC'] == 1][var].dropna()
    group_tic_0 = allData[allData['TIC'] == 0][var].dropna()
    
    # 计算均值和标准差
    mean_tic_1 = group_tic_1.mean()
    std_tic_1 = group_tic_1.std()
    mean_tic_0 = group_tic_0.mean()
    std_tic_0 = group_tic_0.std()
    
    # 执行独立样本t检验
    if len(group_tic_1) > 1 and len(group_tic_0) > 1:
        t_stat, p_value = stats.ttest_ind(group_tic_1, group_tic_0, equal_var=False)  # Welch's t-test
    else:
        t_stat, p_value = np.nan, np.nan
    
    # 保存结果
    ttest_results.append({
        'Variable': var,
        'TIC=1 Mean±SD': f'{mean_tic_1:.4f}±{std_tic_1:.4f}',
        'TIC=1 Mean': mean_tic_1,
        'TIC=1 SD': std_tic_1,
        'TIC=1 N': len(group_tic_1),
        'TIC=0 Mean±SD': f'{mean_tic_0:.4f}±{std_tic_0:.4f}',
        'TIC=0 Mean': mean_tic_0,
        'TIC=0 SD': std_tic_0,
        'TIC=0 N': len(group_tic_0),
        't-statistic': t_stat,
        'P-value': p_value
    })

# 转换为DataFrame
ttest_df = pd.DataFrame(ttest_results)

# 按P值排序
ttest_df = ttest_df.sort_values('P-value')

# 获取当前时间戳
now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d_%H%M%S')

# 保存到CSV
ttest_df.to_csv(f'./DIC/Data/output/TIC/ttest_continuous_variables_{now_str}.csv', index=False, encoding='utf-8-sig')
print(f'T检验结果已保存到: ./DIC/Data/output/TIC/ttest_continuous_variables_{now_str}.csv')

# 打印显著性结果（P<0.05）
significant_vars = ttest_df[ttest_df['P-value'] < 0.05]
print(f'\n显著性变量数量 (P<0.05): {len(significant_vars)}')
print('\n前10个最显著的变量:')
print(ttest_df[['Variable', 'TIC=1 Mean±SD', 'TIC=0 Mean±SD', 'P-value']].head(10).to_string(index=False))


# ============================================================================
# STEP 1.6: UNIVARIATE LOGISTIC REGRESSION AND FOREST PLOT
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 1.6: UNIVARIATE LOGISTIC REGRESSION AND FOREST PLOT')
print(f'{"="*80}\n')

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

variables_for_regression = ['In-Hospital Death','Post-ARDS','Hemothorax','Post-delirium','Post-kidney injury',
                            'Post-PE','Post-DVT','less than 96','Post-sepsis','Post-respiratory failure'
                            ,'ABNORMAL COAGULATION','Endotracheal tube','Invasive mechanical ventilation',
                            'ECMO OR TRACH',
                            'Hypercholesterolemia',
                            'End stage renal disease',
                            'CARDIAC ARREST',
                            'Anemia',
                            'Renal failure',
                            'ACUTE KIDNEY FAILURE',
                            'Urinary tract infection',
                            'Pneumonia','Reoperation',
                            'Hospitalizations',
                            'Pulmonary embolism']
# 准备结果列表
regression_results = []

# 对每个变量进行单因素逻辑回归
for var in variables_for_regression:
    if var not in allData.columns:
        print(f'警告: 变量 {var} 不存在于数据中，跳过')
        continue
    
    # 准备数据（去除缺失值）
    data_subset = allData[[var, 'TIC']].dropna()
    
    if len(data_subset) < 10:
        print(f'警告: 变量 {var} 有效样本量不足，跳过')
        continue
    
    X_var = data_subset[[var]].values
    y_var = data_subset['TIC'].values
    
    # 训练逻辑回归模型
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_var, y_var)
    
    # 获取系数（log odds）
    coef = lr_model.coef_[0][0]
    
    # 计算OR值
    or_value = np.exp(coef)
    
    # 计算95%置信区间
    # 使用bootstrap方法估计置信区间
    n_bootstrap = 1000
    or_bootstrap = []
    
    for _ in range(n_bootstrap):
        # Bootstrap抽样
        indices = np.random.choice(len(X_var), size=len(X_var), replace=True)
        X_boot = X_var[indices]
        y_boot = y_var[indices]
        
        # 训练模型
        try:
            lr_boot = LogisticRegression(max_iter=1000)
            lr_boot.fit(X_boot, y_boot)
            coef_boot = lr_boot.coef_[0][0]
            or_bootstrap.append(np.exp(coef_boot))
        except:
            continue
    
    # 计算95%置信区间
    ci_lower = np.percentile(or_bootstrap, 2.5)
    ci_upper = np.percentile(or_bootstrap, 97.5)
    
    # 计算P值（使用Wald检验）
    from scipy import stats as sp_stats
    # 标准误估计
    std_err = np.std(or_bootstrap) / np.sqrt(len(or_bootstrap))
    z_score = coef / (std_err / or_value) if std_err > 0 else 0
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))
    
    # 保存结果
    regression_results.append({
        'Variable': var,
        'OR': or_value,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'OR (95% CI)': f'{or_value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})',
        'P-value': p_value,
        'N': len(data_subset)
    })
    
    print(f'{var}: OR={or_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), P={p_value:.4f}')

# 转换为DataFrame
regression_df = pd.DataFrame(regression_results)

# 保存到CSV
now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d_%H%M%S')
regression_df.to_csv(f'./DIC/Data/output/TIC/univariate_logistic_regression_{now_str}.csv', index=False, encoding='utf-8-sig')
print(f'\n单因素逻辑回归结果已保存到: ./DIC/Data/output/TIC/univariate_logistic_regression_{now_str}.csv')

# 绘制森林图
if len(regression_results) > 0:
    fig, ax = plt.subplots(figsize=(10, len(regression_results) * 0.6 + 2))
    
    # 反转顺序，使第一个变量在顶部
    regression_df_sorted = regression_df.iloc[::-1].reset_index(drop=True)
    
    y_positions = range(len(regression_df_sorted))
    
    # 绘制OR值和置信区间
    for i, row in regression_df_sorted.iterrows():
        # 绘制置信区间线
        ax.plot([row['CI_lower'], row['CI_upper']], [i, i], 'k-', linewidth=2)
        
        # 绘制OR点
        color = 'red' if row['P-value'] < 0.05 else 'blue'
        ax.plot(row['OR'], i, 'o', markersize=8, color=color)
        
        # 添加文本标签（OR和95%CI）
        label_text = f"{row['OR']:.2f} ({row['CI_lower']:.2f}-{row['CI_upper']:.2f})"
        ax.text(row['CI_upper'] + 0.1, i, label_text, va='center', fontsize=9)
    
    # 添加参考线（OR=1）
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    
    # 设置y轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(regression_df_sorted['Variable'])
    
    # 设置x轴
    ax.set_xlabel('Odds Ratio (OR)', fontsize=12, fontweight='bold')
    ax.set_title('Forest Plot - Univariate Logistic Regression for TIC', fontsize=14, fontweight='bold')
    
    # 设置x轴范围
    all_values = list(regression_df_sorted['CI_lower']) + list(regression_df_sorted['CI_upper'])
    x_min = max(0, min(all_values) - 0.5)
    x_max = max(all_values) + 1
    ax.set_xlim(x_min, x_max)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='P < 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='P ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'./DIC/Data/output/TIC/forest_plot_{now_str}.png', dpi=300, bbox_inches='tight')
    print(f'森林图已保存到: ./DIC/Data/output/TIC/forest_plot_{now_str}.png')
    plt.close()


# ============================================================================
# STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS')
print(f'{"="*80}\n')

# Handle NaN values if present
allData['Admission time'] = allData['Admission time'].fillna('')

# Ensure 'Admission time' column is of string type
allData['Admission time'] = allData['Admission time'].astype(str)

# 筛选出 inSideData，条件是('source'==3 且‘year_group’<2017) 或者 ('source'!=3 且‘Admission time’的前4位<2021)
# 尝试将 'Admission time' 列转换为时间格式
def extract_year(admission_time):
    try:
        ddd=allData['Admission time']
        return pd.to_datetime(admission_time).year
    except ValueError:
        return int(admission_time[:4])

# 应用到 'Admission time' 列
allData['Admission time'] = allData['Admission time'].apply(extract_year)

#inSideData = allData.query("(`source` == 3 and `year_group` < 2017) or (`source` != 3 and `Admission time`.str[:4].astype(int) < 2022)")
# inSideData = allData[
#     ((allData['source'] == 3) & (allData['year_group'] < 2015)) |
#     ((allData['source'] != 3) & (allData['Admission time'] < 2021))
# ]
allData =allData [allData['source'].isin([0,1,3])]
inSideData = allData[allData['source'].isin([0,1])]
# 其余数据放到 outSideData 中
outSideData = allData[~allData.index.isin(inSideData.index)]

positive_count = inSideData['TIC'].sum()
total_count = inSideData.shape[0]
positive_rate = positive_count / total_count if total_count > 0 else 0
print(f'inSideData  的总数：{total_count}， TIC 阳性率: {positive_rate:.2%}, 阳性总数: {positive_count}')

positive_count = outSideData['TIC'].sum()
total_count = outSideData.shape[0]
positive_rate = positive_count / total_count if total_count > 0 else 0
print(f'outSideData  的总数：{total_count}， TIC 阳性率: {positive_rate:.2%}, 阳性总数: {positive_count}')    

for source in [0, 1,2, 3]:
    source_data = inSideData[inSideData['source'] == source]
    positive_count = source_data['TIC'].sum()
    total_count = source_data.shape[0]
    positive_rate = positive_count / total_count if total_count > 0 else 0
    print(f'inSideData: Source {source}  的总数：{total_count}， TIC 阳性率: {positive_rate:.2%}, 阳性总数: {positive_count}')
for source in [0, 1,2, 3]:
    source_data = outSideData[outSideData['source'] == source]
    positive_count = source_data['TIC'].sum()
    total_count = source_data.shape[0]
    positive_rate = positive_count / total_count if total_count > 0 else 0
    print(f'outSideData: Source {source}  的总数：{total_count}， TIC 阳性率: {positive_rate:.2%}, 阳性总数: {positive_count}')

inSideData = inSideData.sample(frac=1).reset_index(drop=True)
inSideData = inSideData.sample(frac=1).reset_index(drop=True)

grouped_stats = allData.groupby('source').describe()
print(grouped_stats)
grouped_stats.to_csv(r'./DIC/Data/output/TIC/grouped_stats.csv')

outSideData = outSideData.drop('source', axis=1)
inSideData = inSideData.drop('source', axis=1)
X = inSideData.drop(labelCol, axis=1)
y = inSideData[labelCol]

X_outTest=outSideData.drop(labelCol, axis=1)
y_outTest=outSideData[labelCol]

X_All=allData.drop('source', axis=1).drop(labelCol, axis=1)
y_All=allData.drop('source', axis=1)[labelCol]

X_train, X_insidetest, y_train, y_insidetest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ============================================================================
# STEP 3: HANDLE CLASS IMBALANCE (SMOTE/UNDERSAMPLING)
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 3: HANDLE CLASS IMBALANCE')
print(f'{"="*80}\n')

outTestRate = y_outTest.sum()/len(y_outTest)
print(f'outTestRate: {outTestRate}')
trainRate= y.sum()/len(y)
print(f'trainTestRate:{trainRate}')
useSmote=outTestRate>trainRate
targetNumber=int(len(y) * outTestRate)
print(f'targetNumber:{targetNumber}')

if(useSmote):
    smote = SMOTE(sampling_strategy={1:targetNumber}, random_state=42)
    X, y = smote.fit_resample(X, y)
else:
    undersample = RandomUnderSampler(sampling_strategy={1: targetNumber})
    X, y = undersample.fit_resample(X, y)
    

X_train, X_insidetest, y_train, y_insidetest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ============================================================================
# STEP 4: FEATURE SELECTION
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 4: FEATURE SELECTION')
print(f'{"="*80}\n')


from scipy.stats import ks_2samp
import pandas as pd

ks_results = []

# 对每个特征执行 KS 检验
for column in X_train.columns:
    ks_stat, p_value = ks_2samp(X_train[column], X_outTest[column])
    ks_results.append({'Feature': column, 'KS Statistic': ks_stat, 'P-Value': p_value})
    print(f"特征: {column}, KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")

# 转换为 DataFrame
ks_results_df = pd.DataFrame(ks_results)

# Bonferroni correction
alpha = 0.05
m = len(ks_results_df)
alpha_adjusted = alpha / m

ks_results_df['Bonferroni_Significant'] = ks_results_df['P-Value'] > alpha_adjusted
significant_features = ks_results_df[ks_results_df['Bonferroni_Significant']]['Feature'].tolist()

# Calculate Pearson correlation
correlations = X_All.corrwith(y_All)

top_50_features = correlations.abs().nlargest(16).index
top_50_features_list = list(top_50_features)

if 'Admission time' in top_50_features_list:
    top_50_features_list.remove('Admission time')

top_50_features = pd.Index(top_50_features_list)
X_top_50 = X_All[top_50_features]
selected_features = X_top_50.columns

print(f"{datetime.datetime.now()}-- Using LassoCV for cross-validation")
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_top_50, y_All)

best_alpha = lasso_cv.alpha_
print(f"{datetime.datetime.now()}-- Best alpha value: {best_alpha}")

lasso = Lasso(alpha=best_alpha)
lasso.fit(X_top_50, y_All)

model = SelectFromModel(lasso, prefit=True)
selected_features = X_top_50.columns[model.get_support()]

print("Selected features:", selected_features)

# Transform data with selected features
X = X[selected_features]
X_train = X_train[selected_features]
X_insidetest = X_insidetest[selected_features]
X_outTest = X_outTest[selected_features]


# ============================================================================
# STEP 5: STANDARDIZATION AND OUTLIER REMOVAL
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 5: STANDARDIZATION AND OUTLIER REMOVAL')
print(f'{"="*80}\n')
# 创建标准化器
scaler = StandardScaler()
X_train_original = X_train.copy()
# 对训练数据进行标准化
X_train = scaler.fit_transform(X_train)

# 使用相同的标准化器对测试数据进行标准化
X_insidetest=scaler.transform(X_insidetest)
X_outTest =scaler.transform(X_outTest)

# 假设 X 是你的特征数据
X_train = np.array(X_train)

# 计算Z-score
z_scores = np.abs(stats.zscore(X_train))

# 定义一个阈值，通常我们选择3，这意味着所有Z-score大于3的点都被认为是异常值
threshold = 3

# 获取异常值的位置
outliers = np.where(z_scores > threshold)

# 处理异常值，这里我们选择将它们替换为阈值
X_train[outliers] = threshold


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设 X_train 和 X_test 是训练集和测试集的特征矩阵
# 为训练集和测试集添加标签
X_combined = np.vstack((X_train, X_outTest))
y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_outTest))))  # 0 表示训练集，1 表示测试集

# 使用 PCA 将数据降到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)
# 分离训练集和测试集
X_train_pca = X_pca[y_combined == 0]
X_test_pca = X_pca[y_combined == 1]
# 计算每个类别的质心
centroids = {
    label: np.mean(X_pca[y_combined == label], axis=0)
    for label in np.unique(y_combined)
}

# 计算每个点到其所属类别质心的距离
distances = np.array([
    np.linalg.norm(X_pca[i] - centroids[y_combined[i]])
    for i in range(len(X_pca))
])

# 设置距离阈值（如 95% 分位数）
threshold = np.percentile(distances, 95)

# 标记异常点
outliers = distances > threshold

# 去掉异常点对应的 X_outTest 数据
X_outTest = X_outTest[~outliers[len(X_train):]]
y_outTest = y_outTest[~outliers[len(X_train):]]
outlier_indices = np.where(outliers)[0]
outlier_points = X_pca[outlier_indices]
# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label='Train', alpha=0.5, c='blue')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label='Test', alpha=0.5, c='orange')

plt.scatter(outlier_points[:, 0], outlier_points[:, 1], label='Outliers', alpha=0.8, c='red', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Outliers')
plt.legend()
now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d_%H%M%S')
plt.savefig(f'./DIC/Data/output/TIC/pca_visualization_{now_str}.png', dpi=300, bbox_inches='tight')  # dpi=300 确保高分辨率
print(f"图像已保存为 pca_visualization_{now_str}.png")

# 再来一遍
scaler = StandardScaler()
X_train_original = X_train.copy()
# 对训练数据进行标准化
X_train = scaler.fit_transform(X_train)

# 使用相同的标准化器对测试数据进行标准化
X_insidetest=scaler.transform(X_insidetest)
X_outTest =scaler.transform(X_outTest)

# 假设 X 是你的特征数据
X_train = np.array(X_train)

# 计算Z-score
z_scores = np.abs(stats.zscore(X_train))

# 定义一个阈值，通常我们选择3，这意味着所有Z-score大于3的点都被认为是异常值
threshold = 3

# 获取异常值的位置
outliers = np.where(z_scores > threshold)

# 处理异常值，这里我们选择将它们替换为阈值
X_train[outliers] = threshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设 X_train 和 X_test 是训练集和测试集的特征矩阵
# 为训练集和测试集添加标签
X_combined = np.vstack((X_train, X_outTest))
y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_outTest))))  # 0 表示训练集，1 表示测试集

# 使用 PCA 将数据降到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)
# 分离训练集和测试集
X_train_pca = X_pca[y_combined == 0]
X_test_pca = X_pca[y_combined == 1]
# 计算每个类别的质心
centroids = {
    label: np.mean(X_pca[y_combined == label], axis=0)
    for label in np.unique(y_combined)
}

# 计算每个点到其所属类别质心的距离
distances = np.array([
    np.linalg.norm(X_pca[i] - centroids[y_combined[i]])
    for i in range(len(X_pca))
])

# 设置距离阈值（如 95% 分位数）
threshold = np.percentile(distances, 95)

# 标记异常点
outliers = distances > threshold

X_outTest = X_outTest[~outliers[len(X_train):]]
y_outTest = y_outTest[~outliers[len(X_train):]]
outlier_indices = np.where(outliers)[0]
outlier_points = X_pca[outlier_indices]
# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label='Train', alpha=0.5, c='blue')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label='Test', alpha=0.5, c='orange')

plt.scatter(outlier_points[:, 0], outlier_points[:, 1], label='Outliers', alpha=0.8, c='red', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Outliers')
plt.legend()
plt.savefig(f'./DIC/Data/output/TIC/pca_visualization1_{now_str}.png', dpi=300, bbox_inches='tight')  # dpi=300 确保高分辨率
print(f"图像已保存为 pca_visualization1_{now_str}.png")


# ============================================================================
# STEP 6: DEFINE MACHINE LEARNING MODELS (WITH REGULARIZATION)
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 6: DEFINE MODELS')
print(f'{"="*80}\n')
models = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        min_impurity_decrease=0.01,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=20,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.8,
        random_state=42
    )
}


# ============================================================================
# STEP 7: HYPERPARAMETER OPTIMIZATION
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 7: HYPERPARAMETER OPTIMIZATION')
print(f'{"="*80}\n')
from sklearn.model_selection import GridSearchCV

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    },
    "Random Forest": {
        'n_estimators': [30, 50, 80],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [20, 30, 40],
        'min_samples_leaf': [10, 15, 20],
        'max_features': ['sqrt', 'log2']
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Neural Networks": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

best_models = {}
for name, model in models.items():
    print(f"Optimizing {name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
models = best_models


# ============================================================================
# STEP 8: MODEL EVALUATION AND RESULTS
# ============================================================================
print(f'\n{"="*80}')
print(f'STEP 8: MODEL EVALUATION')
print(f'{"="*80}\n')
feature_importances = pd.DataFrame(index=selected_features)

# 创建一个空的DataFrame来保存结果
train_95CI = pd.DataFrame()
test_95CI = pd.DataFrame()
insidetest_95CI = pd.DataFrame()
outside_95CI = pd.DataFrame()

# 初始化一个列表来保存ROC曲线数据
roc_data = []
roc_data1 = []
roc_data2 = []
roc_data3 = []
# 计算每个模型的 MCC 曲线
outMccResults = []
insideMccResults = []
trainMccResults = []
# 定义评分参数
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0),
    'roc_auc': 'roc_auc',  # 直接使用字符串名称，cross_validate会自动处理
    'log_loss': 'neg_log_loss',  # 使用内置的评分名称
    'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
}

# 创建一个空的DataFrame来保存结果
results = pd.DataFrame()
# 创建一个空的DataFrame来保存结果
bootstrapResults = pd.DataFrame()
bootstrapResults1 = pd.DataFrame()
# 创建一个空列表来保存每个模型的Precision-Recall曲线的数据
pr_curves = []
pr_curves1 = []
pr_curves2 = []
pr_curves3 = []
# 创建一个空列表来保存每个模型的校准曲线的数据
calibration_data = []
calibration_data1 = []
calibration_data2 = []
calibration_data3 = []

for name, model in models.items():
    model.fit(X_train, y_train)

    print(f"{datetime.datetime.now()}-- {name}--start 95CI...")    
    
    train_95CIitem = calculate_metrics(model, X_train, y_train, name)
    train_95CI = pd.concat([train_95CI,train_95CIitem]) 
    insidetest_95CIitem=calculate_metrics(model, X_insidetest, y_insidetest, name)
    insidetest_95CI = pd.concat([insidetest_95CI,insidetest_95CIitem])  
    outside_95CIitem=calculate_metrics(model, X_outTest, y_outTest, name)
    outside_95CI = pd.concat([outside_95CI,outside_95CIitem])
    print(f"{datetime.datetime.now()}-- {name}--end 95CI...")    
    
    
    print(f"{datetime.datetime.now()}-- {name}--start feature importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        feature_importances[name] = model.coef_[0]

    print(f"{datetime.datetime.now()}-- {name}--end feature importance...")    
    print(f"{datetime.datetime.now()}-- {name}--start outTest...")
    predictions = model.predict(X_outTest)
    print(f"{datetime.datetime.now()}-- {name}--start cross_validate()...")
    try:
        scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, error_score='raise')  # 5折交叉验证
        # 计算平均得分并保存到DataFrame
        for key in scores:
            if key.startswith('test_'):
                for i, score in enumerate(scores[key]):
                    results.loc[f'{name}_{i}', f'{key}'] = score
        print(f"{datetime.datetime.now()}-- {name}--cross_validate() completed successfully")
    except Exception as e:
        print(f"{datetime.datetime.now()}-- {name}--cross_validate() ERROR: {str(e)}")
        # 如果cross_validate失败，尝试不使用需要概率的评分指标
        try:
            scoring_without_proba = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='weighted'),
                'recall': make_scorer(recall_score, average='weighted'),
                'f1': make_scorer(f1_score, average='weighted'),
                'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
            }
            scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring_without_proba, error_score='raise')
            for key in scores:
                if key.startswith('test_'):
                    for i, score in enumerate(scores[key]):
                        results.loc[f'{name}_{i}', f'{key}'] = score
            print(f"{datetime.datetime.now()}-- {name}--cross_validate() completed without proba metrics")
        except Exception as e2:
            print(f"{datetime.datetime.now()}-- {name}--cross_validate() FAILED completely: {str(e2)}")
    
    print(f"{datetime.datetime.now()}-- {name}--start bootstrap_scores()...")            
    bootstrap_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for i in range(500):  # 进行100次Bootstrap
        # 生成Bootstrap样本
        X_resample, y_resample = resample(X_train, y_train)
        # 训练模型并计算得分
        model.fit(X_resample, y_resample)
        y_pred = model.predict(X_train)
        bootstrap_scores['accuracy'].append(accuracy_score(y_train, y_pred))
        bootstrap_scores['precision'].append(precision_score(y_train, y_pred, average='weighted'))
        bootstrap_scores['recall'].append(recall_score(y_train, y_pred, average='weighted'))
        bootstrap_scores['f1'].append(f1_score(y_train, y_pred, average='weighted'))
    # 计算平均得分并保存到DataFrame
    for metric in bootstrap_scores.keys():
        mean = np.mean(bootstrap_scores[metric])
        lower = np.percentile(bootstrap_scores[metric], 2.5)  # 计算2.5百分位数，即置信区间的下限
        
        upper = np.percentile(bootstrap_scores[metric], 97.5)  # 计算97.5百分位数，即置信区间的上限
        bootstrapResults.loc[name, f'{metric} Mean'] = mean
        bootstrapResults.loc[name, f'{metric} Lower 95% CI'] = lower
        bootstrapResults.loc[name, f'{metric} Upper 95% CI'] = upper
    print(f"{datetime.datetime.now()}-- {name}--end bootstrap_scores()...")       
    
    predictions = model.predict(X_outTest)
    predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_outTest, predictions)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    
    print(f"{datetime.datetime.now()}-- {name}--start cross_validate()...")
    y_score = model.predict_proba(X_outTest)[:, 1]
    
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_outTest, y_outTest)
    outMccResults.append((thresholds, mcc_scores, name))

    print(f"{datetime.datetime.now()}-- {name}--start outTest Compute ROC curve and ROC area...")     
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_outTest, y_score)
    roc_auc = auc(fpr, tpr)
    # 假设 y_outTest 和 y_score 是你要保存的数据
    # 首先，我们将它们转换为DataFrame
    ddf = pd.DataFrame({
        'y_outTest': y_outTest,
        'y_score': y_score
    })

    # 然后，我们可以使用 to_csv 方法将 DataFrame 保存为 CSV 文件
    ddf.to_csv('y_outTest-score.csv', index=False)
    # 保存ROC曲线数据
    roc_data.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线
    # 计算 Precision-Recall 曲线
    precision, recall, thresholds = precision_recall_curve(y_outTest, y_score)

    # 计算 F1 分数
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # 找到最佳阈值
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"{datetime.datetime.now()}-- {name}--最佳阈值: {best_threshold}")
    threshold = best_threshold  # 设置阈值为0.5
    predictions = (y_score >= threshold).astype(int)  # 将预测概率值大于阈值的设置为1，小于等于阈值的设置为0
    
    # 计算最终的精确率、召回率和 F1 分数
    final_precision = precision[np.argmax(f1_scores)]
    final_recall = recall[np.argmax(f1_scores)]
    final_f1 = f1_scores[np.argmax(f1_scores)]
    
    print(f"{datetime.datetime.now()}-- {name}--最终精确率: {final_precision}")
    print(f"{datetime.datetime.now()}-- {name}--最终召回率: {final_recall}")
    print(f"{datetime.datetime.now()}-- {name}--最终 F1 分数: {final_f1}")

    # 计算PR AUC作为图例显示（与CSV中的指标一致）
    pr_auc_score = auc(recall, precision)
    brier_score = brier_score_loss(y_outTest, y_score)
    # 将Precision-Recall曲线的数据添加到列表中（使用PR AUC而不是单点precision）
    pr_curves.append((precision, recall, name, pr_auc_score))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_outTest, y_score, n_bins=5)
    calibration_data.append((fraction_of_positives, mean_predicted_value, name,brier_score))

    ####################### insidetest
    print(f"{datetime.datetime.now()}-- {name}--start insidetest...") 
    predictions = model.predict(X_insidetest)
    
    accuracy = accuracy_score(y_insidetest, predictions)
    print(f'{name} insidetestData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_insidetest)[:, 1]
    
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_insidetest, y_insidetest)
    insideMccResults.append((thresholds, mcc_scores, name))

    # Compute ROC curve and ROC area
    print(f"{datetime.datetime.now()}-- {name}--start insidetest Compute ROC curve and ROC area...") 
    fpr, tpr, _ = roc_curve(y_insidetest, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"{datetime.datetime.now()}-- {name}--start 保存ROC曲线数据...")    
    # 保存ROC曲线数据
    roc_data1.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线（使用PR AUC）
    precision, recall, _ = precision_recall_curve(y_insidetest, y_score)
    pr_auc_score = auc(recall, precision)
    brier_score = brier_score_loss(y_insidetest, y_score)
    # 将Precision-Recall曲线的数据添加到列表中（使用PR AUC）
    pr_curves1.append((precision, recall, name, pr_auc_score))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_insidetest, y_score, n_bins=5)
    calibration_data1.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    #################### train
    print(f"{datetime.datetime.now()}-- {name}--start train...") 
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    print(f'{name} trainData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_train)[:, 1]
        
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_train, y_train)
    trainMccResults.append((thresholds, mcc_scores, name))

    # Compute ROC curve and ROC area
    print(f"{datetime.datetime.now()}-- {name}--start train Compute ROC curve and ROC area...") 
    fpr, tpr, _ = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"{datetime.datetime.now()}-- {name}--start 保存ROC曲线数据...")  
    # 保存ROC曲线数据
    roc_data3.append((fpr, tpr, roc_auc, name))
    # 计算模型的Precision-Recall曲线（使用PR AUC）
    precision, recall, _ = precision_recall_curve(y_train, y_score)
    pr_auc_score = auc(recall, precision)
    brier_score = brier_score_loss(y_train, y_score)
    # 将Precision-Recall曲线的数据添加到列表中（使用PR AUC）
    pr_curves3.append((precision, recall, name, pr_auc_score))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, y_score, n_bins=5)
    calibration_data3.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    # 获取当前日期和时间
    now = datetime.datetime.now()

    # 将日期和时间格式化为字符串
    now_str = now.strftime('%Y%m%d_%H%M')    
    # 计算SHAP值
    # 使用shap.sample对数据进行采样
    samplesize = 10
    if name in ["Random Forest", "AdaBoost"]:
        samplesize = 10
    X_sample = shap.sample(X_train, nsamples=samplesize, random_state=42)

    # 根据模型类型选择解释器
    if name in ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    elif name == "Neural Networks":
        try:
            explainer = shap.DeepExplainer(model, X_train)
        except Exception:
            try:
                shap.KernelExplainer(model.predict, X_sample)
            except Exception:
                continue
    elif name in ["Logistic Regression"]:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample)


    shap_values = explainer.shap_values(X_sample)

    # 处理二分类问题：统一SHAP值格式
    print(f"{name}: shap_values type = {type(shap_values)}")
    if isinstance(shap_values, list):
        # 如果是列表格式 [class0, class1]，提取正类（class 1）
        print(f"{name}: shap_values is list with {len(shap_values)} elements")
        print(f"{name}: shap_values[0] shape = {shap_values[0].shape}, shap_values[1] shape = {shap_values[1].shape}")
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        # 如果是三维数组 (samples, features, classes)，提取正类（索引1）
        print(f"{name}: shap_values shape = {shap_values.shape}, extracting class 1")
        shap_values = shap_values[:, :, 1]
    else:
        print(f"{name}: shap_values shape = {shap_values.shape}")
    
    print(f"{name}: final shap_values shape = {shap_values.shape}")

    plt.figure()
    # 绘制SHAP值
    # 确保绘制summary plot
    shap.summary_plot(shap_values, X_sample, max_display=15, feature_names=selected_features, show=False)
    plt.savefig(f'./DIC/Data/output/TIC/shap/{name}-shap_plot-{now_str}.png',dpi=700)
    plt.close()
    plt.clf()

# 获取当前日期和时间
now = datetime.datetime.now()

# 将日期和时间格式化为字符串
now_str = now.strftime('%Y%m%d_%H%M%S')    
# 保存结果到CSV文件
results.to_csv(f'./DIC/Data/output/TIC/cross_validation_results_{now_str}.csv')
bootstrapResults.to_csv(f'./DIC/Data/output/TIC/bootstrap_results_{now_str}.csv')


train_95CI.to_csv(f'./DIC/Data/output/TIC/train_95CI_results_{now_str}.csv')
insidetest_95CI.to_csv(f'./DIC/Data/output/TIC/insidetest_95CI_results_{now_str}.csv')
test_95CI.to_csv(f'./DIC/Data/output/TIC/test_95CI_results_{now_str}.csv')
outside_95CI.to_csv(f'./DIC/Data/output/TIC/outside_95CI_results_{now_str}.csv')
   
# 过滤出重要性大于0.05的特征
threshold = 0.0005
important_features = feature_importances[feature_importances > threshold].dropna(how='all')

# 保存到CSV文件
important_features.to_csv(f'./DIC/Data/output/TIC/important_features_{now_str}.csv', encoding='UTF-8')    
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Outside Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/TIC/Outside roc_curves_{now_str}.png')
#plt.show()

# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data1:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Inside Test Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/TIC/Inside Test roc_curves_{now_str}.png')
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data3:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/TIC/Train Data roc_curves_{now_str}.png')
#plt.show()
# 创建一个新的图形
plt.figure()

# 对于每个模型的ROC曲线数据，绘制ROC曲线
for fpr, tpr, roc_auc, name in roc_data2:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/TIC/Test Data roc_curves_{now_str}.png')
#plt.show()
plot_calibration_curves(calibration_data, 'Calibration curves', f'./DIC/Data/output/TIC/outside_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data1, 'InsideTest Data Calibration curves', f'./DIC/Data/output/TIC/insidetest_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data3, 'Train Data Calibration curves', f'./DIC/Data/output/TIC/train_data_calibration_curves_{now_str}.png')

plot_precision_recall_curves(pr_curves, 'Precision-Recall curve', f'./DIC/Data/output/TIC/outside_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves1, 'Precision-Recall curve', f'./DIC/Data/output/TIC/insidetest_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves3, 'Precision-Recall curve', f'./DIC/Data/output/TIC/train_data_precision_recall_curve_{now_str}.png')

plot_mcc_curves(outMccResults, f'./DIC/Data/output/TIC/outside_data_mcc_curve_{now_str}.png')
plot_mcc_curves(insideMccResults, f'./DIC/Data/output/TIC/inside_data_mcc_curve_{now_str}.png')
plot_mcc_curves(trainMccResults, f'./DIC/Data/output/TIC/train_data_mcc_curve_{now_str}.png')
#endregion evaluate models


