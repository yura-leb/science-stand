"""
This program is an .ipybn notebook utility.

"""

# %%
'''Import libraries'''
from sklearn import tree, preprocessing, model_selection, ensemble, metrics, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import openpyxl
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random
import json
import itertools
import signal
import scipy
from displayresults import get_data_columns
from collections import defaultdict
# from pure_sklearn.map import convert_estimator

'''We are using this libraries, because they are observed in "SDTR: Soft Decision Tree Regressor for Tabular
Data" article - HAORAN LUO , FAN CHENG , (Member, IEEE), HENG YU , (Graduate Student Member, IEEE), AND YUQI YI , (Student Member, IEEE)'''
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from deepforest import CascadeForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import time


# %%
'''Define pathes and general settings'''

'''Xlsx file to parse data from'''
xlsx_df_source = "cc_perf_tests/RENO_CUBIC_BBR2_mean_deviation_ML_v4.1.xlsx"
'''Path, where to save ml cross val results'''
cc_dir = "cc_ml/1task_november_start"
'''Filenames with ml results'''
ml_result_names = ['ml4y2_mean_experiment_Tue_Dec_6_17.44.51_2022']
sns.set_theme(style="whitegrid")

# %%
'''Import data and put in pd.DataFrame format'''

def importdf2(path):
    '''Import data from an xlsx with all algos in it'''
    wb = openpyxl.load_workbook(path)
    ws = wb["Sheet1"]
    columns = next(ws.values)[0:]
    df = pd.DataFrame(ws.values, columns=columns)
    df = df.iloc[1:, 1:]
    df = df.drop(['Channel Jitter (ms)'], axis=1)
    return df

def importdf():
    '''Import data from xlsxes with these specific three algos'''
    algos = ['BBR2', 'CUBIC', 'RENO']
    frames = [0]*3
    for idx, algo in enumerate(algos):
        wb = openpyxl.load_workbook(f"cc_perf_tests_fix/{algo}_ML_v3.1.xlsx")
        ws = wb["Sheet1"]
        columns = next(ws.values)[0:]
        frames[idx] = pd.DataFrame(ws.values, columns=columns)
        frames[idx] = frames[idx].iloc[1:, 1:]
    df = pd.concat(frames)
    df = df.reset_index()
    return df

# df = importdf2(xlsx_df_source)
df = importdf2(xlsx_df_source)
# df_testonly = importdf2(xlsx_df_source)

'''If somebody asks...'''
df['Channel Loss (%)'].unique()

# %%
def understand_distribution(data):
    '''Source: https://gist.github.com/mungoliabhishek/df69fc0f269eaefe59275fd4bd4cddd7'''
    dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta',
                  'invgauss', 'uniform', 'gamma', 'expon',   
                  'lognorm', 'pearson3', 'triang', 'laplace']
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(data, percentile_bins)
    observed_frequency, bins = (np.histogram(data, bins = percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(data)
        print("{}\t{}".format(dist, param))

        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * data.shape[0]
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_expected_frequency)
        chi_square_statistics.append(ss)

    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)

    print('\nDistributions listed by Betterment of fit:')
    print('............................................')
    print(results)

def plot_distribution(data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle("Distribution overview")
    ls = np.linspace(data.min(), data.max(), 101)
    # Histogram Plot of Observed Data
    axes[0, 0].hist(data, bins=50)
    axes[0, 0].set_title("Mean speed samples quantity histogram")
    axes[0, 0].set_xlabel('Mean speed position relatively to minimum and maximum speed')
    axes[0, 0].set_ylabel('Experiments number')
    # axes[0, 1].plot(ls, scipy.stats.weibull_max.pdf(ls, 4.010884654437827, 0.9497579518557138, 0.48572305476667677))
    axes[1, 0].plot(ls, scipy.stats.laplace.pdf(ls, 0.5035656774787212, 0.09874069646583596))
    axes[1, 0].set_title("Laplace distribution probability density function")
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('F(x)')
    axes[1, 1].plot(ls, scipy.stats.norm.pdf(ls, 0.5093598981003012, 0.1235147677357962))
    axes[1, 1].set_title("Normal distribution probability density function")
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('F(x)')
    fig.tight_layout()

'''Define distribution of target variables'''
df_distr = preprocessing.MinMaxScaler().fit_transform(df[['Sender Speed (Kbit/s)', 'MinimalSpeed', 'MaximalSpeed']].transpose())
df_distr = pd.DataFrame({'Mean Speed': df_distr[0]})
# plt.plot([0.2, 0.3], [100, 200])
# df_distr.hist(bins = 30)
understand_distribution(df_distr['Mean Speed'])
plot_distribution(df_distr['Mean Speed'])

# %%
'''Preprocessing'''
def preprocess(df, encoder='one_hot_encoder'):
    '''Convert data types'''
    col = get_data_columns(speed=False, formulas=False)

    '''Remove columns, that we don't need for learning'''
    col.remove('Channel Jitter (ms)')

    '''Add new statistical columns'''
    col.append('MinimalSpeed')
    col.append('MaximalSpeed')
    col.append('Deviation')
    col.append('Speed deviation percent (%)')

    '''Float64'''
    d = dict()
    for i in col:
        d[i] = 'float64'

    '''Categorial'''
    d['Congestion Controller'] = 'str'

    df = df.astype(d)

    if (encoder == 'one_hot_encoder'):
        '''One hot encoder (This way of encoding performs well)'''
        '''(As some algorithms, that we want to test do not support categorial features, we want to use encoders)'''
        algo_df = pd.DataFrame(df, columns=['Congestion Controller'])
        dum_df = pd.get_dummies(algo_df, columns=["Congestion Controller"], prefix=["CC_is"] )
        algo_df = algo_df.join(dum_df)
        df = pd.concat([dum_df, df.iloc[:, 1:]], axis=1, join='inner')

        # scaler = preprocessing.MinMaxScaler()
        # df[df.columns] = scaler.fit_transform(df[df.columns])

        X = pd.DataFrame(df, columns = ["CC_is_BBR2", "CC_is_CUBIC", "CC_is_RENO", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)"])
    elif (encoder == 'label_encoder'):
        df["Congestion Controller"] = df["Congestion Controller"].apply(lambda x: list(df["Congestion Controller"].unique()).index(x) + 1)
        X = pd.DataFrame(df, columns = ["Congestion Controller", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)"])
        
        pd.to_numeric(df['Congestion Controller'])
    else:
        pass
    y1 = df["Sender Speed (Kbit/s)"]
    y2 = df["MinimalSpeed"]
    y3 = df["MaximalSpeed"]
    return X, y1, y2, y3

X_df, y1_df, y2_df, y3_df = preprocess(df, 'label_encoder')
# X_testonly, y1_testonly, y2_testonly = preprocess(df_testonly)

# %%
'''Split train & test'''
random_state_ = 42

X_cv, X_test, y1_cv, y1_test = model_selection.train_test_split(X_df, y1_df, test_size=0.05, random_state=random_state_, shuffle=True)
X_cv, X_test, y2_cv, y2_test = model_selection.train_test_split(X_df, y2_df, test_size=0.05, random_state=random_state_, shuffle=True)
X_cv, X_test, y3_cv, y3_test = model_selection.train_test_split(X_df, y3_df, test_size=0.05, random_state=random_state_, shuffle=True)

X = X_cv
y1 = y1_cv
y2 = y2_cv
y3 = y3_cv

print(X_cv.shape, X_test.shape, y1_cv.shape, y1_test.shape)


# %%
'''Clear results'''
res_list = list()

# %%

'''Print cross val results and save them in 'cc_dir' '''
def conclude_and_save(res_list, total_time, finalists_numb=10):
    print(f'\nTotal {total_time} taken.')
    res_list = sorted(res_list, key=lambda x: -x['test_NMAE'])
    print(f"\nTop {finalists_numb} models: ", res_list[:10])
    fname = f'{cc_dir}/ml5y_mean_experiment_'+'.'.join('_'.join(time.asctime().split()).split(':'))
    with open(fname, 'w') as f:
        json.dump(res_list, f, indent = 6)
    print("\nName: ", fname)

'''Custom regression metrics'''
def negative_sqrt_mean_squared_error(y_true, y_pred):
    '''This is to apply sqrt and use "greater_is_better=True"'''
    res = metrics.mean_squared_error(y_true, y_pred)
    return - np.sqrt(res)

def negative_mean_absolute_error(y_true, y_pred):
    '''This is to use "greater_is_better=True"'''
    res = metrics.mean_absolute_error(y_true, y_pred)
    return - res

def negative_error_percent(y_true, y_pred):
    '''Custom metric'''
    res = -((y_true - y_pred) / y_true * 100).abs().mean()
    return res

'''Cross val'''
def cross_val(X, y, paramGrid, modelF, libName, expConstraint=10000, timeConstraint=60):
    total_time = time.time()
    i = 1
    '''Constraint on holding data, this would be enough'''
    max_paramGrid_size = 1000000
    paramGrid_size = min(math.prod([len(paramGrid[key]) for key in paramGrid.keys()]), max_paramGrid_size)

    iteratedGrid_size = 0
    iteratedGrid_set = []

    class TooLongExperiment(Exception):
        '''Raise if algorithm train is way too long'''
        pass

    def signal_handler(signum, frame):
        raise TooLongExperiment("\nPrevious experiment timed out!")

    '''Put regression metrics into a tuple'''
    metrics_tuple = {
        'NSMSE': metrics.make_scorer(negative_sqrt_mean_squared_error, greater_is_better=True),
        'NMAE': metrics.make_scorer(negative_mean_absolute_error, greater_is_better=True),
        'R2': metrics.make_scorer(metrics.r2_score, greater_is_better=True),
        'NEP': metrics.make_scorer(negative_error_percent, greater_is_better=True),
    }

    '''2 is the unreachable best result for each metric'''
    metrics_val = {
        'test_NSMSE': 2,
        'test_NMAE': 2,
        'test_R2': 2,
        'test_NEP': 2
    }

    '''For current cross val status output'''
    metrics_val_best = {
        'test_NSMSE': 2,
        'test_NMAE': 2,
        'test_R2': 2,
        'test_NEP': 2
    }
    while ((iteratedGrid_size != expConstraint) and (iteratedGrid_size != paramGrid_size // 2)):
        '''We don't want to go through all of the iterations => '... // 2' '''
        ijk = dict()
        for key in paramGrid.keys():
            ijk[key] = random.choice(paramGrid[key])

        if ijk in iteratedGrid_set:
            '''If this experiment is already done, we don't need to repeat'''
            continue

        iteratedGrid_size += 1
        iteratedGrid_set.append(ijk)

        initialtime = time.time()
        model = modelF(ijk)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(timeConstraint)
        try:
            cv_results = model_selection.cross_validate(model, X, y, cv=model_selection.ShuffleSplit(n_splits=5, test_size=0.25, random_state=0), scoring=metrics_tuple, n_jobs=-1)
        except TooLongExperiment as E:
            print(E)
            signal.alarm(0)
            continue
        except KeyboardInterrupt as E:
            print('\nKeyboardInterrupt => Returning...')
            signal.alarm(0)
            break
        signal.alarm(0)
        for key in metrics_val.keys():
            metrics_val[key] = cv_results[key].mean()
        resulttime = time.time() - initialtime

        '''Experiment Results'''
        print(f'\nExperiment {i}/{min(paramGrid_size // 2, expConstraint)};   Time {resulttime};   Parameters: {ijk}')
        for key in metrics_val.keys():
            if (metrics_val_best[key] < metrics_val[key]) or (metrics_val_best[key] == 2):
                metrics_val_best[key] = metrics_val[key]
            print(f'Score {key[5:]}\t {metrics_val[key]} \t(Best {metrics_val_best[key]})')
        i += 1
        exp_dict = dict()
        exp_dict['library'] = libName
        for key in paramGrid.keys():
            exp_dict[key] = ijk[key]
        exp_dict['training_duration'] = resulttime
        for key in metrics_val.keys():
            exp_dict[key] = metrics_val[key]
        res_list.append(exp_dict)

    return res_list, time.time() - total_time


# %%
'''LightGBM'''
paramGrid = {
    'learning_rate': list(np.random.uniform(0.001, 0.4, 50)),
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'n_estimators': [int(i) for i in list(np.random.uniform(10, 2000, 50))],
    'num_leaves': [2, 3, 4, 5, 6, 7, 8, 9]
}

def LGBMRegressor_model(ijk):
    return TransformedTargetRegressor(regressor=LGBMRegressor(**ijk),
                                      transformer = preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y2, paramGrid, LGBMRegressor_model, 'LightGBM_y3', 500, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''CatBoost'''
paramGrid = {
    'num_trees': [int(i) for i in list(np.random.uniform(100, 1000, 10)) + list(np.random.uniform(1000, 3000, 20))],
    'depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'l2_leaf_reg': [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}

def CatBoost_model(ijk):
    return CatBoostRegressor(**ijk, silent=True)

res_list, total_time = cross_val(X, y2, paramGrid, CatBoost_model, 'CatBoost', 500, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''XGBoost'''
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))
paramGrid = {
    'learning_rate': list(np.random.uniform(0.01, 0.1, 20)) + list(np.random.uniform(0.1, 1, 20)),
    'n_estimators': [int(i) for i in list(np.random.uniform(0, 1000, 5)) + list(np.random.uniform(1000, 3000, 5))],
    'max_depth': [int(i) for i in list(np.random.uniform(2, 14, 30))],
    'subsample': list(np.random.uniform(0.5, 1, 20)),
    'colsample_bytree': list(np.random.uniform(0.5, 1, 20)),
    'colsample_bylevel': list(np.random.uniform(0.5, 1, 20)),
    'min_child_weight': list(np.random.uniform(1, 7, 20)),
    'reg_alpha': list(np.random.uniform(0, 12, 20)) + [0],
    'reg_lambda': list(np.random.uniform(0, 12, 20)) + [0],
    'gamma': list(np.random.uniform(0.5, 1, 20)) + [0]
}

def XGBoost_model(ijk):
    return TransformedTargetRegressor(regressor=XGBRegressor(**ijk),
                                      transformer = preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y1, paramGrid, XGBoost_model, 'XGBoost', 200, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Sklearn Tree'''
paramGrid = {
    'n_estimators': [int(i) for i in list(np.random.uniform(1, 450, 20))],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10]
}

def Sklearn_model(ijk):
    return TransformedTargetRegressor(regressor=ensemble.RandomForestRegressor(**ijk),
                                      transformer=preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y1, paramGrid, Sklearn_model, 'Sklearn_y1', 400, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''
res_list = list()

# %%
'''Sklearn Lasso'''
paramGrid = {
    'alpha': [0.0, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'selection': ['cyclic', 'random']
}

def Sklearn_Lasso_model(ijk):
    return Lasso(**ijk)

res_list, total_time = cross_val(X, y1, paramGrid, Sklearn_Lasso_model, 'Sklearn Lasso')

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Sklearn Ridge'''
paramGrid = {
    'alpha': [0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag']
}

def Sklearn_Ridge_model(ijk):
    return Ridge(**ijk)

res_list, total_time = cross_val(X, y1, paramGrid, Sklearn_Ridge_model, 'Sklearn Ridge')

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Sklearn SVR'''
paramGrid = {
    'kernel': ['poly'], # + ['rbf', 'sigmoid', 'linear'],
    'degree': list(np.random.uniform(1, 15, 30)),
    'epsilon': [0.1] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'C': [1] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'gamma': 10*['auto'] + 5*['scale'] + list(np.random.uniform(0.1, 1, 5)),
    'coef0': 20*[0.0] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'shrinking': 8*[True] + [False]
}

def Sklearn_SVR(ijk):
    return make_pipeline(StandardScaler(), svm.SVR(**ijk))

res_list, total_time = cross_val(X, y1, paramGrid, Sklearn_SVR, 'Sklearn SVR', timeConstraint=20)

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Invsee Cross Validation progress, build graphs to better understand parameter samples quality.
Put the files with ML cross val results in 'ml_result_names' list
The parameters chosen and the metrics must be same for all the experiments!'''

def plot_cross_val_rating(cc_dir, ml_result_names):
    results = []
    for filename in ml_result_names:
        with open(cc_dir + '/' + filename, 'r') as f:
            results += json.load(f)

    '''Pick a certain amount of best models, to place on a graph'''
    best_models_amount = 200
    results = results[:best_models_amount]
    n = 2
    m = 6
    fig, axes = plt.subplots(n, m, figsize=(30, 10))
    fig.suptitle('Dependencies between chosen parameters and accuracy score.')

    k = 0
    for i in range(n):
        for j in range(m):
            k += 1
            if ((k >= len(results[0].keys())) or (list(results[0].keys())[k] == "gamma")):
                axes[i, j].set_title('None')
                continue
            axes[i, j].set_title('Parameter ' + list(results[0].keys())[k])
            x_ = [p[list(results[0].keys())[k]] for p in results]
            y_ = [p["test_R2"] for p in results]
            sns.scatterplot(ax=axes[i, j], data=pd.DataFrame({"Parameter value": x_, "Accuracy": y_}), x="Parameter value", y="Accuracy")

plot_cross_val_rating(cc_dir, ml_result_names)

# %%
'''Correlation betweeen chosen metrics'''

def plot_metrics_correlation(cc_dir, ml_result_names):
    results = []
    for filename in ml_result_names:
        with open(cc_dir + '/' + filename, 'r') as f:
            results += json.load(f)

    best_models_amount = 80
    results = results[:best_models_amount]

    fig, axes = plt.subplots(4, 4, figsize=(28, 28))
    fig.suptitle('Dependencies between chosen parameters and accuracy score.')

    metrics_names = ['test_NSMSE', 'test_NMAE', 'test_R2', 'test_NEP']
    for i in range(4):
        for j in range(4):
            axes[i, j].set_title(f'Dependency {metrics_names[i]} by {metrics_names[j]}')
            x_ = [p[metrics_names[j]] for p in results]
            y_ = [p[metrics_names[i]] for p in results]
            sns.lineplot(ax=axes[i, j], data=pd.DataFrame({f"Metric {metrics_names[i]}": x_, f"Metric {metrics_names[j]}": y_}), x=f"Metric {metrics_names[i]}", y=f"Metric {metrics_names[j]}", ci=100)

plot_metrics_correlation(cc_dir, ml_result_names)

# %%
'''Functions to experiment with models after crossvalidation finished.'''

'''Train & Test'''
def manual_train_test(model, X_train, y_train, X_test, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = []
    for metric in metrics:
        test_score.append(metric(y_test, y_pred))
    return y_pred, test_score

'''For next function'''
target_type_str = [
    'Sender Speed (Kbit/s)',
    'MinimalSpeed',
    'MaximalSpeed'
]
alpha = 1

'''Examine if results differ much'''
def manual_invsee_results(target_str, y_test, y_pred, metric_results):
    results = y_test.to_frame().assign(Predicted=y_pred)
    # sorted_results = y_test.to_frame().assign(Predicted=y_pred).sort_values(by=target_str)
    results['Abs Error (%)'] = (results[target_str] - results['Predicted']) / results[target_str] * 100
    results['Squared Error'] = (results[target_str] - results['Predicted']) ** 2 / results[target_str] ** 2 * 100
    results.rename({'Predicted': 'Predicted value (Kbit/s)'}, inplace=True)
    print("Metrics:") 
    for key, val in metric_results.items():
        print(key, val)
    print("\n", results.tail(10))
    return results

'''Try to ensemble two plots, and see how they combine'''
def model_2_ensemble_plot(y_pred1, y_pred2, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    alpha_min = 0
    alpha_max = 1
    ls = np.linspace(alpha_min, alpha_max, 101)
    quals = []
    for metric in metrics:
        quals.append([metric(y_test, y_pred1 * i + (1 - i) * y_pred2) for i in ls])
    plt.figure(figsize=(8, 8 * len(metrics)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
    fig.suptitle('Changing alpha parameter to see, how models impact in overall result.')
    for idx, (qual, metric) in enumerate(zip(quals, metrics)):
        axes[idx].set_title('Metric: ' + str(metric.__name__))
        axes[idx].plot(ls, qual, lw=2)
        axes[idx].set_xlabel('Alpha')
        axes[idx].set_ylabel('Quality')
        axes[idx].set(xlim=[alpha_min, alpha_max])
    for qual in quals:
        print('Max dual metric: ', max(qual), '\nAlpha: ', ls[qual.index(max(qual))])

'''Try to ensemble two plots, and see how they combine'''
def model_3_ensemble_plot(y_pred1, y_pred2, y_pred3, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    alpha_min = 0
    alpha_max = 1
    size = 20
    quals = []
    max_quals = []
    first_quals = []
    alphas = [0 for i in range(len(metrics))]
    betas = [0 for i in range(len(metrics))]
    for metric in metrics:
        max_quals.append(metric(y_test, y_pred1))
        first_quals.append(metric(y_test, y_pred1))
    ls = []
    for idx, metric in enumerate(metrics):
        sup_quals = []
        for k, j in enumerate(np.linspace(alpha_min, alpha_max, size + 1)):
            sub_quals = []
            sub_ls = []
            for i in np.linspace(alpha_min, alpha_max - j, size + 1 - k):
                qual = metric(y_test, y_pred1 * j + y_pred2 * i + y_pred3 * (1 - i - j))
                if (max_quals[idx] < qual):
                    max_quals[idx] = qual
                    alphas[idx] = j
                    betas[idx] = i
                sub_quals.append(qual)
                sub_ls.append(i)
            for i in np.linspace(alpha_max - j, alpha_max, k):
                sub_quals.append(-200)
            sup_quals.append(sub_quals)
            ls.append(sub_ls)
        quals.append(sup_quals)
    # print(quals[0])
    plt.figure(figsize=(8, 12 * len(metrics)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
    fig.suptitle('Changing alpha, beta parameter to see, how models impact in overall result.')
    for idx, qual in enumerate(quals):
        # print(ls, qual)
        sns.heatmap(qual, ax=axes[idx], vmin=min(qual[0][0]/(100/99), qual[0][0]*(7/6)))
        axes[idx].set_title('Metric: ' + str(metrics[idx].__name__))
        axes[idx].set_xlabel('Beta')
        axes[idx].set_ylabel('Alpha')
        # axes[idx].set_xticks(np.linspace(alpha_min, alpha_max, size + 1, 10))
        # axes[idx].set_yticks(np.linspace(alpha_min, alpha_max, size + 1, 10))
        # axes[idx].set(xlim=[alpha_min, alpha_max])
    print('Max dual metrics: ', max_quals, '\nAlpha: ', alphas, '\nBeta: ', betas)

'''Do we want to handle more experiments for new input samples?'''
def train_size_plot(model, X, y, metric):
    iterations = 20
    iterations_for_each = 25
    ls = np.linspace(0.3, 0.93, iterations)
    sizes = []
    quals = []
    for idx, i in enumerate(ls):
        quals_tmp = []
        for j in range(iterations_for_each):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1-i, shuffle=True)
            _, cv_score = manual_train_test(model, X_train, y_train, X_test, y_test, metrics=[metric])
            quals_tmp.append(cv_score[0])
        quals.append(np.mean(quals_tmp))
        sizes.append(X_train.shape[0])
        print("train_size_plot: [", idx+1, " / ", iterations, "]")
    plt.figure(figsize=(10, 10))
    plt.plot(sizes, quals, lw=2)
    plt.xlabel('Train size')
    plt.ylabel('Quality')

'''Try using stacking - Current results are worse than models themselves'''
def stacking_attempt(model1, model2):
    '''This function uses globals'''
    estimators = [
        ('1', model1),
        ('2', model2)
    ]
    stacking = ensemble.StackingRegressor(
        estimators=estimators,
        final_estimator=ensemble.RandomForestRegressor(n_estimators=2000, max_depth=8)
    )

    y_pred0, cv_score0 = manual_train_test(stacking, X, y1, X_test, y1_test)
    results = manual_invsee_results(target_type_str[0], y1_test, y_pred0, cv_score0)

'''Show some metrics correllation'''
def plot_features_and_target_correlation(X_df, y1_df, y2_df, y3_df):
    plt.imshow(pd.concat([X_df, y1_df, y2_df, y3_df], axis=1).corr(method='kendall').apply(lambda x: np.abs(x)))
    pd.concat([X_df, y1_df, y2_df, y3_df], axis=1).corr(method='kendall')

'''Just a plot to invsee results.
Plot in the next function is customizable depending on the constraints set'''
def custom_test_train_plot(X, y1_test, y2_test, y3_test, y1_pred, y2_pred, y3_pred):
    '''Cusom constraints:'''
    X = X.loc[(X['Congestion Controller'] == 3)]
    X = X.loc[(X['Channel BW (Kbit/s)'] >= 50000) & (X['Channel BW (Kbit/s)'] >= 250000)]
    X = X.loc[(X['Channel RTT (ms)'] >= 50) & (X['Channel RTT (ms)'] >= 0)]
    X = X.loc[(X['Channel Loss (%)'] >= 1) & (X['Channel Loss (%)'] >= 0)]
    x_feature = 'Channel BW (Kbit/s)'

    '''Make a plot'''
    X = X.sort_values(by=x_feature, ascending=False)
    idx = list(X.index.values)
    plt.figure(figsize=(13, 10))
    X_axis = X[x_feature]

    # X_axis = str(X_axis[x_feature])

    plt.plot(X_axis, y1_test[idx], label='Speed (Kbit/s)', color='#85F1B8') # 85F1B8
    plt.plot(X_axis, pd.Series(y1_pred, index=y1_test.index)[idx], label='Predicted Mean Speed (Kbit/s)', color='#0DD26A')
    plt.plot(X_axis, y2_test[idx], label='Real Min Speed (Kbit/s)', color='#F4AE7B')
    plt.plot(X_axis, pd.Series(y2_pred, index=y2_test.index)[idx], label='Predicted Min Speed (Kbit/s)', color='#DE630B')
    plt.plot(X_axis, y3_test[idx], label='Real Max Speed (Kbit/s)', color='#7BB6F4')
    plt.plot(X_axis, pd.Series(y3_pred, index=y3_test.index)[idx], label='Predicted Max Speed (Kbit/s)', color='#146CCA')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(x_feature)
    plt.ylabel('Sender Speed (Kbit/s)')
    plt.legend()

'''Final ensemble'''
def net_speed_predict_ensemble(best_models, X_test, measuretime=False):
    channels = X_test #.head(20)
    predictions = dict()
    if measuretime:
        start_time = time.time()
    for key, model in best_models.items():
        predictions[key] = model.predict(channels)
    '''Emperical alpha/beta below:'''
    predictions[1] = predictions[11] * (0.85) + predictions[12] * (0.0) + predictions[13] * (0.15)
    predictions[2] = predictions[21] * (0.25) + predictions[22] * (0.5) + predictions[23] * (0.25)
    predictions[3] = predictions[31] * (0.5) + predictions[32] * (0.3) + predictions[33] * (0.2)
    model_3_ensemble_plot(predictions[1], predictions[2], predictions[3], y1_test)
    y_pred = predictions[1] * (0.85) + predictions[2] * (0.1) + predictions[3] * (0.05)
    if measuretime:
        print("Time taken for predict in seconds:", time.time() - start_time)
    return y_pred

'''Measure predict time. '''
def measure_time_best_models_no_ensemble(best_models, X_test):
    times = []
    for i in range(10):
        start_time = time.time()
        for key, model in best_models.items():
            _ = model.predict(X_test.head(20))
        times.append(time.time() - start_time)
    print(sum(times) / len(times))
    '''Let 0.3s. be a nice threshhold'''

# %%
'''Demonstration cell (best models based on cv with RMSE or R2SCORE metrics)'''
best_models = {}

'''For y1 task: -4.960250716495482'''
best_models[11] = XGBRegressor(max_depth=12, learning_rate=0.06, n_estimators=1650) #(max_depth=6, learning_rate=0.035, n_estimators=1184)
best_models[12] = TransformedTargetRegressor(regressor=ensemble.RandomForestRegressor(n_estimators=320, max_depth=10, max_features='auto', min_samples_leaf=1, min_samples_split=2),
                                         transformer=preprocessing.MinMaxScaler())
best_models[13] = CatBoostRegressor(num_trees=1068, depth=9, l2_leaf_reg=0)
# alpha1 = 0.35
# beta1 = 

'''For y2 task: -8.719683381937244'''
best_models[21] = XGBRegressor(max_depth=8, learning_rate=0.013, n_estimators=1100)
best_models[22] = TransformedTargetRegressor(regressor=ensemble.RandomForestRegressor(n_estimators=167, max_depth=13, max_features='auto', min_samples_leaf=2, min_samples_split=2),
                                         transformer=preprocessing.MinMaxScaler())
best_models[23] = CatBoostRegressor(num_trees=1086, depth=8, l2_leaf_reg=4)
# alpha2 =
# beta2 = 

'''For y3 task: -6.9597026281023044'''
best_models[31] = XGBRegressor(max_depth=7, learning_rate=0.063, n_estimators=2167)
best_models[32] = TransformedTargetRegressor(regressor=ensemble.RandomForestRegressor(n_estimators=320, max_depth=10, max_features='auto', min_samples_leaf=1, min_samples_split=2),
                                         transformer=preprocessing.MinMaxScaler())
best_models[33] = CatBoostRegressor(num_trees=817, depth=9, l2_leaf_reg=0)
# alpha3 =
# beta3 = 

# %%
'''Train all the best models'''
test_pred = {}
for key, model in best_models.items():
    test_pred[key], test_score = manual_train_test(model, X, eval("y" + str(key)[0]), X_test, eval("y" + str(key)[0] + "_test"))
    print("Best model ", key, " scores: ", test_score)

# %%
'''Get best alpha/betas'''
model_3_ensemble_plot(test_pred[11], test_pred[12], test_pred[13], y1_test)
model_3_ensemble_plot(test_pred[21], test_pred[22], test_pred[23], y2_test)
model_3_ensemble_plot(test_pred[31], test_pred[32], test_pred[33], y3_test)

# %%
'''Measure predict time. '''
# measure_time_best_models_no_ensemble(best_models, X_test)

# %%
'''Run ensemble'''
predictions = net_speed_predict_ensemble(best_models, X_test, measuretime=True)
manual_invsee_results(target_type_str[0], y1_test, predictions, {"Negative error percent:": negative_error_percent(predictions, y1_test), "R2 score:": metrics.r2_score(predictions, y1_test)})
# print(metrics.r2_score(predictions, y1_test), negative_error_percent(predictions, y1_test))


# %%
'''Do we want to handle more experiments for new input samples?'''
# train_size_plot(TransformedTargetRegressor(regressor=LGBMRegressor(),
#                                            transformer=preprocessing.MinMaxScaler()), 
#                                            X, y1, negative_error_percent) 

# %%
