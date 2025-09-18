import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats

player_df = pd.read_csv("fifa19.csv")

numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']

player_df = player_df[numcols+catcols]

traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()

traindf = pd.DataFrame(traindf,columns=features)

y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']

X.head()

len(X.columns)

feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30

def cor_selector(X, y, num_feats):
    cor_list = []
    for col in X.columns:
        cor = np.corrcoef(X[col], y)[0, 1]
        if np.isnan(cor):
            cor = 0
        cor_list.append(cor)
    cor_list = [abs(i) for i in cor_list]

    cor_support = [False] * len(cor_list)
    cor_feature = X.iloc[:, np.argsort(cor_list)[-num_feats:]].columns.tolist()
    for i in np.argsort(cor_list)[-num_feats:]:
        cor_support[i] = True

    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

cor_feature

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

def chi_squared_selector(X, y, num_feats):
    X_norm = X.copy().fillna(0)
    X_norm = MinMaxScaler().fit_transform(X_norm)

    k = min(num_feats, X_norm.shape[1])

    chi_selector = SelectKBest(score_func=chi2, k=k)
    chi_selector.fit(X_norm, y)

    chi_support = chi_selector.get_support()
    chi_feature = X.columns[chi_support].tolist()

    return chi_support, chi_feature

chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')

chi_feature

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def rfe_selector(X, y, num_feats):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000, solver="liblinear")
    rfe = RFE(estimator=model, n_features_to_select=min(num_feats, X.shape[1]))
    rfe.fit(X_scaled, y)

    rfe_support = rfe.get_support()
    rfe_feature = X.columns[rfe_support].tolist()

    return rfe_support, rfe_feature

rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')

rfe_feature

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def embedded_log_reg_selector(X, y, num_feats):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)
    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X, y)
    embedded_lr_support = selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')

embedded_lr_feature

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def embedded_rf_selector(X, y, num_feats):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X, y)
    embedded_rf_support = selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature


embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedder_rf_feature)), 'selected features')

embedder_rf_feature

from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

def embedded_lgbm_selector(X, y, num_feats):

    model = LGBMClassifier(n_estimators=200, random_state=42)

    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X, y)
    embedded_lgbm_support = selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')

embedded_lgbm_feature

pd.set_option('display.max_rows', None)

# Put all selection together
feature_selection_df = pd.DataFrame({
    'Feature': feature_name,
    'Pearson': cor_support,
    'Chi-2': chi_support,
    'RFE': rfe_support,
    'Logistics': embedded_lr_support,      
    'Random Forest': embedder_rf_support, 
    'LightGBM': embedded_lgbm_support      
})

# Count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df.drop(columns=['Feature']), axis=1)

# Sort by Total
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=[False, True])
feature_selection_df.reset_index(drop=True, inplace=True)

feature_selection_df

def preprocess_dataset(dataset_path):
    player_df = pd.read_csv(dataset_path)

    numcols = ['Overall', 'Crossing','Finishing','ShortPassing','Dribbling','LongPassing',
               'BallControl','Acceleration','SprintSpeed','Agility','Stamina','Volleys',
               'FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots',
               'Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']

    player_df = player_df[numcols+catcols]

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    traindf = traindf.dropna()
    features = traindf.columns
    traindf = pd.DataFrame(traindf, columns=features)

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    num_feats = 30
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods, num_feats=20):
    import pandas as pd
    import numpy as np

    # Use your preprocessing function to get data
    X, y, default_num_feats = preprocess_dataset(dataset_path)

    # If user didn’t specify num_feats, fall back to preprocess default
    if num_feats is None:
        num_feats = default_num_feats

    feature_name = X.columns
    results = {}

    # Run each method requested
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        results['Pearson'] = cor_support
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        results['Chi-2'] = chi_support
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        results['RFE'] = rfe_support
    if 'log-reg' in methods:
    
        lr_support, lr_feature = embedded_log_reg_selector(X, y, num_feats)
        results['Logistic'] = lr_support
    if 'rf' in methods:
        rf_support, rf_feature = embedded_rf_selector(X, y, num_feats)
        results['Random Forest'] = rf_support
    if 'lgbm' in methods:
        lgbm_support, lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        results['LightGBM'] = lgbm_support

    # Combine into DataFrame
    feature_selection_df = pd.DataFrame({'Feature': feature_name})
    for method, support in results.items():
        feature_selection_df[method] = support

    # Count how many times a feature was selected
    feature_selection_df['Total'] = np.sum(
        feature_selection_df.drop(columns=['Feature']), axis=1
    )
    feature_selection_df = feature_selection_df.sort_values(
        ['Total', 'Feature'], ascending=[False, True]
    ).reset_index(drop=True)

    # Best features are those selected at least once
    best_features = feature_selection_df[
        feature_selection_df['Total'] > 0
    ]['Feature'].tolist()

    return best_features, feature_selection_df

best_features, summary = autoFeatureSelector(
    dataset_path="fifa19.csv",
    methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'],
    num_feats=30
)

print("Best features selected:", best_features)
summary

