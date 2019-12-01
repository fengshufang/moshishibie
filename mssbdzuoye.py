#importing necessary models and libraries

#Math tools
from scipy import stats
from scipy.stats import skew, norm  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import scipy.stats as stats

#Visualizing tools
import seaborn as sns
import matplotlib.pyplot as plt

#preprocessing tools
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

#ML Algoirthm
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
import sklearn.linear_model as linear_model
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

train = pd.read_csv("train.csv")                #加载训练集
test = pd.read_csv("test.csv")                  #加载测试集
train_size = train.shape[0]                     #训练集的规模
submission = pd.read_csv("sample_submission.csv")             #加载测试集的结果
y_test = submission['SalePrice']
print(train.shape)
import warnings
warnings.filterwarnings(action="ignore")

numeric_cols = train.select_dtypes(exclude='object').columns        #提取训练集中特征为数字序列的向量
numeric_cols_length = len(numeric_cols)                             #计算训练集中特征为数字序列的向量的长度

corr = train.select_dtypes(include='number').corr()                 #计算训练集中特征为数字序列的方差

corWithSalePrice = train.corr().nlargest(10, 'SalePrice')['SalePrice'].index        #选出和Saleprice协方差最大的9个特征

y = np.log1p(train['SalePrice'])                                    #对训练集输出进行（ln(x+1))操作

#将过于大的特征量去除
def remove_overfit_features(df,weight):
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > weight:
            overfit.append(i)
    overfit = list(overfit)
    return overfit

overfitted_features = remove_overfit_features(train, 99)
train.drop(overfitted_features, inplace=True, axis=1)
test.drop(overfitted_features, inplace=True, axis=1)
print(train.shape)
train_labels = y
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
print(train.shape)
# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features.drop('Id', inplace=True, axis=1)
print(all_features.shape)

#### Imputing missing values
print("Total Number of missing value {} before Imputation".format(sum(all_features.isnull().sum())))

def fill_missing_values():
    fillSaleType = all_features[all_features['SaleCondition'] == 'Normal']['SaleType'].mode()[0]        #选择SaleCondition为normal时，SaleType列出现次数最多的值
    all_features['SaleType'].fillna(fillSaleType, inplace=True)

    fillElectrical = all_features[all_features['Neighborhood'] == 'Timber']['Electrical'].mode()[0]
    all_features['Electrical'].fillna(fillElectrical, inplace=True)

    exterior1_neighbor = all_features[all_features['Exterior1st'].isnull()]['Neighborhood'].values[0]     #选择Exteriorlst列为NA时，Neigbourhood列最小的值
    fillExterior1 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior1st'].fillna(fillExterior1, inplace=True)

    exterior2_neighbor = all_features[all_features['Exterior2nd'].isnull()]['Neighborhood'].values[0]
    fillExterior2 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior2nd'].fillna(fillExterior2, inplace=True)

    bsmtNeigh = all_features[all_features['BsmtFinSF1'].isnull()]['Neighborhood'].values[0]
    fillBsmtFinSf1 = all_features[all_features['Neighborhood'] == bsmtNeigh]['BsmtFinSF1'].mode()[0]
    all_features['BsmtFinSF1'].fillna(fillBsmtFinSf1, inplace=True)

    kitchen_grade = all_features[all_features['KitchenQual'].isnull()]['KitchenAbvGr'].values[0]
    fillKitchenQual = all_features[all_features['KitchenAbvGr'] == kitchen_grade]['KitchenQual'].mode()[0]
    all_features['KitchenQual'].fillna(fillKitchenQual, inplace=True)

    all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))#利用MSSubClass对MSZoning列进行分组，随后对每组的缺失值进行赋予本组出现次数最多的值

    all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))                                                                              #利用Neighborhood对LotFrontage进行分组，随后将每组中的缺失值赋予本组的中位数值

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'PoolQC']:
        all_features[col] = all_features[col].fillna('None')

    categorical_cols = all_features.select_dtypes(include='object').columns
    all_features[categorical_cols] = all_features[categorical_cols].fillna('None')                                   #将所有以上未涉及的特征为非数的向量的缺失值设为None

    numeric_cols = all_features.select_dtypes(include='number').columns
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)                                                #将所有以上未涉及的特征为数的向量的缺失值设为0

    all_features['Shed'] = np.where(all_features['MiscFeature'] == 'Shed', 1, 0)                                     #建立新特征Shed

# GarageYrBlt -  missing values there for the building which has no Garage, imputing 0 makes huge difference with other buildings,
# imputing mean doesn't make sense since there is no Garage. So we'll drop it
    all_features.drop(['GarageYrBlt', 'MiscFeature'], inplace=True, axis=1)            #将GarageYrBlt与MiscFeature列删除

    all_features['QualitySF'] = all_features['GrLivArea'] * all_features['OverallQual']


fill_missing_values()

print("Total Number of missing value {} after Imputation".format(sum(all_features.isnull().sum())))         #检查是否还有特征为空的数据

all_features = all_features.drop(['PoolQC'], axis=1)                                    #PoolQC与PoolArea重复，删去

### Feature Creation

all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)       #出售日期-改性年份 可以表示房屋的新旧程度
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']                          #总体材料和加工质量+总体状况的评价 = 总体房屋质量

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']             #地下室总平方英尺+一楼平方英尺+二楼平方英尺 = 房屋总面积
all_features['YrBltAndRemod'] = all_features['YearRemodAdd'] - all_features['YearBuilt']                                #房屋改型日期 -原始建房日期 可以表征房屋的使用情况
all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])                                   #1型成品平方英尺+2型成品平方英尺+一楼平方英尺+二楼平方英尺 = 房屋总落脚面积
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))                     #总浴室数目
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])                                                               #开放式阳台面积+3季门廊面积+封闭门廊面积+屏幕门廊面积+木制甲板面积 = 门廊面积总和
all_features = all_features.drop(['OverallQual','OverallCond','TotalBsmtSF','1stFlrSF'], axis=1)
all_features = all_features.drop(['2ndFlrSF','BsmtFinSF1','BsmtFinSF2','FullBath'], axis=1)
all_features = all_features.drop(['HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF'], axis=1)
all_features = all_features.drop(['3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF'], axis=1)
# Exponential features

#对特征中出现的0进行处理
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

#置车库面积、总地下室面积、2层面积、壁炉、木制甲板面积、封闭门廊面积、3季门廊面积、屏幕门廊面积为1
def booleanFeatures(columns):
    for col in columns:
        all_features[col+"_bool"] = all_features[col].apply(lambda x: 1 if x > 0 else 0)
booleanFeatures(['GarageArea','Fireplaces'])

def logs(columns):
    for col in columns:
        all_features[col+"_log"] = np.log(1.01+all_features[col])

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtUnfSF', 'LowQualFinSF', 'GrLivArea',
                  'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                'MiscVal', 'YearRemodAdd', 'TotalSF']

logs(log_features)

def squares(columns):
    for col in columns:
        all_features[col+"_sq"] =  all_features[col] * all_features[col]

squared_features = ['GarageCars_log','YearRemodAdd', 'LotFrontage_log','GrLivArea_log' ]

squares(squared_features)

#### Feature Transformation
# There is a natural order in their values for few categories, so converting them to numbers gives more meaning
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
quality_cols = ['BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond', 'FireplaceQu', 'GarageQual', 'GarageCond',
                'KitchenQual', 'HeatingQC']
for col in quality_cols:
    all_features[col] = all_features[col].replace(quality_map)

all_features['BsmtExposure'] = all_features['BsmtExposure'].replace({"No": 0, "Mn": 1, "Av": 2, "Gd": 3})

all_features["PavedDrive"] = all_features["PavedDrive"].replace({"N": 0, "P": 1, "Y": 2})

bsmt_ratings = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
bsmt_col = ['BsmtFinType1', 'BsmtFinType2']
for col in bsmt_col:
    all_features[col] = all_features[col].replace(bsmt_ratings)

all_features["GarageScore"] = all_features["GarageQual"] * all_features["GarageCond"]
all_features["ExterScore"] = all_features["ExterQual"] * all_features["ExterCond"]

all_features = pd.get_dummies(all_features).reset_index(drop=True)

X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
train_labels = train_labels.drop(y.index[outliers])

overfitted_features = remove_overfit_features(X, 99)

X = X.drop(overfitted_features, axis=1)
X_test = X_test.drop(overfitted_features, axis=1)

#### Train a model

kf = KFold(n_splits=12, random_state=42, shuffle=True)
# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))                                                          #预测值与真值之间的均方误差

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))      #对模型进行评分，利用模型和数据之间距离的度量的负数作为模型得分，越小越好
    return (rmse)

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression',
                       num_leaves=6,
                       learning_rate=0.01,
                       n_estimators=1000,
                       max_bin=200,
                       bagging_fraction=0.8,
                       bagging_freq=4,
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon=0.008, gamma=0.0003))

# # Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=1000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=1000,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006,random_state=42)

# StackingCVRegressor
stackReg = StackingCVRegressor(regressors=(xgboost, svr, ridge, gbr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True,random_state=42)

model_score = {}

score = cv_rmse(lightgbm)
lgb_model_full_data = lightgbm.fit(X, train_labels)
print("lightgbm: {:.4f}".format(score.mean()))
model_score['lgb'] = score.mean()

score = cv_rmse(xgboost)
xgb_model_full_data = xgboost.fit(X, train_labels)
print("xgboost: {:.4f})".format(score.mean()))
model_score['xgb'] = score.mean()

score = cv_rmse(svr)
svr_model_full_data = svr.fit(X, train_labels)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
model_score['svr'] = score.mean()

score = cv_rmse(ridge)
ridge_model_full_data = ridge.fit(X, train_labels)
print("ridge: {:.4f}".format(score.mean()))
model_score['ridge'] = score.mean()

score = cv_rmse(gbr)
gbr_model_full_data = gbr.fit(X, train_labels)
print("gbr: {:.4f}".format(score.mean()))
model_score['gbr'] = score.mean()

stack_reg_model = stackReg.fit(np.array(X), np.array(train_labels))

def blended_predictions(X,weight):
    return ((weight[0] * ridge_model_full_data.predict(X)) + \
            (weight[1] * svr_model_full_data.predict(X)) + \
            (weight[2] * gbr_model_full_data.predict(X)) + \
            (weight[3] * xgb_model_full_data.predict(X)) + \
            (weight[4] * lgb_model_full_data.predict(X)) + \
            (weight[5] * stack_reg_model.predict(np.array(X))))

# Get final precitions from the blended model

blended_score = rmsle(train_labels, blended_predictions(X,[0.15,0.2,0.1,0.15,0.1,0.3]))
print("blended score: {:.4f}".format(blended_score))
model_score['blended_model'] =  blended_score

pd.Series(model_score).sort_values(ascending=True)

### Predictions

# Read submission csv
submission = pd.read_csv("sample_submission.csv")
# Predictions
submission.iloc[:, 1] = np.floor(np.expm1(blended_predictions(X_test, [0.15, 0.2, 0.1, 0.15, 0.1, 0.3])))

###Predictions
# Write to csv
submission.to_csv("submission", index=False)
print("Mean Absolute Error : " + str(mean_absolute_error(submission.iloc[:, 1], y_test)))