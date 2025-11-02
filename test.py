from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
from zlib import crc32
import random
import joblib

data = pd.read_csv(r"C:\Users\Shubham Wats\Desktop\trash 0.x\housing.csv\housing.csv")

""" 

[[[[[[[BEKAAR TAREEKA]]]]]]]

def check_in_test_by_id(identifier, test_ratio):
    hash_id = crc32(np.int64(identifier))
    threshold = test_ratio * 2**32

    return hash_id < threshold

def shuffle_split(data, test_ratio, id_column):
    ids = data[id_column]
    in_test = ids.apply(lambda id_: check_in_test_by_id(id_, test_ratio))
    return data.loc[~in_test], data.loc[in_test]

data_with_index = data.reset_index()

test_set, train_set = shuffle_split(data_with_index, 0.2, "index")

print(train_set)
print("--------------------------------------------------------------------------")
print(test_set)

"""

data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

"""

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(data, data["income_cat"]):
    strat_train_set_n = data.iloc[train_index]
    strat_test_set_n = data.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

strat_test_set["income_cat"].value_counts()

strat_props = strat_test_set["income_cat"].value_counts(normalize=True)

print(strat_props)

"""
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_indice, test_indice in splitter.split(data, data["income_cat"]):
    strat_train_set_n = data.iloc[train_indice]
    strat_test_set_n = data.iloc[test_indice]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]



housing = strat_train_set.drop("median_house_value", axis=1)

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

housing_num = housing.select_dtypes(include=[np.number])
housing_labels = strat_train_set["median_house_value"].copy()

#simple_imputer = SimpleImputer()
iterative_imputer = IterativeImputer(max_iter=10, random_state = 42)

#housing_tr = simple_imputer.fit_transform(housing_num)
housing_tr = iterative_imputer.fit_transform(housing_num)
housing_tr_csv = pd.DataFrame(data=housing_tr, columns=housing_num.columns, index=housing_num.index)


target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing_tr_csv, scaled_labels)

some_new_data = housing_tr_csv.iloc[:100]
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)



actual = housing_labels


comparison = pd.DataFrame({
   # "median_income": housing["median_income"].iloc[:5],
    "median_house_value": actual.iloc[:100],
    "predicted_house_value": predictions.ravel()
})
err = ((comparison['median_house_value'] - comparison['predicted_house_value']).abs())/comparison['median_house_value']
error_ = err*100
comparison["error %"] = error_
pd.set_option('display.max_rows', None)
print(comparison.head(100))

print("-----------------------------------------------------")

mean_error_in_percent = comparison['error %'].mean()
median_error_in_percent = comparison['error %'].median()
max_error_in_percent = comparison['error %'].max()
min_error_in_percent = comparison['error %'].min()
print("mean_error_% :  ", mean_error_in_percent)
print("median_error_% :  ", median_error_in_percent)
print("max_error_% :  ", max_error_in_percent)
print("min_error_% :  ", min_error_in_percent)

print("-----------------------------------------------------")

