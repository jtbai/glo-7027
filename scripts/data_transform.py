import pandas as pd
import numpy as np
from pickle import dump
from os import path

from pandas.tests.io.parser import index_col

DATA_PATH = "data"
train_input_file_name = "train.csv"
test_input_file_name = "test.csv"
outout_file_name = 'prepared_data.pyk'

train = pd.read_csv(path.join(DATA_PATH, train_input_file_name), index_col = "Id")
test = pd.read_csv(path.join(DATA_PATH, test_input_file_name), index_col = "Id")

train["data_type"] = "train"
test["data_type"] = "test"

dataset = train.append(test)

##################
# Missing values #
##################

# Transform NA = None in category
variables_na = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond",
                "PoolQC", "Fence", "MiscFeature"]

for i in variables_na:
    nan_places = pd.isna(dataset[i])
    dataset.set_value(nan_places, i, "None")

# Transform par mode

nan_places = pd.isna(dataset["GarageCars"])
dataset.set_value(nan_places, "GarageCars", 0)
nan_places = pd.isna(dataset["GarageArea"])
dataset.set_value(nan_places, "GarageArea", 0)

nan_places = pd.isna(dataset["KitchenQual"])
dataset.set_value(nan_places, "KitchenQual", "TA")


nan_places = pd.isna(dataset["BsmtHalfBath"])
dataset.set_value(nan_places, "BsmtHalfBath", "None")
nan_places = pd.isna(dataset["BsmtFullBath"])
dataset.set_value(nan_places, "BsmtFullBath", "None")


nan_places = pd.isna(dataset["MasVnrType"])
dataset.set_value(nan_places, "MasVnrType", "None")
nan_places = pd.isna(dataset["MasVnrType"])
dataset.set_value(nan_places, "MasVnrType", 0)
nan_places = pd.isna(dataset["Electrical"])
mode_to_use = dataset['Electrical'].mode()
dataset.set_value(nan_places, "Electrical", mode_to_use[0])

nan_places = pd.isna(dataset["BsmtFinSF1"])
dataset.set_value(nan_places, "BsmtFinSF1", 0)
nan_places = pd.isna(dataset["BsmtFinSF2"])
dataset.set_value(nan_places, "BsmtFinSF2", 0)
nan_places = pd.isna(dataset["BsmtUnfSF"])
dataset.set_value(nan_places, "BsmtUnfSF", 0)
nan_places = pd.isna(dataset["TotalBsmtSF"])
dataset.set_value(nan_places, "TotalBsmtSF", 0)



dataset.LotFrontage = dataset.LotFrontage.fillna(dataset.LotFrontage.mean())
dataset.MasVnrArea = dataset.LotFrontage.fillna(dataset.MasVnrArea.mean())

#########################
# Transform and binning #
#########################

# Transform SalePrice to log scale
if "SalePrice" in dataset:
    dataset["SalePrice"] = np.log(dataset["SalePrice"])

# Remove outliers

dataset = dataset[~ np.logical_and(dataset.GrLivArea > 4000, dataset.data_type == "train")]

# Qualitaive ordinal to quantitative ordinal

cond_qual_ordinal = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "No": 1, "Mn": 2, "Av": 3,
                     "GLQ": 5, "ALQ": 4, "Rec": 4, "BLQ": 3, "LwQ": 2, "Unf": 1, "AllPub": 4, "NoSewr": 3,
                     "NoSeWa": 2, "ELO": 1, "Fin": 3, "RFn": 2, "Y": 2, "P": 1, "N": 0, "GdPrv": 2, "GdWo": 2,
                     "MnPrv": 1, "MnWw": 1, "FuseA": 2, "FuseF": 1, "SBrkr": 3, "FuseP": 0, "Mix": 2, "Maj1": 3,
                     'Maj2': 2, "Min1": 6, "Min2": 5, "Mod": 4, "Sev": 1, "Sal": 0, "Typ": 7}

yesno = {"Yes": 1, "No": 0, "Y": 1, "N": 0, "P": 1}

col_to_replace = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                  "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
                  "BsmtFinType1", "BsmtFinType2", "Utilities", "GarageFinish", "Fence", "Electrical", "Functional"]

default_values = {
    'Utilities': "AllPub",
    "Functional": "Typ"
}

for index, row in dataset.iterrows():
    dataset.set_value(index, "CentralAir", yesno[row["CentralAir"]])
    dataset.set_value(index, "PavedDrive", yesno[row["PavedDrive"]])
    for current_col_to_replace in col_to_replace:
        if current_col_to_replace in default_values and isinstance(row[current_col_to_replace], float) and np.isnan(row[current_col_to_replace]):
            dataset.set_value(index, current_col_to_replace, default_values[current_col_to_replace])
            row = dataset.iloc[index]
        dataset.set_value(index, current_col_to_replace, cond_qual_ordinal[row[current_col_to_replace]])

# Qualitative to quantitative

variables_to_zero = ["MSZoning", "LandContour", "LotConfig", "LandSlope", "Condition1", "Condition2",
                     "BldgType", "RoofMatl", "Heating", "SaleType", "SaleCondition"]
zero_value = ["RL", "Lvl", "Inside", "Gtl", "Norm", "1Fam", "CompShg", "Gas", "WD", "Normal"]

def isHighSeason(sale):
    month = sale["MoSold"]
    if month in [3, 4, 5, 6]:
        return True
    else:
        return False

def garageYear(sale):
    garage = sale["GarageYrBlt"]
    if garage == 2207:
        garage = 2007
    if garage == "None":
        return 0
    else:
        return garage

for index, sale in dataset.iterrows():
    for col in variables_to_zero:
        current_val = dataset[col][index]
        if current_val in zero_value:
            val_to_replace = 0
        else:
            val_to_replace = 1
        dataset.set_value(index, col, val_to_replace)
    dataset.set_value(index, "MoSold", int(isHighSeason(sale)))
    dataset.set_value(index, "GarageYrBlt", int(garageYear(sale)))

variable_to_categorise = ['MSSubClass', 'MSZoning', 'Street', 'LotShape',
                          'Neighborhood', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                          'Foundation', 'GarageType', 'MiscFeature']

for variable in variable_to_categorise:
    dataset[variable] = dataset[variable].astype('category')

dataset["GarageYrBlt"] = dataset["GarageYrBlt"].astype('float')

dataset = pd.get_dummies(dataset)
dump(dataset[dataset.data_type_train==1], open(path.join(DATA_PATH, "train_{}".format(outout_file_name)), 'wb'))
dump(dataset[dataset.data_type_test==1], open(path.join(DATA_PATH, "test_{}".format(outout_file_name)), 'wb'))
