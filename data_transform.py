import pandas as pd
import numpy as np
from pickle import dump

train = pd.read_csv("train.csv")

##################
# Missing values #
##################

# Transform NA = None in category
variables_na = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond",
                "PoolQC", "Fence", "MiscFeature"]

for i in variables_na:
    nan_places = pd.isna(train[i])
    train.set_value(nan_places, i, "None")

# Transform par mode
nan_places = pd.isna(train["MasVnrType"])
train.set_value(nan_places, "MasVnrType", "None")
nan_places = pd.isna(train["MasVnrType"])
train.set_value(nan_places, "MasVnrType", 0)
nan_places = pd.isna(train["Electrical"])
mode_to_use = train['Electrical'].mode()
train.set_value(nan_places, "Electrical", mode_to_use[0])

#########################
# Transform and binning #
#########################

# Transform SalePrice to log scale
train["SalePrice"] = np.log(train["SalePrice"])

# Remove outliers

train = train[train.GrLivArea < 4000]

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

for index, row in train.iterrows():
    train.set_value(index, "CentralAir", yesno[row["CentralAir"]])
    train.set_value(index, "PavedDrive", yesno[row["PavedDrive"]])
    for current_col_to_replace in col_to_replace:
            train.set_value(index, current_col_to_replace, cond_qual_ordinal[row[current_col_to_replace]])

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


for index, sale in train.iterrows():
    for col in variables_to_zero:
        current_val = train[col][index]
        if current_val in zero_value:
            val_to_replace = 0
        else:
            val_to_replace = 1
        train.set_value(index, col, val_to_replace)
    train.set_value(index, "MoSold", int(isHighSeason(sale)))


variable_to_categorise = ['MSSubClass', 'MSZoning', 'Street', 'LotShape',
                          'Neighborhood', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                          'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature']

for variable in variable_to_categorise:
    train[variable] = train[variable].astype('category')

dump(train, open('prepared_data.pyk', 'wb'))
