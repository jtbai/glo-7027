import pandas as pd
import numpy as np
from scipy import stats

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
# print(train['MasVnrType'].mode())
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

# Quality variables to

cond_qual_ordinal = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5, "No":1, "Mn":2, "Av":3,
                     "GLQ":5, "ALQ":4, "Rec":4, "BLQ":3, "LwQ":2, "Unf":1, "AllPub":4, "NoSewr":3,
                     "NoSeWa":2, "ELO":1, "Fin":3, "RFn":2, "Y":2, "P":1, "N":0, "GdPrv":2, "GdWo":2,
                     "MnPrv":1, "MnWw":1}

yesno = {"Yes":1, "No":0, "Y":1, "N":0}

col_to_replace = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                  "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
                  "BsmtFinType1", "BsmtFinType2", "Utilities", "GarageFinish", "PavedDrive", "Fence"]

for index, row in train.iterrows():
    train.set_value(index, "CentralAir", yesno[row["CentralAir"]])
    for current_col_to_replace in col_to_replace:
            train.set_value(index, current_col_to_replace, cond_qual_ordinal[row[current_col_to_replace]])

# Remove outliers

train = train[train.GrLivArea < 4000]
