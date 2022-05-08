import streamlit as st
import pandas as pd 
import pickle
import sklearn
import numpy as np
from urllib.request import urlopen
import urllib
import cloudpickle as cp

train = pd.read_csv('https://raw.githubusercontent.com/vanilladucky/Housing-Prediction/main/data/cleaned/cleaned_train.csv')
originaltrain = pd.read_csv('https://raw.githubusercontent.com/vanilladucky/Housing-Prediction/main/data/external/train.csv')
originaltrain.fillna('NaN', inplace=True)

st.title("Predicting House Prices in Boston")
st.image('https://images.unsplash.com/photo-1570129477492-45c003edd2be?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80',
         caption = 'Check out how much your imaginery house would cost')


st.header('Dataset')

st.markdown('The **Boston Housing** dataset is used and you can take a look at the dataset with this link.')
st.markdown('https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data')

st.write(originaltrain.head(5)) 

st.markdown("This dataset contains **1460** examples of **79** total features each that could affect a house values in Boston.")



#----------------------------------Hyper Parameters------------------------------------------------#
MSSubClass = st.sidebar.selectbox("Buildling Class", train['MSSubClass'].sort_values().unique())
LotFrontage = st.sidebar.slider("Lot Frontage", float(train['LotFrontage'].min()), float(train['LotFrontage'].max()))
LotArea = st.sidebar.slider("Lot Area", float(train['LotArea'].min()), float(train['LotArea'].max()))
OverallQual = st.sidebar.selectbox("Overall Quality", train['OverallQual'].sort_values().unique())
OverallCond = st.sidebar.selectbox("Overall Condition", train['OverallCond'].sort_values().unique())
YearBuilt = st.sidebar.slider("Year Built", float(train['YearBuilt'].min()), float(train['YearBuilt'].max()))
YearRemodAdd = st.sidebar.slider("Remodel date", float(train['YearRemodAdd'].min())-50, float(train['YearRemodAdd'].max()))
MasVnrArea = st.sidebar.slider("Masonry veneer area", float(train['MasVnrArea'].min()), float(train['MasVnrArea'].max()))
BsmtFinSF1 = st.sidebar.slider("Type 1 finished square feet", float(train['BsmtFinSF1'].min()), float(train['BsmtFinSF1'].max()))
BsmtFinSF2 = st.sidebar.slider("Type 2 finished square feet", float(train['BsmtFinSF2'].min()), float(train['BsmtFinSF2'].max()))
BsmtUnfSF = st.sidebar.slider("Unfinished square feet of basement area", float(train['BsmtUnfSF'].min()), float(train['BsmtUnfSF'].max()))
TotalBsmtSF = st.sidebar.slider("Total square feet of basement area", float(train['TotalBsmtSF'].min()), float(train['TotalBsmtSF'].max()))
firstFlrSF = st.sidebar.slider("First Floor square feet", float(train['1stFlrSF'].min()), float(train['1stFlrSF'].max()))
secondndFlrSF = st.sidebar.slider("Second floor square feet", float(train['2ndFlrSF'].min()), float(train['2ndFlrSF'].max()))
LowQualFinSF = st.sidebar.slider("Low quality finished square feet", float(train['LowQualFinSF'].min()), float(train['LowQualFinSF'].max()))
GrLivArea = st.sidebar.slider("Above ground living area square feet", float(train['GrLivArea'].min()), float(train['GrLivArea'].max()))
BsmtFullBath = st.sidebar.selectbox("Basement full bathrooms", train['BsmtFullBath'].sort_values().unique())
BsmtHalfBath = st.sidebar.selectbox("Basement half bathrooms", train['BsmtHalfBath'].sort_values().unique())
FullBath = st.sidebar.selectbox("Full bathrooms above grade", train['FullBath'].sort_values().unique())
HalfBath = st.sidebar.selectbox("Half baths above grade", train['HalfBath'].sort_values().unique())
BedroomAbvGr = st.sidebar.selectbox("Number of bedrooms above basement level", train['BedroomAbvGr'].sort_values().unique())
KitchenAbvGr = st.sidebar.selectbox("Number of kitchens", train['KitchenAbvGr'].sort_values().unique())
TotRmsAbvGrd = st.sidebar.selectbox("Total rooms above ground", train['TotRmsAbvGrd'].sort_values().unique())
Fireplaces = st.sidebar.selectbox("Number of fireplaces", train['Fireplaces'].sort_values().unique())
GarageYrBlt = st.sidebar.slider("How old is the garage", float(train['GarageYrBlt'].min()), float(train['GarageYrBlt'].max()))
GarageCars = st.sidebar.slider("Size of garage in car capacity", float(train['GarageCars'].min()), float(train['GarageCars'].max()))
GarageArea = st.sidebar.slider("Size of garage in square feet", float(train['GarageArea'].min()), float(train['GarageArea'].max()))
WoodDeckSF = st.sidebar.slider("Wood deck area in square feet", float(train['WoodDeckSF'].min()), float(train['WoodDeckSF'].max()))
OpenPorchSF = st.sidebar.slider("Open porch area in square feet", float(train['OpenPorchSF'].min()), float(train['OpenPorchSF'].max()))
EnclosedPorch = st.sidebar.slider("Enclosed porch area in square feet", float(train['EnclosedPorch'].min()), float(train['EnclosedPorch'].max()))
threeSsnPorch = st.sidebar.slider("Three season porch area in square feet", float(train['3SsnPorch'].min()), float(train['3SsnPorch'].max()))
ScreenPorch = st.sidebar.slider("Screen porch area in square feet", float(train['ScreenPorch'].min()), float(train['ScreenPorch'].max()))
PoolArea = st.sidebar.slider("Pool area in square feet", float(train['PoolArea'].min()), float(train['PoolArea'].max()))
MiscVal = st.sidebar.slider("Value of miscellaneous feature", float(train['MiscVal'].min()), float(train['MiscVal'].max()))
MoSold = st.sidebar.selectbox("Month Sold", (i for i in range(1, 13)))
YrSold = st.sidebar.selectbox("How long ago was the house sold",train['YrSold'].sort_values().unique())
MSZoning = st.sidebar.selectbox("The general zoning classification", originaltrain['MSZoning'].sort_values().unique())
Street = st.sidebar.selectbox("Type of road access", originaltrain['Street'].sort_values().unique())
Alley = st.sidebar.selectbox("Type of alley access", originaltrain['Alley'].sort_values().unique())
LotShape = st.sidebar.selectbox("General shape of property", originaltrain['LotShape'].sort_values().unique())
LandContour = st.sidebar.selectbox("Flatness of the property", originaltrain['LandContour'].sort_values().unique())
Utilities = st.sidebar.selectbox("Type of utilities available", originaltrain['Utilities'].sort_values().unique())
LotConfig = st.sidebar.selectbox("Lot configuration", originaltrain['LotConfig'].sort_values().unique())
LandSlope = st.sidebar.selectbox("Slope of property", originaltrain['LandSlope'].sort_values().unique())
Neighborhood = st.sidebar.selectbox("Physical locations within Ames city limits", originaltrain['Neighborhood'].sort_values().unique())
Condition1 = st.sidebar.selectbox("Proximity to main road or railroad", originaltrain['Condition1'].sort_values().unique())
Condition2 = st.sidebar.selectbox("Proximity to main road or railroad (if a second is present)", originaltrain['Condition2'].sort_values().unique())
BldgType = st.sidebar.selectbox("Type of dwelling", originaltrain['BldgType'].sort_values().unique())
HouseStyle = st.sidebar.selectbox("Style of dwelling", originaltrain['HouseStyle'].sort_values().unique())
RoofStyle = st.sidebar.selectbox("Type of roof", originaltrain['RoofStyle'].sort_values().unique())
RoofMatl = st.sidebar.selectbox("Roof material", originaltrain['RoofMatl'].sort_values().unique())
Exterior1st = st.sidebar.selectbox("Exterior covering on house", originaltrain['Exterior1st'].sort_values().unique())
Exterior2nd = st.sidebar.selectbox("Exterior covering on house (if more than one material", originaltrain['Exterior2nd'].sort_values().unique())
MasVnrType = st.sidebar.selectbox("Masonry veneer type", originaltrain['MasVnrType'].sort_values().unique())
ExterQual = st.sidebar.selectbox("Exterior material quality", originaltrain['ExterQual'].sort_values().unique())
ExterCond = st.sidebar.selectbox("Present condition of the material on the exterior", originaltrain['ExterCond'].sort_values().unique())
Foundation = st.sidebar.selectbox("Type of foundation", originaltrain['Foundation'].sort_values().unique())
BsmtQual = st.sidebar.selectbox("Height of the basement", originaltrain['BsmtQual'].sort_values().unique())
BsmtCond = st.sidebar.selectbox("General condition of the basement", originaltrain['BsmtCond'].sort_values().unique())
BsmtExposure = st.sidebar.selectbox("Walkout or garden level basement walls", originaltrain['BsmtExposure'].sort_values().unique())
BsmtFinType1 = st.sidebar.selectbox("Quality of basement finished area", originaltrain['BsmtFinType1'].sort_values().unique())
BsmtFinType2 = st.sidebar.selectbox("Quality of second finished area (if present)", originaltrain['BsmtFinType2'].sort_values().unique())
Heating = st.sidebar.selectbox("Type of heating", originaltrain['Heating'].sort_values().unique())
HeatingQC = st.sidebar.selectbox("Heating quality and condition", originaltrain['HeatingQC'].sort_values().unique())
CentralAir = st.sidebar.selectbox("Central air conditioning", originaltrain['CentralAir'].sort_values().unique())
Electrical = st.sidebar.selectbox("Electrical system", originaltrain['Electrical'].sort_values().unique())
KitchenQual = st.sidebar.selectbox("Kitchen quality", originaltrain['KitchenQual'].sort_values().unique())
Functional = st.sidebar.selectbox("Home functionality rating", originaltrain['Functional'].sort_values().unique())
FireplaceQu = st.sidebar.selectbox("Fireplace quality", originaltrain['FireplaceQu'].sort_values().unique())
GarageType = st.sidebar.selectbox("Garage location", originaltrain['GarageType'].sort_values().unique())
GarageFinish = st.sidebar.selectbox("Interior finish of the garage", originaltrain['GarageFinish'].sort_values().unique())
GarageQual = st.sidebar.selectbox("Garage quality", originaltrain['GarageQual'].sort_values().unique())
GarageCond = st.sidebar.selectbox("Garage condition", originaltrain['GarageCond'].sort_values().unique())
PavedDrive = st.sidebar.selectbox("Paved driveway", originaltrain['PavedDrive'].sort_values().unique())
PoolQC = st.sidebar.selectbox("Pool quality", originaltrain['PoolQC'].sort_values().unique())
Fence = st.sidebar.selectbox("Fence quality", originaltrain['Fence'].sort_values().unique())
MiscFeature = st.sidebar.selectbox("Miscellaneous feature", originaltrain['MiscFeature'].sort_values().unique())
SaleType = st.sidebar.selectbox("Type of sale", originaltrain['SaleType'].sort_values().unique())
SaleCondition = st.sidebar.selectbox("Condition  of sale", originaltrain['SaleCondition'].sort_values().unique())
#----------------------------------Hyper Parameters------------------------------------------------#

#----------------------------Categorical Mapping-----------------------------------#
categorical_map = {'MSZoning': {'C (all)': 0, 'FV': 1, 'RH': 2, 'RL': 3, 'RM': 4},
 'Street': {'Grvl': 0, 'Pave': 1},
 'Alley': {'Grvl': 0, 'Pave': 1, 'NaN': 2},
 'LotShape': {'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3},
 'LandContour': {'Bnk': 0, 'HLS': 1, 'Low': 2, 'Lvl': 3},
 'Utilities': {'AllPub': 0, 'NoSeWa': 1},
 'LotConfig': {'Corner': 0, 'CulDSac': 1, 'FR2': 2, 'FR3': 3, 'Inside': 4},
 'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
 'Neighborhood': {'Blmngtn': 0,
  'Blueste': 1,
  'BrDale': 2,
  'BrkSide': 3,
  'ClearCr': 4,
  'CollgCr': 5,
  'Crawfor': 6,
  'Edwards': 7,
  'Gilbert': 8,
  'IDOTRR': 9,
  'MeadowV': 10,
  'Mitchel': 11,
  'NAmes': 12,
  'NPkVill': 13,
  'NWAmes': 14,
  'NoRidge': 15,
  'NridgHt': 16,
  'OldTown': 17,
  'SWISU': 18,
  'Sawyer': 19,
  'SawyerW': 20,
  'Somerst': 21,
  'StoneBr': 22,
  'Timber': 23,
  'Veenker': 24},
 'Condition1': {'Artery': 0,
  'Feedr': 1,
  'Norm': 2,
  'PosA': 3,
  'PosN': 4,
  'RRAe': 5,
  'RRAn': 6,
  'RRNe': 7,
  'RRNn': 8},
 'Condition2': {'Artery': 0,
  'Feedr': 1,
  'Norm': 2,
  'PosA': 3,
  'PosN': 4,
  'RRAe': 5,
  'RRAn': 6,
  'RRNn': 7},
 'BldgType': {'1Fam': 0, '2fmCon': 1, 'Duplex': 2, 'Twnhs': 3, 'TwnhsE': 4},
 'HouseStyle': {'1.5Fin': 0,
  '1.5Unf': 1,
  '1Story': 2,
  '2.5Fin': 3,
  '2.5Unf': 4,
  '2Story': 5,
  'SFoyer': 6,
  'SLvl': 7},
 'RoofStyle': {'Flat': 0,
  'Gable': 1,
  'Gambrel': 2,
  'Hip': 3,
  'Mansard': 4,
  'Shed': 5},
 'RoofMatl': {'ClyTile': 0,
  'CompShg': 1,
  'Membran': 2,
  'Metal': 3,
  'Roll': 4,
  'Tar&Grv': 5,
  'WdShake': 6,
  'WdShngl': 7},
 'Exterior1st': {'AsbShng': 0,
  'AsphShn': 1,
  'BrkComm': 2,
  'BrkFace': 3,
  'CBlock': 4,
  'CemntBd': 5,
  'HdBoard': 6,
  'ImStucc': 7,
  'MetalSd': 8,
  'Plywood': 9,
  'Stone': 10,
  'Stucco': 11,
  'VinylSd': 12,
  'Wd Sdng': 13,
  'WdShing': 14},
 'Exterior2nd': {'AsbShng': 0,
  'AsphShn': 1,
  'Brk Cmn': 2,
  'BrkFace': 3,
  'CBlock': 4,
  'CmentBd': 5,
  'HdBoard': 6,
  'ImStucc': 7,
  'MetalSd': 8,
  'Other': 9,
  'Plywood': 10,
  'Stone': 11,
  'Stucco': 12,
  'VinylSd': 13,
  'Wd Sdng': 14,
  'Wd Shng': 15},
 'MasVnrType': {'BrkCmn': 0, 'BrkFace': 1, 'None': 2, 'Stone': 3, 'NaN': 4},
 'ExterQual': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3},
 'ExterCond': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4},
 'Foundation': {'BrkTil': 0,
  'CBlock': 1,
  'PConc': 2,
  'Slab': 3,
  'Stone': 4,
  'Wood': 5},
 'BsmtQual': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3, 'NaN': 4},
 'BsmtCond': {'Fa': 0, 'Gd': 1, 'Po': 2, 'TA': 3, 'NaN': 4},
 'BsmtExposure': {'Av': 0, 'Gd': 1, 'Mn': 2, 'No': 3, 'NaN': 4},
 'BsmtFinType1': {'ALQ': 0,
  'BLQ': 1,
  'GLQ': 2,
  'LwQ': 3,
  'Rec': 4,
  'Unf': 5,
  'NaN': 6},
 'BsmtFinType2': {'ALQ': 0,
  'BLQ': 1,
  'GLQ': 2,
  'LwQ': 3,
  'Rec': 4,
  'Unf': 5,
  'NaN': 6},
 'Heating': {'Floor': 0,
  'GasA': 1,
  'GasW': 2,
  'Grav': 3,
  'OthW': 4,
  'Wall': 5},
 'HeatingQC': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4},
 'CentralAir': {'N': 0, 'Y': 1},
 'Electrical': {'FuseA': 0,
  'FuseF': 1,
  'FuseP': 2,
  'Mix': 3,
  'SBrkr': 4,
  'NaN': 5},
 'KitchenQual': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3},
 'Functional': {'Maj1': 0,
  'Maj2': 1,
  'Min1': 2,
  'Min2': 3,
  'Mod': 4,
  'Sev': 5,
  'Typ': 6},
 'FireplaceQu': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4, 'NaN': 5},
 'GarageType': {'2Types': 0,
  'Attchd': 1,
  'Basment': 2,
  'BuiltIn': 3,
  'CarPort': 4,
  'Detchd': 5,
  'NaN': 6},
 'GarageFinish': {'Fin': 0, 'RFn': 1, 'Unf': 2, 'NaN': 3},
 'GarageQual': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4, 'NaN': 5},
 'GarageCond': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4, 'NaN': 5},
 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
 'PoolQC': {'Ex': 0, 'Fa': 1, 'Gd': 2, 'NaN': 3},
 'Fence': {'GdPrv': 0, 'GdWo': 1, 'MnPrv': 2, 'MnWw': 3, 'NaN': 4},
 'MiscFeature': {'Gar2': 0, 'Othr': 1, 'Shed': 2, 'TenC': 3, 'NaN': 4},
 'SaleType': {'COD': 0,
  'CWD': 1,
  'Con': 2,
  'ConLD': 3,
  'ConLI': 4,
  'ConLw': 5,
  'New': 6,
  'Oth': 7,
  'WD': 8},
 'SaleCondition': {'Abnorml': 0,
  'AdjLand': 1,
  'Alloca': 2,
  'Family': 3,
  'Normal': 4,
  'Partial': 5}}
#----------------------------Categorical Mapping-----------------------------------#


features = [[MSSubClass,
 LotFrontage,
 LotArea,
 OverallQual,
 OverallCond,
 YearBuilt,
 YearRemodAdd,
 MasVnrArea,
 BsmtFinSF1,
 BsmtFinSF2,
 BsmtUnfSF,
 TotalBsmtSF,
 firstFlrSF,
 secondndFlrSF,
 LowQualFinSF,
 GrLivArea,
 BsmtFullBath,
 BsmtHalfBath,
 FullBath,
 HalfBath,
 BedroomAbvGr,
 KitchenAbvGr,
 TotRmsAbvGrd,
 Fireplaces,
 GarageYrBlt,
 GarageCars,
 GarageArea,
 WoodDeckSF,
 OpenPorchSF,
 EnclosedPorch,
 threeSsnPorch,
 ScreenPorch,
 PoolArea,
 MiscVal,
 MoSold,
 YrSold,
 categorical_map['MSZoning'][MSZoning],
 categorical_map['Street'][Street],
 categorical_map['Alley'][Alley],
 categorical_map['LotShape'][LotShape],
 categorical_map['LandContour'][LandContour],
 categorical_map['Utilities'][Utilities],
 categorical_map['LotConfig'][LotConfig],
 categorical_map['LandSlope'][LandSlope],
 categorical_map['Neighborhood'][Neighborhood],
 categorical_map['Condition1'][Condition1],
 categorical_map['Condition2'][Condition2],
 categorical_map['BldgType'][BldgType],
 categorical_map['HouseStyle'][HouseStyle],
 categorical_map['RoofStyle'][RoofStyle],
 categorical_map['RoofMatl'][RoofMatl],
 categorical_map['Exterior1st'][Exterior1st],
 categorical_map['Exterior2nd'][Exterior2nd],
 categorical_map['MasVnrType'][MasVnrType],
 categorical_map['ExterQual'][ExterQual],
 categorical_map['ExterCond'][ExterCond],
 categorical_map['Foundation'][Foundation],
 categorical_map['BsmtQual'][BsmtQual],
 categorical_map['BsmtCond'][BsmtCond],
 categorical_map['BsmtExposure'][BsmtExposure],
 categorical_map['BsmtFinType1'][BsmtFinType1],
 categorical_map['BsmtFinType2'][BsmtFinType2],
 categorical_map['Heating'][Heating],
 categorical_map['HeatingQC'][HeatingQC],
 categorical_map['CentralAir'][CentralAir],
 categorical_map['Electrical'][Electrical],
 categorical_map['KitchenQual'][KitchenQual],
 categorical_map['Functional'][Functional],
 categorical_map['FireplaceQu'][FireplaceQu],
 categorical_map['GarageType'][GarageType],
 categorical_map['GarageFinish'][GarageFinish],
 categorical_map['GarageQual'][GarageQual],
 categorical_map['GarageCond'][GarageCond],
 categorical_map['PavedDrive'][PavedDrive],
 categorical_map['PoolQC'][PoolQC],
 categorical_map['Fence'][Fence],
 categorical_map['MiscFeature'][MiscFeature],
 categorical_map['SaleType'][SaleType],
 categorical_map['SaleCondition'][SaleCondition]]]

st.header("Predicting")
st.write("Go ahead and alter the 79 different features that could possibly influence the price of a house.")
st.write("If you are unsure of any of the features options, please do visit the link above where more explanation is provided.")
st.write("After which, you can press the button below to see how much your house might cost.")

#--------Prediction------------#
if st.button('Predict'):
    # Loading in our saved model
    model = cp.load(urlopen("https://www.dropbox.com/s/toogefk4yiecipn/finalized_model.sav?dl=0", 'rb')) 
    predicted_price = model.predict(np.array(features))
    st.write("# Your predicted home price is")
    st.write("# ${:.2f}".format(predicted_price[0]))
#--------Prediction------------#

st.header("Explanation")
st.markdown("The prediction algorithm is mainly based on two supervised learning, regression algorithms, known as **Random Forest** and **XGBoost**, which are some of the most efficient algorithms out there.")
st.markdown("These two were picked out from several other algorithms and after undergoing **random search** and then cross validation **grid search** hyperparameter tuning, they were further **stacked** to create an ensembled model which showcased a lower error value with the given dataset.")
st.markdown("For those of you interested in the detailed code behind this project, do feel free to visit [my github page](https://github.com/vanilladucky/Housing-Prediction) to get a better understanding of the whole process from data cleaning, feature engineering to model selection")

