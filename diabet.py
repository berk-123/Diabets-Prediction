#diabetes
"""
Pregnancies = Hamile kalma sayısı
Glucose = Glikoz
Blood Pressure = Kan basıncı
Skin Thickness = Deri kalınlığı
Insulin = İnsülin
BMI (Body Mass Index) = Beden kitle endeksi
Diabetes Pedigree Function = Genetiğe göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
Age = Yaş
Outcome = Diyabet olup olmadığı bilgisi
"""
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile
from sklearn.model_selection import GridSearchCV
from helpers.data_prep import *
from helpers.eda import *
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)

df = pd.read_csv("dataset/diabetes.csv")
df.head()
df.describe().T

df["Insulin"]
df["BloodPressure"]

sns.boxplot(x = df["Insulin"])
plt.show()

sns.boxplot(x = df["Glucose"])
plt.show()

sns.boxplot(x = df["BloodPressure"])
plt.show()



catch_missin_val = [col for col in df.columns if (df[col].min() == 0) and col not in ["Outcome","Pregnancies"]]

df[catch_missin_val] = df[catch_missin_val].replace(0, np.NAN)
df.isnull().sum()

for i in catch_missin_val:
    df[[i]] = df[[i]].replace(0, np.NaN)

def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]

num_cols = [i for i in df.columns if df[i].dtypes != "O" and df[i].nunique() > 10]



for i in num_cols:
    print(i, check_outlier(df, i))

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df,"SkinThickness")


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"SkinThickness")

from helpers.eda import grab_col_names

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df,col))


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


replace_with_thresholds(df, "SkinThickness")

check_outlier(df, "SkinThickness")

df["SkinThickness"].describe().T

df.loc[(df["BMI"] < 18.5), "NEW_BMI_CAT"] ="yağsız,zayıf"
df.loc[(df["BMI"] > 18.5) & (df["BMI"] < 24.9), "NEW_BMI_CAT"] ="sağlıklı kiloda"
df.loc[(df["BMI"] > 25) & (df["BMI"] < 29.9), "NEW_BMI_CAT"] ="çok kilolu"
df.loc[(df["BMI"] > 30) & (df["BMI"] < 39.9), "NEW_BMI_CAT"] ="obezite"
df.loc[(df["BMI"] > 40), "NEW_BMI_CAT"] ="ileri obezite"

df["NEW_BMI_CAT"].head

df.loc[(df["Age"] <= 21), "NEW_AGE_CAT"] = "genç"
df.loc[(df["Age"] > 21) & (df["Age"] <= 65), "NEW_AGE_CAT"] = "yeişkin"
df.loc[(df["Age"] > 65), "NEW_AGE_CAT"] = "yaşlı"


df.loc[(df["Glucose"]) < 70, "NEW_GLUCOSE_NOM"] = "low"
df.loc[(df["Glucose"] >= 70) & (df["Glucose"] < 100), "NEW_GLUCOSE_NOM"] = "normal"
df.loc[(df["Glucose"] >= 100) & (df["Glucose"] <= 125), "NEW_GLUCOSE_NOM"] = "glukoz toleransı"
df.loc[(df["Glucose"]) > 125, "NEW_GLUCOSE_NOM"] = "yüksek"

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"

df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

df.loc[(df["BloodPressure"] < 79), "NEW_BLOODPRESSURE_CAT"] = "Normal"
df.loc[(df["BloodPressure"] > 79) & (df["BloodPressure"] < 89), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S1"
df.loc[(df["BloodPressure"] > 89) & (df["BloodPressure"] < 123), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S2"


df["PREG_AGE"] = df["Pregnancies"] * df["Age"]

df["DiabetesPedigreeFunction"].value_counts()

df["DiaPedFunc_Cat"] = pd.qcut(df["DiabetesPedigreeFunction"], 3, labels=["low", "middle", "up"])

def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.head()

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17)

# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# train hatası
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
