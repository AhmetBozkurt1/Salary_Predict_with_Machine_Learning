# SALARY PREDICT WITH MACHINE LEARNING

# İŞ PROBLEMİ
# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir
# makine öğrenmesi modeli geliştiriniz.

# VERİ  SETİ HİKAYESİ
# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır. Veri seti
# 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri orijinal olarak Sports
# Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing
# Company, New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

# DEĞİŞKENLER
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Veri Seti Okutma
df = pd.read_csv("hitters.csv")
df.head()
df.isnull().sum()
df.shape
df.info()

# Değişkenleri Sınıflandırma
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik,numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir
    cat_th: int, float
        Numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        Kategorik fakat kardinal değişkenlerin sınıf eşik değeri

      Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik Değişkenlerin Veriseti İçindeki Durumları
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# Kategorik Değişkenlerle Hedef Değişkeni İnceleyelim.
def target_category(dataframe,  target, col_category):
    print(dataframe.groupby(col_category).agg({target: "mean"}))
    print("#" * 40)

for col in cat_cols:
    target_category(df, "Salary", col)

# Aykırı Değerlere Bakalım.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe.loc[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit

df[num_cols].describe().T

# Eksik Değerlere Bakalım.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns

missing_values_table(df)

# Korelasyon İnceleyelim.
def corr_map(df, width=14, height=6, annot_kws=15, corr_th=0.7):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize = (width,height))
    sns.heatmap(df.corr(),
                annot= True,
                fmt = ".2f",
                ax=ax,
                vmin = -1,
                vmax = 1,
                cmap = "RdBu",
                mask = mtx,
                linewidth = 0.4,
                linecolor = "black",
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0,size=15)
    plt.xticks(rotation=75,size=15)
    plt.title('\nCorrelation Map\n', size = 40)
    plt.show();
    return drop_list

corr_map(df[num_cols])

# BASE MODEL
df_base = df.copy()
# Eksik değerli satırları da kaldırıyorum.
df_base = df_base.dropna()

X = df_base.drop("Salary", axis=1)
y = df_base["Salary"]

# Kategorik değişkenlerde sadece ikişer unique değer olduğu için sadece LabelEncoder yapalım.
def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return

for col in cat_cols:
    label_encoder(X, col)

models = [('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbosity=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, model in models:
    rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: ({round(rmse, 3)}), ({name})")

# Model Sonuçları
# RMSE: (366.688), (CART)
# RMSE: (269.238), (RF)
# RMSE: (278.375), (GBM)
# RMSE: (322.857), (XGBoost)
# RMSE: (281.516), (LightGBM)
# RMSE: (267.494), (CatBoost)

# FEATURE ENGINEERING

# Ekisk değer Salary değişkeninde yani hedef değişkeninde olduğu için bunu hiç doldurmuyorum.İlk olarak bu boş satırları
# daha sonra tahmin etmek için ayrı bir nesneye atıyorum.Sonra veri setinden çıkarıyorum.
null_salary = df.loc[df["Salary"].isnull()]
df = df.dropna()

# Aykır Değerlere bakalım.
for col in num_cols:
    print(col, check_outlier(df, col))  # Aykırı Değer YOK!

# Şimdi de veri seti içerisinden yeni feature'lar üretelim.
df.columns = [col.title() for col in df.columns]

# Lig Değiştirenleri Göstereliö.
df["League_Change"] = df.apply(lambda x: 1 if x["League"] != x["Newleague"] else 0, axis=1)

# Oynadığı sene ile Elde edilen Skorlar
df["New_Years_Cat"] = df["Catbat"] / df["Years"]
df["New_Years_Chits"] = df["Chits"] / df["Years"]
df["New_Years_Chm"] = df["Chmrun"] / df["Years"]
df["New_Years_Cruns"] = df["Cruns"] / df["Years"]
df["New_Years_Crbi"] = df["Crbi"] / df["Years"]
df["New_Years_Cwalks"] = df["Cwalks"] / df["Years"]

# Genel istatisklere göre sezonda başarılı olma durumları
features = [("New_Years_Cat", "Atbat"),
            ("New_Years_Chits", "Hits"),
            ("New_Years_Chm", "Hmrun"),
            ("New_Years_Cruns", "Runs"),
            ("New_Years_Crbi", "Rbi"),
            ("New_Years_Cwalks", "Walks")]
def player_sucsess(data):
    for new_feature, feature in data:
        df["Sucsess_" + feature] = df.apply(lambda x: 1 if x[new_feature] <= x[feature] else 0, axis=1)

player_sucsess(features)

# Vuruş İsabet Yüzdesi
df["New_Sucsess_Hits"] = (df["Hits"] / df["Atbat"]) * 100

# En Değerli Vuruş Yüzdesi
df["New_Sucsess_Hmrun"] = (df["Hmrun"] / df["Atbat"]) * 100

# Değişkenleri tekrar sınıflandırıyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Encoding İşlemleri

# Değişkenlerden sadece 3 tane object dtype değişken var onlara encode uygulayalım.
binary_col = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

# Label Encoder
def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_col:
    label_encoder(df, col)


# Scale İşlemleri
scale_list = [col for col in df.columns if df[col].nunique() > 2 and col not in "Salary"]
rs_scale = RobustScaler()
df[scale_list] = rs_scale.fit_transform(df[scale_list])

# MODEL AŞAMASI
X = df.drop("Salary", axis=1)
y = df["Salary"]

models = [('CART', DecisionTreeRegressor()),
          ('LR', LinearRegression()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbosity=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False)),
          ('KNN', KNeighborsRegressor())]

for name, model in models:
    rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: ({round(rmse, 4)}), ({name})")

# Model Başarıları
# RMSE: (391.9794), (CART)
# RMSE: (338.8778), (LR)
# RMSE: (264.7759), (RF)
# RMSE: (275.9017), (GBM)
# RMSE: (305.0661), (XGBoost)
# RMSE: (287.9951), (LightGBM)
# RMSE: (265.8829), (CatBoost)
# RMSE: (305.0237), (KNN)

# Hiperparametre Optimizasyonu
# Model olarak RandomForest ile devam edelim.
rf_model = RandomForestRegressor()

# RandomSearchCV ile parametreleri belirleyelim.
rf_random_search = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random_search_best_grid = RandomizedSearchCV(estimator=rf_model,
                                                param_distributions=rf_random_search,
                                                n_iter=100,
                                                n_jobs=-1,
                                                verbose=True,
                                                cv=5,
                                                random_state=17).fit(X, y)

rf_rs_params = rf_random_search_best_grid.best_params_


# GridSearchCV kullanalım.
rf_params = {"max_depth": [30, 33, 35, 37, 42],
             "max_features": [4, 5, 8],
             "min_samples_split": [1, 2, 3],
             "n_estimators": [1000, 1250, 1450, 1600]}

rf_model_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final Model
rf_final_model = rf_model.set_params(**rf_model_best_grid.best_params_).fit(X, y)
rmse_final = np.mean(np.sqrt(-cross_val_score(rf_final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: ({round(rmse_final, 4)}), RF")  # RMSE: (261.444), RF
# Modelimizde RMSE skorunu bu optimizasyon ile biraz düşürdük.

################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdirelim..
################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_final_model, X, num=15)
# Modelimize etki eden değişkenlere baktığımızda yeni feature olarak oluşturduğumuz birçok değişken bulunuyor.

# Grafikte modelimizin tahminlerini inceleyelim.
train_predictions = rf_final_model.predict(X)

plt.figure(figsize=(14, 6))
plt.subplot2grid((1, 2), (0, 0), colspan=2, rowspan=1)
plt.plot(y.values, label='Gerçek Değerler')
plt.plot(train_predictions, label='Tahmin Edilen Değerler', linestyle='--')
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.title('Eğitim Seti: Gerçek vs Tahmin')
plt.legend()
plt.tight_layout()
plt.show()
