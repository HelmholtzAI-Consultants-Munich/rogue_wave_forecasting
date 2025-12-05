import time
import pandas as pd


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from thundersvm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

seed = 42

# Data file
file_data = "../data/abin_matrix_full_encoded.csv"

# Thresholds
threshold_upper_limit = 2.7
threshold_non_rogue_wave = 1.5
threshold_rogue_wave = 2.0

# Features and target
target = "AI_10min"
target_cat = "AI_10min_cat"
features = [
    "H_s",
    "lambda_40",
    "lambda_30",
    "L_deep",
    "s",
    "mu",
    "kh",
    "T_p",
    "nu",
    "Q_p",
    "BFI",
    "r",
    "v_wind",
    "v_gust",
    "T_air",
    "p",
    "Delta_p_1h",
]


data_rogue_waves = pd.read_csv(file_data)

data_rogue_waves = data_rogue_waves.loc[data_rogue_waves.AI_10min < threshold_upper_limit]
data_rogue_waves = data_rogue_waves.loc[:, [target] + features]

data_rogue_waves_class = data_rogue_waves.copy()
data_rogue_waves_class[target_cat] = data_rogue_waves_class[target].apply(
    lambda x: 0 if x < threshold_non_rogue_wave else (1 if x < threshold_rogue_wave else 2)
)
data_rogue_waves_class[target_cat] = data_rogue_waves_class[target_cat].astype(int)

X = data_rogue_waves_class.drop(columns=["AI_10min", "AI_10min_cat"])
y = data_rogue_waves_class["AI_10min"]
y_cat = data_rogue_waves_class["AI_10min_cat"]


# SVM hyperparameters for best model
hyperparameters = {"C": 10, "epsilon": 0.01, "gamma": 1, "kernel": "rbf"}

for i in range(5):
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split(X, y_cat)):
        fold = i + 1

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        start = time.time()
        model = SVR(**hyperparameters)
        model.fit(X_train_transformed, y_train)

        end = time.time()
        print(f"Time to fit the SVR model: {end - start} seconds")

        y_pred = model.predict(X_test)
        y_true = y_test

        print(f"MSE fold {fold}: {round(mean_squared_error(y_true, y_pred), 3)}")
        print(f"MAE fold {fold}: {round(mean_absolute_error(y_true, y_pred), 3)}")
        print(f"R^2 fold {fold}: {round(r2_score(y_true, y_pred), 3)}")
        print(f"Spearman R fold {fold}: {round(spearmanr(y_true, y_pred), 3)}")
