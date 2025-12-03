import time
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from thundersvm import SVR


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


data_train, data_test = train_test_split(
    data_rogue_waves_class, stratify=data_rogue_waves_class[target_cat], train_size=0.80, random_state=seed
)

data_train.reset_index(inplace=True, drop=True)
data_test.reset_index(inplace=True, drop=True)

X_train = data_train[features]
y_train = data_train[target]


# SVM hyperparameters for best model
C = 10
epsilon = 0.01
gamma = 1
kernel = "rbf"

scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)

start = time.time()
model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

model.fit(X_train_transformed, y_train)
end = time.time()
print(f"Time to fit the SVR model: {end - start} seconds")
