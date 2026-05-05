import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path, sep=";")
    return df


def preprocess_data(df, target):
    # convertir a clasificación binaria
    df[target] = (df[target] >= 6).astype(int)

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler
