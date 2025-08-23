
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------- Utility functions ----------
def load_csv(path):
    return pd.read_csv(path)

def create_synthetic_binary_target(df, feature_cols=None, quantile=0.5):
    df2 = df.copy()
    if feature_cols is None:
        feature_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    row_sums = df2[feature_cols].sum(axis=1)
    thresh = row_sums.quantile(quantile)
    df2["TARGET"] = (row_sums > thresh).astype(int)
    return df2, "TARGET"

def classification_metrics(model, X_train, y_train, X_test, y_test):
    results = {}
    for split_name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        results[split_name] = {"confusion_matrix": cm, "precision": prec, "recall": rec, "f1": f1}
    return results

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

def generate_synthetic_train_data():
    np.random.seed(42)
    X_train = np.random.uniform(1, 10, (20, 2))
    y_train = np.random.choice([0, 1], size=20)
    return X_train, y_train

def plot_training_data(X_train, y_train):
    colors = ["blue" if c == 0 else "red" for c in y_train]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
    plt.title("Synthetic Training Data")
    plt.show()

def generate_test_grid(step=0.1):
    x_values = np.arange(0, 10.1, step)
    y_values = np.arange(0, 10.1, step)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return grid_points, xx, yy

def plot_classification_result(xx, yy, Z, title):
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.title(title)
    plt.show()

def run_knn_and_plot(X_train, y_train, k, step=0.1):
    grid_points, xx, yy = generate_test_grid(step)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    plot_classification_result(xx, yy, Z, f"kNN Classification (k={k})")

# ---------- Main ----------
if __name__ == "__main__":
    # Load and prepare data
    df = load_csv("features_raw.csv")
    df, target_col = create_synthetic_binary_target(df)
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()

    # Train-test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=42, stratify=df[target_col])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # A1: Confusion matrix + metrics
    model_k3 = KNeighborsClassifier(n_neighbors=3)
    model_k3.fit(X_train, y_train)
    metrics_result = classification_metrics(model_k3, X_train, y_train, X_test, y_test)

    # A2: Regression metrics using one feature to predict another
    reg_X = df[[feature_cols[0]]].values
    reg_y = df[feature_cols[1]].values
    Xtr, Xte, ytr, yte = train_test_split(reg_X, reg_y, test_size=0.3, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(Xtr, ytr)
    y_pred = reg_model.predict(Xte)
    reg_metrics = regression_metrics(yte, y_pred)

    # A3: Generate synthetic training data and plot
    syn_X_train, syn_y_train = generate_synthetic_train_data()
    plot_training_data(syn_X_train, syn_y_train)

    # A4: Classify dense grid with k=3
    run_knn_and_plot(syn_X_train, syn_y_train, k=3)

    # A5: Repeat A4 for k=1,5,9
    for k in [1, 5, 9]:
        run_knn_and_plot(syn_X_train, syn_y_train, k=k)

    # A6: Repeat A3–A5 for project data (two features from CSV)
    proj_X = df[[feature_cols[0], feature_cols[1]]].values
    proj_y = df[target_col].values
    plot_training_data(proj_X, proj_y)
    run_knn_and_plot(proj_X, proj_y, k=3)
    for k in [1, 5, 9]:
        run_knn_and_plot(proj_X, proj_y, k=k)

    # A7: Hyperparameter tuning for k
    param_grid = {"n_neighbors": list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    best_k = grid.best_params_["n_neighbors"]

    # Prints
    print("A1) Classification Metrics:", metrics_result)
    print("A2) Regression Metrics:", reg_metrics)
    print("A7) Best k from GridSearchCV:", best_k)

