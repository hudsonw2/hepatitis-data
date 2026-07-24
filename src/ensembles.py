import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


BASE_ESTIMATOR = SVC(probability=True, random_state=42)


def get_combinations():
    bagging = Pipeline([
        ("scale", StandardScaler()),
        ("clf", BaggingClassifier(estimator=BASE_ESTIMATOR, n_estimators=20, random_state=42)),
    ])

    boosting = Pipeline([
        ("scale", StandardScaler()),
        ("clf", AdaBoostClassifier(estimator=BASE_ESTIMATOR, n_estimators=20, random_state=42)),
    ])


    stacking = Pipeline([
        ("scale", StandardScaler()),
        ("clf", StackingClassifier(
            estimators=[
                ("svm", SVC(probability=True, random_state=42)),
                ("mlp", MLPClassifier(max_iter=2000, random_state=42)),
                ("knn", KNeighborsClassifier()),
            ],
            final_estimator=LogisticRegression(),
            cv=5,
        )),
    ])

    return {"Bagging+SMO": bagging, "AdaBoostM1+SMO": boosting, "Stacking": stacking}


def get_baseline():
    return Pipeline([("scale", StandardScaler()), ("clf", BASE_ESTIMATOR)])


def evaluate(X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    baseline_scores = cross_val_score(get_baseline(), X, y, cv=cv, scoring="accuracy")

    rows = [{"model": "SMO (baseline)", "accuracy": baseline_scores.mean(), "p_value": np.nan}]

    for name, model in get_combinations().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        _, p_value = stats.ttest_rel(scores, baseline_scores)
        rows.append({"model": name, "accuracy": scores.mean(), "p_value": p_value})

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def plot_results(results, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(results["model"], results["accuracy"] * 100, color="#2c3e50")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(50, 100)
    plt.xticks(rotation=15)

    for bar, acc in zip(bars, results["accuracy"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(dataset_name):
    df = pd.read_csv(DATA_DIR / f"hepatitis_{dataset_name}.csv")
    X = df.drop(columns="class")
    y = df["class"]

    results = evaluate(X, y)
    print(f"\n{dataset_name} ({len(df)} instances)")
    print(results.to_string(index=False))

    results.to_csv(RESULTS_DIR / f"ensembles_{dataset_name}.csv", index=False)
    plot_results(
        results,
        f"Ensemble comparison - {dataset_name} data ({len(df)} instances)",
        RESULTS_DIR / f"ensembles_{dataset_name}.png",
    )

    return results


if __name__ == "__main__":
    run("dropna")
    run("imputed")
