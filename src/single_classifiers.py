import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def get_models():
    return {
        "NB": Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
        "J48": Pipeline([("scale", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))]),
        "IBk": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
        "SMO": Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
        "MLP": Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(max_iter=2000, random_state=42))]),
    }


def evaluate_models(X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rows = []

    for name, model in get_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        rows.append({
            "classifier": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
        })

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def plot_accuracy(results, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(results["classifier"], results["accuracy"] * 100, color="#c0392b")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(50, 100)

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

    results = evaluate_models(X, y)
    print(f"\n{dataset_name} ({len(df)} instances)")
    print(results.to_string(index=False))

    results.to_csv(RESULTS_DIR / f"single_classifiers_{dataset_name}.csv", index=False)
    plot_accuracy(
        results,
        f"Single classifiers - {dataset_name} data ({len(df)} instances)",
        RESULTS_DIR / f"single_classifiers_{dataset_name}.png",
    )

    return results


if __name__ == "__main__":
    run("dropna")
    run("imputed")
