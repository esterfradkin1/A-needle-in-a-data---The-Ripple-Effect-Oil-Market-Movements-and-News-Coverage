from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
VALIDATION_FILE = OUTPUT_DIR / "sample_sentiment_validation.csv"
OUTPUT_PLOT = OUTPUT_DIR / "vader_confusion_matrix.png"


def run_full_validation_pipeline():
    print("Populating manual labels and evaluating VADER...")

    df = pd.read_csv(VALIDATION_FILE)

    # רשימת תיוג ידני כלכלי-הקשרי ל-150 כותרות המדגם
    manual_labels = [
        # 0-49: כותרות שסווגו במקור ע"י VADER כשליליות
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        # 50-99: כותרות שסווגו במקור ע"י VADER כניטרליות
        "positive",
        "negative",
        "neutral",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "negative",
        "neutral",
        "positive",
        "positive",
        "neutral",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "neutral",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "positive",
        "neutral",
        "negative",
        "neutral",
        "negative",
        "neutral",
        "negative",
        "negative",
        # 100-149: כותרות שסווגו במקור ע"י VADER כחיוביות
        "positive",
        "positive",
        "positive",
        "positive",
        "neutral",
        "negative",
        "positive",
        "negative",
        "negative",
        "positive",
        "positive",
        "negative",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "positive",
        "neutral",
        "negative",
        "positive",
        "neutral",
        "negative",
        "positive",
        "negative",
        "positive",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "negative",
        "positive",
        "neutral",
        "negative",
        "neutral",
        "negative",
        "neutral",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "negative",
        "negative",
        "positive",
    ]

    df["manual_label"] = manual_labels
    df["is_model_correct"] = (
        df["manual_label"] == df["sentiment_label"]
    ).astype(str)

    # שמירת הקובץ המעודכן חזרה
    df.to_csv(VALIDATION_FILE, index=False)
    print(f"Updated {VALIDATION_FILE} with manual ground-truth labels.")

    labels = ["negative", "neutral", "positive"]

    # חישוב Confusion Matrix
    cm = confusion_matrix(
        df["manual_label"], df["sentiment_label"], labels=labels
    )

    # חישוב דוח ביצועים
    report = classification_report(
        df["manual_label"],
        df["sentiment_label"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    print("\n" + "=" * 55)
    print("VADER VALIDATION CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_df.round(3).to_string())

    # יצירת גרף Confusion Matrix מעוצב
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=int(cm[i, j]),
                va="center",
                ha="center",
                size="x-large",
                weight="bold",
                color="white" if cm[i, j] > 25 else "black",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(
        ["Negative", "Neutral", "Positive"], fontsize=10, ha="center"
    )
    ax.set_yticklabels(["Negative", "Neutral", "Positive"], fontsize=10)

    plt.xlabel("Predicted Label (VADER)", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth (Manual Label)", fontsize=11, labelpad=10)
    plt.title(
        "VADER Sentiment Classification Confusion Matrix (N=150)", pad=15
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved Confusion Matrix chart: {OUTPUT_PLOT}")


if __name__ == "__main__":
    run_full_validation_pipeline()