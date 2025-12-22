"""
eda_jobrole.py - minimal, working EDA script (no ydata-profiling)

Saves images to ./eda_outputs/
Run: python eda_jobrole.py
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

plt.rcParams.update({"figure.dpi": 120})

INPUT_PATH = Path("JobRole.xlsx")
OUTPUT_DIR = Path("eda_outputs")
CGPA_COL = "CGPA"
EXP_COL = "Years of Experience"
JOBROLE_COL = "Job Role"
SKILLS_COL = "Skills"

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_data(path: Path):
    if not path.exists():
        print(f"ERROR: Input file not found at: {path.resolve()}")
        return None
    try:
        df = pd.read_excel(path, engine="openpyxl")
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
        return df
    except Exception as e:
        print("Failed to read Excel file:", e)
        return None

def correlation_heatmap(df, outpath):
    print("Creating correlation heatmap...")
    corr_df = df.select_dtypes(include=["number"]).copy()
    # add some encoded categorical columns if present
    for c in ["Degree", "Major", "Preferred Industry", JOBROLE_COL]:
        if c in df.columns:
            corr_df[c + "_code"] = df[c].astype("category").cat.codes
    if corr_df.shape[1] < 2:
        print("Not enough numeric columns for correlation heatmap. Skipping.")
        return
    corr = corr_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink":0.7})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath.name)

def scatter_experience_vs_cgpa(df, cgpa_col, exp_col, outpath):
    print("Creating Experience vs CGPA scatter plot...")
    if cgpa_col not in df.columns or exp_col not in df.columns:
        print("Required columns missing for scatter plot. Skipping.")
        return
    sub = df[[exp_col, cgpa_col]].dropna()
    if sub.empty:
        print("No data for scatter plot after dropping NA. Skipping.")
        return
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=sub, x=exp_col, y=cgpa_col, alpha=0.7)
    try:
        sns.regplot(data=sub, x=exp_col, y=cgpa_col, scatter=False, lowess=True)
    except Exception:
        pass
    plt.xlabel(exp_col)
    plt.ylabel(cgpa_col)
    plt.title(f"{exp_col} vs {cgpa_col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath.name)

def jobrole_frequency_plot(df, jobrole_col, outpath, top_n=20):
    print("Creating job role frequency plot...")
    if jobrole_col not in df.columns:
        print("Job role column not found. Skipping.")
        return
    counts = df[jobrole_col].fillna("Unknown").value_counts().nlargest(top_n)
    plt.figure(figsize=(8, max(3, 0.3 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, orient="h")
    plt.xlabel("Count")
    plt.ylabel("Job Role")
    plt.title(f"Top {len(counts)} Job Roles")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath.name)

def skills_wordcloud(df, skills_col, outpath):
    print("Creating skills word cloud...")
    if skills_col not in df.columns:
        print("Skills column not found. Skipping.")
        return
    skills_series = df[skills_col].dropna().astype(str)
    tokens = []
    for cell in skills_series:
        parts = [p.strip() for p in cell.replace(";", ",").replace("/", ",").split(",") if p.strip()]
        tokens.extend(parts)
    if not tokens:
        print("No tokens found for word cloud. Skipping.")
        return
    freq = {}
    for t in tokens:
        key = t.lower()
        freq[key] = freq.get(key, 0) + 1
    wc = WordCloud(width=1200, height=600, background_color="white", collocations=False)
    wc.generate_from_frequencies(freq)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath.name)

def main():
    print("SCRIPT STARTED")
    safe_mkdir(OUTPUT_DIR)
    df = load_data(INPUT_PATH)
    if df is None:
        print("Exiting due to missing data.")
        return
    correlation_heatmap(df, OUTPUT_DIR / "correlation_heatmap.png")
    scatter_experience_vs_cgpa(df, CGPA_COL, EXP_COL, OUTPUT_DIR / "scatter_experience_cgpa.png")
    jobrole_frequency_plot(df, JOBROLE_COL, OUTPUT_DIR / "jobrole_freq.png")
    skills_wordcloud(df, SKILLS_COL, OUTPUT_DIR / "skills_wordcloud.png")
    print("SCRIPT FINISHED. Check the folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
