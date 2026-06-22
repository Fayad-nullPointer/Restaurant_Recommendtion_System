import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px

# General style
sns.set_style("whitegrid")

def plot_count(df, column, figsize=(8, 5), show_percentage=False):
    """
    Count plot for categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
    column : str
        Column to visualize.
    figsize : tuple
        Figure size.
    show_percentage : bool
        Whether to display percentages alongside counts.
    """

    plt.figure(figsize=figsize)

    ax = sns.countplot(
        data=df,
        x=column,
        hue=column,
        palette="viridis",
        legend=False
    )

    total = len(df)

    for container in ax.containers:
        labels = []

        for value in container.datavalues:
            if show_percentage:
                percentage = value / total * 100
                labels.append(f"{int(value)}\n({percentage:.2f}%)")
            else:
                labels.append(f"{int(value)}")

        ax.bar_label(container, labels=labels)

    plt.title(f"Count Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_bar(df, x, y, estimator='mean', figsize=(8, 5)):
    """
    Bar plot showing aggregated values.
    """
    plt.figure(figsize=figsize)

    sns.barplot(
        data=df,
        x=x,
        y=y,
        estimator=estimator
    )

    plt.title(f"{y} by {x}")
    plt.tight_layout()
    plt.show()


def plot_box(df, x, y, figsize=(8, 5)):
    """
    Boxplot for outlier detection.
    """
    plt.figure(figsize=figsize)

    sns.boxplot(
        data=df,
        x=x,
        y=y
    )

    plt.title(f"Boxplot of {y} by {x}")
    plt.tight_layout()
    plt.show()


def plot_histogram(df, column, bins=30, kde=True, figsize=(8, 5)):
    """
    Histogram for numerical variables.
    """
    plt.figure(figsize=figsize)

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=kde
    )

    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def plot_kde(df, column, figsize=(8, 5)):
    """
    KDE distribution plot.
    """
    plt.figure(figsize=figsize)

    sns.kdeplot(
        data=df,
        x=column,
        fill=True
    )

    plt.title(f"KDE Plot of {column}")
    plt.tight_layout()
    plt.show()


def plot_violin(df, x, y, figsize=(8, 5)):
    """
    Violin plot for distribution comparison.
    """
    plt.figure(figsize=figsize)

    sns.violinplot(
        data=df,
        x=x,
        y=y
    )

    plt.title(f"Violin Plot of {y} by {x}")
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x, y, hue=None, figsize=(8, 5)):
    """
    Scatter plot for relationships.
    """
    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue
    )

    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, figsize=(12, 8)):
    """
    Correlation heatmap for numerical features.
    """
    plt.figure(figsize=figsize)

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        center=0
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, columns=None, hue=None):
    """
    Pairwise relationships.
    """
    sns.pairplot(
        df[columns] if columns else df,
        hue=hue
    )
    plt.show()


def plot_missing_values(df, figsize=(10, 5)):
    """
    Missing values visualization.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found.")
        return

    plt.figure(figsize=figsize)

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.xticks(rotation=45)
    plt.title("Missing Values")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_missing_percentage(df, figsize=(10, 5)):
    """
    Missing values percentage visualization.
    """
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

    if len(missing_percentage) == 0:
        print("No missing values found.")
        return

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x=missing_percentage.index,
        y=missing_percentage.values
    )

    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.2f}%" for v in missing_percentage.values],
            padding=3
        )

    plt.xticks(rotation=45)
    plt.title("Missing Values Percentage")
    plt.ylabel("Missing Percentage (%)")
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.show()

def plot_geo_locations(df, lat_col, lon_col, hover_col, title="Locations"):
    """
    Plot geographical locations using latitude and longitude.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing location data.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    hover_col : str
        Column to display when hovering.
    title : str
        Plot title.
    """

    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        hover_name=hover_col
    )

    fig.update_layout(
        title=title,
        title_x=0.5
    )

    fig.show()