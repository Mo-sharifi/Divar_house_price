import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from data_loader import data_load
from data_preprocessing import preprocess_data


def plot_price_vs_area(
    df: pd.DataFrame, x_col: str = "Area", y_col: str = "Price", hue_col: str = "Room"
) -> None:
    """
    Plot a scatterplot of y_col vs x_col with a regression line and optional hue-based coloring.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        x_col (str): Name of the column to plot on the x-axis. Default is "Area".
        y_col (str): Name of the column to plot on the y-axis. Default is "Price".
        hue_col (str): Column used for color grouping (hue). Default is "Room".

    Displays:
        A scatterplot with regression line and colored points based on hue_col.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x=x_col, y=y_col, hue=hue_col, palette="coolwarm", alpha=0.7, data=df, ax=ax
    )

    sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color="black", ax=ax)

    ax.set_title(f"{y_col} vs {x_col} with {hue_col} Info")
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_price_by_room(
    df: pd.DataFrame, price_col: str = "Price", room_col: str = "Room"
) -> None:
    """
    Plot a boxplot showing the distribution of house prices by number of rooms.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing housing data.
        price_col (str): Name of the column containing price values. Default is "Price".
        room_col (str): Name of the column representing the number of rooms. Default is "Room".

    Displays:
        A boxplot of prices (converted to billion Toman) grouped by number of rooms.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # show the Price on billion scale
    df = df.copy()
    df[price_col] = df[price_col] / 1e9

    sns.boxplot(x=room_col, y=price_col, data=df, ax=ax, palette="viridis")

    ax.set_xlabel("Number of Rooms")
    ax.set_ylabel("Price (Billion Toman)")
    ax.set_title("House Price Distribution by Number of Rooms")
    plt.tight_layout()
    plt.show()

    return fig


def plot_price_distribution(df: pd.DataFrame, column: str = "Price") -> None:
    """
    Plot a histogram with KDE curve for the specified price column, scaled to billion Toman.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing housing data.
        column (str): Name of the column containing price values. Default is "Price".

    Displays:
        A histogram and kernel density estimate (KDE) of the price distribution, with prices scaled to billions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # scale on billon
    scaled_prices = df[column] / 1e9

    sns.histplot(scaled_prices, kde=True, ax=ax, color="#ff7f0e", bins=30, edgecolor="black")  # type: ignore

    ax.set_title("Distribution of House Prices (in Billions)")
    ax.set_xlabel("Price (Billion Toman)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot a heatmap of the correlation matrix for all numeric columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numeric and non-numeric data.

    Displays:
        A heatmap visualizing pairwise Pearson correlation coefficients between numeric columns.
    """

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return fig


def plot_flats_by_location(df: pd.DataFrame, column: str = "Address") -> None:
    """
    Plot a horizontal bar chart showing the top 10 most frequent values in the specified location column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing housing data.
        column (str): Name of the column representing locations or addresses. Default is "Address".

    Displays:
        A horizontal bar chart of the top 10 locations with the highest number of flats.
    """

    count_series = df[column].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.barplot(
        x=count_series.values,
        y=count_series.index,
        ax=ax,
        palette="magma",
    )

    plt.xlabel("Number of Flats")
    plt.title("Top 10 Locations with Most Flats")
    plt.tight_layout()
    plt.show()

    return fig


def plot_price_distribution_by_room(
    df: pd.DataFrame, price_col: str = "Price", room_col: str = "Room"
) -> None:
    """
    Draw KDE + histogram plots of price distribution for each room count separately.

    Parameters:
        df (pd.DataFrame): Housing data DataFrame.
        price_col (str): Column name for house prices. Default is "Price".
        room_col (str): Column name for number of rooms. Default is "Room".

    Displays:
        A grid of histogram + KDE plots showing price distribution (in billion Toman) for each room category.
    """

    df = df.copy()
    df[price_col] = df[price_col] / 1e9  

    g = sns.FacetGrid(
        df, col=room_col, col_wrap=3, height=4, sharex=False, sharey=False
    )
    g.map(sns.histplot, price_col, kde=True, bins=30, color="teal", edgecolor="black")

    g.set_axis_labels("Price (Billion Toman)", "Count")
    g.set_titles("Room: {col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Distribution of House Prices by Number of Rooms", fontsize=16)
    plt.show()
    
    return g.fig


if __name__ == "__main__":
    df = data_load()

    df_clean = preprocess_data(df)

    plot_flats_by_location(df_clean)

    plot_price_vs_area(df_clean)

    plot_correlation_matrix(df_clean)

    plot_price_distribution(df_clean)

    plot_price_by_room(df_clean)

    plot_price_distribution_by_room(df_clean)
