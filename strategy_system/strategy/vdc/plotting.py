import matplotlib.pyplot as plt
import pandas as pd


def plot_leveraged_pair(
    df: pd.DataFrame,
    price_col: str = "price_close",
) -> None:
    """
    Plot underlying price and synthetic leveraged
    long/short pair cumulative return.
    """

    required = [
        price_col,
        "leveraged_pair_cum_return",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"missing required columns: {missing}"
        )

    plot_df = df.sort_index()

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    # -------------------------------------------------------
    # BTC price
    # -------------------------------------------------------

    ax1.plot(
        plot_df.index,
        plot_df[price_col],
        label="BTC Price",
        linewidth=1.2,
    )

    ax1.set_xlabel(
        "Time"
    )

    ax1.set_ylabel(
        "BTC Price"
    )

    # -------------------------------------------------------
    # Leveraged pair cumulative return
    # -------------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        plot_df.index,
        plot_df["leveraged_pair_cum_return"] * 100.0,
        label="3x Long + 3x Short Cumulative Return",
        linewidth=1.2,
    )

    ax2.axhline(
        0.0,
        linewidth=0.8,
        linestyle=":",
    )

    ax2.set_ylabel(
        "Cumulative Return (%)"
    )

    # -------------------------------------------------------
    # Legend
    # -------------------------------------------------------

    lines_1, labels_1 = (
        ax1.get_legend_handles_labels()
    )

    lines_2, labels_2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    ax1.set_title(
        "BTC Price vs Synthetic Leveraged Pair"
    )

    fig.tight_layout()

    plt.show()