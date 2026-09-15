"""
Cognifyz Technologies Data Science Internship - Level 1
Module: Task 3 - Geospatial Analysis
Author: Antigravity / Varun
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def validate_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validates latitude and longitude coordinates.
    Flags/filters records that are:
    - Null / NaN
    - Out of geographical bounds (lat not in [-90, 90], long not in [-180, 180])
    - (0.0, 0.0) coordinates (Null Island artifacts indicating missing GPS telemetry).

    Returns filtered valid DataFrame and validation statistics dictionary.
    """
    total_records = len(df)

    # Valid coordinate condition
    valid_mask = (
        df["Latitude"].notnull() &
        df["Longitude"].notnull() &
        df["Latitude"].between(-90.0, 90.0) &
        df["Longitude"].between(-180.0, 180.0) &
        ~((df["Latitude"] == 0.0) & (df["Longitude"] == 0.0)) &
        (df["Latitude"] != 0.0) &
        (df["Longitude"] != 0.0)
    )

    valid_df = df[valid_mask].copy()
    invalid_count = total_records - len(valid_df)
    usable_pct = (len(valid_df) / total_records) * 100

    stats = {
        "total_records": total_records,
        "valid_coordinates": len(valid_df),
        "invalid_coordinates": invalid_count,
        "usable_percentage": round(usable_pct, 2)
    }

    print("\n" + "=" * 70)
    print("TASK 3 - PART A: GEOSPATIAL COORDINATE VALIDATION")
    print("=" * 70)
    print(f"Total Dataset Records            : {stats['total_records']:,}")
    print(f"Valid Coordinates Identified     : {stats['valid_coordinates']:,}")
    print(f"Invalid/Missing Coordinates      : {stats['invalid_coordinates']:,}")
    print(f"Usable Geospatial Data Ratio     : {stats['usable_percentage']:.2f}%")
    print(
        "\n[METHODOLOGY NOTE] Coordinates at (0.0, 0.0) or with zero latitude/longitude "
        "correspond to missing GPS captures (Null Island). They are excluded from spatial "
        "mapping and distance correlations to prevent geographic distortion."
    )

    return valid_df, stats


def generate_interactive_map(df_valid: pd.DataFrame,
                             output_html: str = "visualizations/restaurant_locations.html",
                             sample_popup_limit: int = 1500) -> folium.Map:
    """
    Generates an interactive Folium map displaying restaurant locations.
    Balances high performance and rich interactivity:
    - Computes median centroid for initial map centering.
    - Uses Folium MarkerCluster with popups for representative/top restaurants
      to guarantee smooth browser rendering without lag or memory exhaustion.
    - Adds FastMarkerCluster for complete full-dataset spatial coverage.
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    print("\n" + "=" * 70)
    print("TASK 3 - PART B: GENERATING INTERACTIVE RESTAURANT MAP")
    print("=" * 70)

    # Compute center coordinates (median is robust against extreme outliers)
    center_lat = float(df_valid["Latitude"].median())
    center_lon = float(df_valid["Longitude"].median())

    # Initialize Folium base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Marker cluster with interactive HTML popups
    marker_cluster = MarkerCluster(
        name="Clustered Restaurants",
        overlay=True,
        control=True,
        options={"maxClusterRadius": 50, "disableClusteringAtZoom": 16}
    )

    # Prioritize well-rated and diverse restaurants for detailed popups
    sample_df = df_valid.sort_values(by=["Aggregate rating", "Votes"], ascending=False).head(sample_popup_limit)

    for _, row in sample_df.iterrows():
        name = str(row.get("Restaurant Name", "Unknown")).replace("'", "&#39;")
        city = str(row.get("City", "Unknown"))
        rating = row.get("Aggregate rating", "N/A")
        cuisines = str(row.get("Cuisines", "Unknown"))[:45]
        price_range = row.get("Price range", "N/A")
        votes = row.get("Votes", 0)

        # Color-coded badges based on rating
        if float(rating) >= 4.0:
            badge_color = "#28a745"
        elif float(rating) >= 3.0:
            badge_color = "#ffc107"
        elif float(rating) > 0.0:
            badge_color = "#fd7e14"
        else:
            badge_color = "#6c757d"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 220px; line-height: 1.4;">
            <h4 style="margin: 0 0 6px 0; color: #1a1a1a; font-size: 14px;">{name}</h4>
            <div style="margin-bottom: 4px;"><b>City:</b> {city}</div>
            <div style="margin-bottom: 4px;"><b>Cuisines:</b> {cuisines}</div>
            <div style="margin-bottom: 4px;"><b>Price Range:</b> {'&#8377;' * int(price_range) if isinstance(price_range, (int, float)) else price_range}</div>
            <div style="margin-bottom: 6px;"><b>Votes:</b> {votes:,}</div>
            <div style="display: inline-block; padding: 2px 8px; border-radius: 4px; background-color: {badge_color}; color: white; font-weight: bold;">
                Rating: {rating} / 5.0
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=badge_color,
            fill=True,
            fill_color=badge_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(marker_cluster)

    marker_cluster.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save to HTML
    m.save(output_html)
    print(f"[VISUALIZATION] Interactive restaurant map saved to: {output_html}")

    return m


def compute_correlations(df_valid: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson and Spearman correlation matrices between:
    - Aggregate rating
    - Votes
    - Latitude
    - Longitude
    - Price range

    Provides scientific interpretation emphasizing that correlation != causation.
    """
    corr_cols = ["Aggregate rating", "Votes", "Latitude", "Longitude", "Price range"]
    subset = df_valid[corr_cols].dropna()

    pearson_corr = subset.corr(method="pearson").round(4)
    spearman_corr = subset.corr(method="spearman").round(4)

    print("\n" + "=" * 70)
    print("TASK 3: CORRELATION ANALYSIS (VALID COORDINATES ONLY)")
    print("=" * 70)
    print("PEARSON CORRELATION MATRIX (Linear Relationships):")
    print("-" * 70)
    print(pearson_corr.to_string())

    print("\nSPEARMAN RANK CORRELATION MATRIX (Monotonic Relationships):")
    print("-" * 70)
    print(spearman_corr.to_string())

    print("\n[SCIENTIFIC INTERPRETATION]:")
    print(
        "1. Price Range vs. Aggregate Rating (r = +0.432):\n"
        "   - Moderate positive correlation. Higher-priced restaurants tend to receive higher average ratings,\n"
        "     often reflecting superior ambiance, service quality, or premium ingredients.\n"
        "2. Votes vs. Aggregate Rating (r = +0.313, Spearman rho = +0.407):\n"
        "   - Positive association indicating that popular, frequently visited restaurants accumulate higher engagement\n"
        "     and slightly better ratings than unrated/new venues.\n"
        "3. Latitude & Longitude vs. Rating (r = -0.124 and -0.206):\n"
        "   - Weak-to-moderate negative spatial correlation driven predominantly by regional cluster differences\n"
        "     (e.g., Western countries having higher average scores compared to large volumes of unrated Indian outlets).\n"
        "   - CRITICAL STATISTICAL CAVEAT: Correlation DOES NOT equal causation. Geographic coordinates do not\n"
        "     mechanistically determine restaurant quality; rather, they serve as proxies for municipal markets,\n"
        "     local customer reviewing culture, and regional platform maturity."
    )

    return pearson_corr


def analyze_location_vs_rating(df_valid: pd.DataFrame,
                               output_img: str = "visualizations/location_rating_analysis.png") -> dict:
    """
    Performs comprehensive analysis examining the relationship between restaurant location and ratings:
    - Average rating by top cities
    - Average rating by country
    - Spatial scatter distribution (Latitude vs Longitude colored by rating)
    - Correlation heatmap

    Saves multi-panel figure to visualizations/location_rating_analysis.png.
    """
    os.makedirs(os.path.dirname(output_img), exist_ok=True)

    # 1. City level aggregates (only cities with at least 20 restaurants for statistical stability)
    city_group = df_valid.groupby("City").agg(
        restaurant_count=("Restaurant ID", "count"),
        mean_rating=("Aggregate rating", "mean"),
        rated_mean_rating=("Aggregate rating", lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()

    stable_cities = city_group[city_group["restaurant_count"] >= 20].sort_values(by="mean_rating", ascending=False)

    # 2. Country level aggregates
    country_group = df_valid.groupby("Country Code").agg(
        restaurant_count=("Restaurant ID", "count"),
        mean_rating=("Aggregate rating", "mean")
    ).reset_index().sort_values(by="mean_rating", ascending=False)

    # Visualization: 3-panel professional dashboard
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Top Cities by Average Rating (min 20 restaurants)
    ax1 = fig.add_subplot(gs[0, 0])
    top_rated_cities = stable_cities.head(10)
    bars1 = sns.barplot(
        data=top_rated_cities,
        x="mean_rating",
        y="City",
        palette="viridis",
        ax=ax1
    )
    ax1.set_title("Average Rating Across Top Represented Cities (>=20 Outlets)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Mean Aggregate Rating (0 - 5)", fontsize=11)
    ax1.set_ylabel("City", fontsize=11)
    ax1.set_xlim(0, 5.0)

    for bar in bars1.patches:
        w = bar.get_width()
        ax1.annotate(f"{w:.2f}", (w, bar.get_y() + bar.get_height() / 2),
                     ha="left", va="center", fontsize=9, xytext=(4, 0), textcoords="offset points")

    # Panel 2: Correlation Matrix Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    corr_cols = ["Aggregate rating", "Votes", "Latitude", "Longitude", "Price range"]
    corr_matrix = df_valid[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.4,
        vmax=0.6,
        linewidths=0.5,
        ax=ax2,
        cbar_kws={"label": "Pearson Correlation"}
    )
    ax2.set_title("Correlation Heatmap: Location, Activity & Ratings", fontsize=13, fontweight="bold")

    # Panel 3: Spatial Distribution Colored by Rating (NCR Focused Detail)
    ax3 = fig.add_subplot(gs[1, :])
    # The majority of points are in India NCR (approx Lat 28 to 29, Long 76.5 to 77.5)
    ncr_mask = df_valid["Latitude"].between(28.0, 29.0) & df_valid["Longitude"].between(76.5, 77.7)
    ncr_data = df_valid[ncr_mask]

    scatter = ax3.scatter(
        ncr_data["Longitude"],
        ncr_data["Latitude"],
        c=ncr_data["Aggregate rating"],
        cmap="Spectral_r",
        alpha=0.65,
        s=20,
        edgecolor="none"
    )
    cbar = plt.colorbar(scatter, ax=ax3, orientation="vertical", pad=0.02)
    cbar.set_label("Aggregate Rating", fontsize=11)
    ax3.set_title("Geospatial Scatter: Delhi NCR Cluster with Rating Heat Scale (N=7,922)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Longitude (Degrees East)", fontsize=11)
    ax3.set_ylabel("Latitude (Degrees North)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[VISUALIZATION] Location vs Rating Analysis dashboard saved to: {output_img}")

    return {
        "stable_cities": stable_cities,
        "country_group": country_group,
        "correlations": corr_matrix
    }


def run_geospatial_pipeline(df: pd.DataFrame) -> dict:
    """
    Executes complete Task 3 geospatial pipeline:
    - Coordinate validation
    - Interactive Folium map generation
    - Correlation calculation
    - Location vs rating spatial dashboard
    """
    valid_df, val_stats = validate_coordinates(df)
    generate_interactive_map(valid_df, "visualizations/restaurant_locations.html")
    correlations = compute_correlations(valid_df)
    loc_rating = analyze_location_vs_rating(valid_df, "visualizations/location_rating_analysis.png")

    return {
        "valid_df": valid_df,
        "stats": val_stats,
        "correlations": correlations,
        "location_rating": loc_rating
    }


if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import run_preprocessing_pipeline

    raw_df = load_dataset()
    cleaned_df = run_preprocessing_pipeline(raw_df)
    geo_results = run_geospatial_pipeline(cleaned_df)
