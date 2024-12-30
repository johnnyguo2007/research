import pandas as pd
import os
from typing import List, Dict, Tuple


def create_day_night_summary(
    group_shap_df: pd.DataFrame, all_df: pd.DataFrame, output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a summary DataFrame of day and night contributions for each feature group and KGMajorClass.

    Args:
        group_shap_df: DataFrame containing group SHAP values
        all_df: DataFrame containing all data, including 'local_hour' and 'KGMajorClass'
        output_dir: Directory to save the output summary
    """
    # Create empty lists to store results
    rows: List[Dict] = []

    # Ensure 'local_hour' and 'KGMajorClass' are available for grouping
    group_shap_df = group_shap_df.merge(
        all_df[["local_hour", "KGMajorClass"]], left_index=True, right_index=True
    )

    # Process global data first
    for period, hour_mask in [
        ("Day", lambda x: x.between(18, 19)),  # 18:00 to 19:59
        ("Night", lambda x: x.between(5, 6)),  # 5:00 to 6:59
    ]:
        # Filter data for the current period
        period_data = group_shap_df[hour_mask(group_shap_df["local_hour"])]

        # Calculate mean for each feature group
        group_means = period_data.drop(columns=["local_hour", "KGMajorClass"]).mean()

        # Add to rows with 'Global' as KGMajorClass
        rows.append(
            {
                "KGMajorClass": "Global",
                "Period": f"{period} Mean",
                **group_means.to_dict(),
                "Total": group_means.sum(),
            }
        )

    # Process each KGMajorClass
    for kg_class in group_shap_df["KGMajorClass"].unique():
        kg_data = group_shap_df[group_shap_df["KGMajorClass"] == kg_class]

        for period, hour_mask in [
            ("Day", lambda x: x.between(18, 19)),  # 18:00 to 19:59
            ("Night", lambda x: x.between(5, 6)),  # 5:00 to 6:59
        ]:
            # Filter data for the current period
            period_data = kg_data[hour_mask(kg_data["local_hour"])]

            # Calculate mean for each feature group
            group_means = period_data.drop(columns=["local_hour", "KGMajorClass"]).mean()

            # Add to rows
            rows.append(
                {
                    "KGMajorClass": kg_class,
                    "Period": f"{period} Mean",
                    **group_means.to_dict(),
                    "Total": group_means.sum(),
                }
            )

    # Create DataFrame from rows
    summary_df = pd.DataFrame(rows)

    # Round all numeric columns to 6 decimal places
    numeric_cols = summary_df.select_dtypes(include=["float64", "int64"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(6)

    # Reorder columns to put Total at the end
    cols = (
        ["KGMajorClass", "Period"]
        + [
            col
            for col in summary_df.columns
            if col not in ["KGMajorClass", "Period", "Total"]
        ]
        + ["Total"]
    )
    summary_df = summary_df[cols]

    # Custom sort order for KGMajorClass to put Global first
    kg_class_order = ["Global"] + sorted(
        [x for x in summary_df["KGMajorClass"].unique() if x != "Global" and x is not None]
    )
    summary_df["KGMajorClass"] = pd.Categorical(
        summary_df["KGMajorClass"], categories=kg_class_order, ordered=True
    )

    # Sort by KGMajorClass and Period
    summary_df = summary_df.sort_values(["KGMajorClass", "Period"])

    # Save to CSV
    output_path = os.path.join(output_dir, "feature_group_day_night_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"Day/night summary saved to {output_path}")

    return summary_df

