import pandas as pd
from typing import Dict, List, Optional, Tuple


class GroupData:
    """
    A class to encapsulate group data with a single DataFrame and associated group information.

    Attributes:
        df (pd.DataFrame): DataFrame containing group SHAP values, feature values, and metadata
        group_names (List[str]): List of unique group names
        group_shap_cols (List[str]): List of group SHAP value column names
        group_feature_cols (List[str]): List of group feature value column names
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_names: List[str],
        group_shap_cols: List[str],
        group_feature_cols: List[str],
    ):
        """
        Initialize GroupData with a DataFrame and group information.

        Args:
            df (pd.DataFrame): DataFrame containing group data
            group_names (List[str]): List of unique group names
            group_shap_cols (List[str]): List of group SHAP value column names
            group_feature_cols (List[str]): List of group feature value column names
        """
        self.df = df
        self.group_names = group_names
        self.group_shap_cols = group_shap_cols
        self.group_feature_cols = group_feature_cols

    @classmethod
    def from_all_df_include_kg_local_hour(
        cls,
        both_shap_and_feature_df: pd.DataFrame,
        feature_to_group_mapping: Dict[str, str],
    ) -> "GroupData":
        """
        Create GroupData from a feature DataFrame and feature groups mapping.

        Args:
            both_shap_and_feature_df (pd.DataFrame): DataFrame containing both SHAP and feature values
            feature_groups (dict): Mapping from features to their groups

        Returns:
            GroupData: New instance with processed group data
        """
        # Get unique group names
        group_names: List[str] = list(set(feature_to_group_mapping.values()))

        # Initialize empty DataFrame to store results
        group_all_df = pd.DataFrame()

        # Add KGMajorClass and local_hour if they exist in the input DataFrame
        group_all_df["KGMajorClass"] = both_shap_and_feature_df["KGMajorClass"]

        group_all_df["local_hour"] = both_shap_and_feature_df["local_hour"]

        group_shap_cols = []
        group_feature_cols = []

        for group in group_names:
            # Get features in this group
            group_features: List[str] = [
                f for f, g in feature_to_group_mapping.items() if g == group
            ]

            # Get corresponding SHAP and feature columns
            group_shap_col = f"{group}_shap"
            group_feature_col = f"{group}_feature"

            # Sum SHAP values for features in the group
            group_shap_cols_temp = [f"{f}_shap" for f in group_features]
            group_all_df[group_shap_col] = both_shap_and_feature_df[
                group_shap_cols_temp
            ].sum(axis=1)

            # Sum feature values for the group
            group_all_df[group_feature_col] = both_shap_and_feature_df[
                group_features
            ].sum(axis=1)

            group_shap_cols.append(group_shap_col)
            group_feature_cols.append(group_feature_col)

        return cls(
            df=group_all_df.reset_index(drop=True),
            group_names=group_names,
            group_shap_cols=group_shap_cols,
            group_feature_cols=group_feature_cols,
        )

    def get_shap_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Get DataFrame with only SHAP columns, optionally filtered by KGMajorClass.

        Args:
            kg_class (str, optional): KGMajorClass to filter by

        Returns:
            pd.DataFrame: DataFrame containing only SHAP columns
        """
        df = self.df
        if kg_class and kg_class != "global" and "KGMajorClass" in df.columns:
            df = df[df["KGMajorClass"] == kg_class]
        return df[self.group_shap_cols]

    def get_feature_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Get DataFrame with only feature columns, optionally filtered by KGMajorClass.

        Args:
            kg_class (str, optional): KGMajorClass to filter by

        Returns:
            pd.DataFrame: DataFrame containing only feature columns
        """
        df = self.df
        if kg_class and kg_class != "global" and "KGMajorClass" in df.columns:
            df = df[df["KGMajorClass"] == kg_class]
        return df[self.group_feature_cols]

    def get_group_data(
        self, group_name: str, kg_class: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get SHAP and feature DataFrames for a specific group.

        Args:
            group_name (str): Name of the group to get data for
            kg_class (str, optional): KGMajorClass to filter by

        Returns:
            tuple: (shap_df, feature_df) for the specified group
        """
        df = self.df
        if kg_class and kg_class != "global" and "KGMajorClass" in df.columns:
            df = df[df["KGMajorClass"] == kg_class]

        idx = self.group_names.index(group_name)
        shap_col = self.group_shap_cols[idx]
        feature_col = self.group_feature_cols[idx]

        return df[[shap_col]], df[[feature_col]]

    def get_mean_hourly_shap_df(
        self, df_feature: pd.DataFrame, kg_class: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepares feature group data for plotting.

        Args:
            df_feature (pd.DataFrame): DataFrame containing feature data
            kg_class (str, optional): Climate zone to filter by. If None or 'global',
                                    returns data grouped by local_hour and KGMajorClass.

        Returns:
            pd.DataFrame: Processed feature group data grouped by local_hour and KGMajorClass
        """
        df = self.df.copy()
        
        if not kg_class or kg_class == "global":
            return df.groupby(["local_hour"]).mean().reset_index()
        
        df = df[df["KGMajorClass"] == kg_class]
        df = df.drop("KGMajorClass", axis=1)
        return df.groupby("local_hour").mean().reset_index()


def calculate_group_shap_values(
    both_shap_and_feature_df: pd.DataFrame,
    feature_groups: Dict[str, str],
) -> GroupData:
    """
    Calculate group-level SHAP values and feature values, returning a GroupData instance.

    Args:
        both_shap_and_feature_df (pd.DataFrame): DataFrame containing both SHAP and feature values
        feature_groups (dict): Mapping from features to their groups

    Returns:
        GroupData: Instance containing the processed group data
    """
    return GroupData.from_all_df_include_kg_local_hour(
        both_shap_and_feature_df, feature_groups
    )
