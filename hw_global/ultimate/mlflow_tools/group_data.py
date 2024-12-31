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
        shap_col_names (List[str]): List of SHAP value column names
        feature_cols_names (List[str]): List of feature value column names
        shap_group_col_names (List[str]): List of group SHAP value column names
        feature_group_cols_names (List[str]): List of group feature value column names
        shap_detail_df (pd.DataFrame): DataFrame containing SHAP values
        feature_detail_df (pd.DataFrame): DataFrame containing feature values
        shap_group_detail_df (pd.DataFrame): DataFrame containing group SHAP values
        feature_group_detail_df (pd.DataFrame): DataFrame containing group feature values
        _shap_hourly_mean_cache (Dict[Optional[str], pd.DataFrame]): Cache for shap_hourly_mean_df results
        _feature_hourly_mean_cache (Dict[Optional[str], pd.DataFrame]): Cache for feature_hourly_mean_df results
        _shap_group_hourly_mean_cache (Dict[Optional[str], pd.DataFrame]): Cache for shap_group_hourly_mean_df results
        _feature_group_hourly_mean_cache (Dict[Optional[str], pd.DataFrame]): Cache for feature_group_hourly_mean_df results
    """

    def __init__(
        self,
        both_shap_and_feature_df: pd.DataFrame,
        feature_to_group_mapping: Dict[str, str],
    ):
        """
        Initialize GroupData with a DataFrame and feature-to-group mapping.

        Args:
            both_shap_and_feature_df (pd.DataFrame): DataFrame containing both SHAP and feature values
            feature_to_group_mapping (Dict[str, str]): Mapping from features to their groups
        """
        if not all(col in both_shap_and_feature_df.columns for col in ["local_hour", "KGMajorClass"]):
            raise ValueError("DataFrame must contain 'local_hour' and 'KGMajorClass' columns.")
        
        self.df, self.group_names, self.group_shap_cols, self.group_feature_cols = self._process_data(
            both_shap_and_feature_df, feature_to_group_mapping
        )
        self.shap_col_names = [col for col in both_shap_and_feature_df.columns if "_shap" in col and "group" not in col]
        self.feature_cols_names = [col for col in both_shap_and_feature_df.columns if "_shap" not in col and col not in ["KGMajorClass", "local_hour", "KGClass", "global_event_ID", "lon", "lat", "time", "UHI_diff", "Estimation_Error", "base_value"]]
        self.shap_group_col_names = self.group_shap_cols
        self.feature_group_cols_names = self.group_feature_cols

        self.shap_detail_df = both_shap_and_feature_df[self.shap_col_names]
        self.feature_detail_df = both_shap_and_feature_df[self.feature_cols_names]

        self.shap_group_detail_df = self.df[self.shap_group_col_names]
        self.feature_group_detail_df = self.df[self.feature_group_cols_names]

        self._shap_hourly_mean_cache: Dict[Optional[str], pd.DataFrame] = {}
        self._feature_hourly_mean_cache: Dict[Optional[str], pd.DataFrame] = {}
        self._shap_group_hourly_mean_cache: Dict[Optional[str], pd.DataFrame] = {}
        self._feature_group_hourly_mean_cache: Dict[Optional[str], pd.DataFrame] = {}

    def _process_data(
        self,
        both_shap_and_feature_df: pd.DataFrame,
        feature_to_group_mapping: Dict[str, str],
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
        """
        Helper function to process data and create group-level SHAP and feature values.

        Args:
            both_shap_and_feature_df (pd.DataFrame): DataFrame containing both SHAP and feature values
            feature_to_group_mapping (Dict[str, str]): Mapping from features to their groups

        Returns:
            Tuple[pd.DataFrame, List[str], List[str], List[str]]: Processed DataFrame, group names, group SHAP columns, group feature columns
        """
        group_names: List[str] = list(set(feature_to_group_mapping.values()))
        group_all_df = pd.DataFrame()

        group_all_df["KGMajorClass"] = both_shap_and_feature_df["KGMajorClass"]
        group_all_df["local_hour"] = both_shap_and_feature_df["local_hour"]

        group_shap_cols = []
        group_feature_cols = []

        for group in group_names:
            group_features: List[str] = [
                f for f, g in feature_to_group_mapping.items() if g == group
            ]
            group_shap_col = f"{group}_shap"
            group_feature_col = f"{group}_feature"

            group_shap_cols_temp = [f"{f}_shap" for f in group_features]
            group_all_df[group_shap_col] = both_shap_and_feature_df[
                group_shap_cols_temp
            ].sum(axis=1)

            group_all_df[group_feature_col] = both_shap_and_feature_df[
                group_features
            ].sum(axis=1)

            group_shap_cols.append(group_shap_col)
            group_feature_cols.append(group_feature_col)

        return group_all_df, group_names, group_shap_cols, group_feature_cols

    def shap_hourly_mean_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the hourly mean of SHAP values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of SHAP values.
        """
        if kg_class not in self._shap_hourly_mean_cache:
            df = pd.concat([self.shap_detail_df, self.df[["local_hour", "KGMajorClass"]]], axis=1)

            if not kg_class or kg_class == "global":
                result = df.drop("KGMajorClass", axis=1).groupby(["local_hour"]).mean().reset_index()
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()
            
            self._shap_hourly_mean_cache[kg_class] = result

        return self._shap_hourly_mean_cache[kg_class]

    def feature_hourly_mean_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the hourly mean of feature values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of feature values.
        """
        if kg_class not in self._feature_hourly_mean_cache:
            df = pd.concat([self.feature_detail_df, self.df[["local_hour", "KGMajorClass"]]], axis=1)
            if not kg_class or kg_class == "global":
                result = df.drop("KGMajorClass", axis=1).groupby(["local_hour"]).mean().reset_index()
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

            self._feature_hourly_mean_cache[kg_class] = result
        
        return self._feature_hourly_mean_cache[kg_class]

    def shap_group_hourly_mean_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the hourly mean of group SHAP values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of group SHAP values.
        """
        if kg_class not in self._shap_group_hourly_mean_cache:
            df = self.df.copy()
            if not kg_class or kg_class == "global":
                result = df.drop("KGMajorClass", axis=1).groupby(["local_hour"]).mean().reset_index()
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

            self._shap_group_hourly_mean_cache[kg_class] = result

        return self._shap_group_hourly_mean_cache[kg_class]

    def feature_group_hourly_mean_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the hourly mean of group feature values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of group feature values.
        """
        if kg_class not in self._feature_group_hourly_mean_cache:
            df = self.df.copy()
            if not kg_class or kg_class == "global":
                result = df.drop("KGMajorClass", axis=1).groupby(["local_hour"]).mean().reset_index()
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

            self._feature_group_hourly_mean_cache[kg_class] = result

        return self._feature_group_hourly_mean_cache[kg_class]

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
    return GroupData(both_shap_and_feature_df, feature_groups)