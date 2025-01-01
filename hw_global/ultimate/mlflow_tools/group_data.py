import pandas as pd
from typing import Dict, List, Optional, Tuple

class GroupData:
    """
    A class to encapsulate group data with a single DataFrame and associated group information.

    Attributes:
        df (pd.DataFrame): DataFrame containing group SHAP values, feature values, and metadata, with original column names.
    """

    def __init__(
        self,
        both_shap_and_feature_df: pd.DataFrame,
        feature_to_group_mapping: Dict[str, str],
    ):
        """
        Initialize GroupData with a DataFrame and feature-to-group mapping.

        Args:
            both_shap_and_feature_df (pd.DataFrame): DataFrame containing both SHAP and feature values with original column names.
            feature_to_group_mapping (Dict[str, str]): Mapping from features to their groups
        """
        if not all(
            col in both_shap_and_feature_df.columns
            for col in ["local_hour", "KGMajorClass"]
        ):
            raise ValueError(
                "DataFrame must contain 'local_hour' and 'KGMajorClass' columns."
            )

        self._df, self._group_names, self._group_shap_cols, self._group_feature_cols = (
            self._process_data(both_shap_and_feature_df, feature_to_group_mapping)
        )

        self._shap_col_names = [
            col
            for col in both_shap_and_feature_df.columns
            if col.endswith("_shap") and "group" not in col
        ]
        self._feature_cols_names = [
            col.replace("_shap", "") for col in self._shap_col_names
        ]
        
        self._shap_detail_df = both_shap_and_feature_df[self._shap_col_names]
        self._feature_detail_df = both_shap_and_feature_df[self._feature_cols_names]
        
        self._shap_group_detail_df = self._df[self._group_shap_cols]
        self._feature_group_detail_df = self._df[self._group_feature_cols]
        
        # Caches for DataFrames with modified column names
        self._processed_df_cache: Optional[pd.DataFrame] = None
        self._shap_detail_df_processed_cache: Optional[pd.DataFrame] = None
        self._feature_detail_df_processed_cache: Optional[pd.DataFrame] = None
        self._shap_group_detail_df_processed_cache: Optional[pd.DataFrame] = None
        self._feature_group_detail_df_processed_cache: Optional[pd.DataFrame] = None

        # Caches for hourly mean calculations
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
            group_shap_col = f"{group}_group_shap"
            group_feature_col = f"{group}_group_feature"

            group_shap_cols_temp = [f"{f}_shap" for f in group_features]
            group_all_df[group_shap_col] = both_shap_and_feature_df[
                group_shap_cols_temp
            ].sum(axis=1)

            group_feature_cols_temp = [f"{f}" for f in group_features]
            group_all_df[group_feature_col] = both_shap_and_feature_df[
                group_feature_cols_temp
            ].sum(axis=1)

            group_shap_cols.append(group_shap_col)
            group_feature_cols.append(group_feature_col)

        return group_all_df, group_names, group_shap_cols, group_feature_cols
    
    def _get_processed_df(self) -> pd.DataFrame:
        """Returns the processed df with modified column names, using cache if available."""
        if self._processed_df_cache is None:
            self._processed_df_cache = self._df.rename(columns={col: col.replace("_shap", "").replace("_feature", "").replace("_group", "") for col in self._df.columns})
        return self._processed_df_cache

    def _get_shap_detail_df_processed(self) -> pd.DataFrame:
        """Returns the processed shap_detail_df with modified column names, using cache if available."""
        if self._shap_detail_df_processed_cache is None:
            self._shap_detail_df_processed_cache = self._shap_detail_df.rename(columns={col: col.replace("_shap", "") for col in self._shap_detail_df.columns})
        return self._shap_detail_df_processed_cache

    def _get_feature_detail_df_processed(self) -> pd.DataFrame:
        """Returns the processed feature_detail_df with modified column names, using cache if available."""
        if self._feature_detail_df_processed_cache is None:
            self._feature_detail_df_processed_cache = self._feature_detail_df.copy() # No renaming needed as _shap is already removed
        return self._feature_detail_df_processed_cache

    def _get_shap_group_detail_df_processed(self) -> pd.DataFrame:
        """Returns the processed shap_group_detail_df with modified column names, using cache if available."""
        if self._shap_group_detail_df_processed_cache is None:
            self._shap_group_detail_df_processed_cache = self._shap_group_detail_df.rename(columns={col: col.replace("_shap", "").replace("_group", "") for col in self._shap_group_detail_df.columns})
        return self._shap_group_detail_df_processed_cache

    def _get_feature_group_detail_df_processed(self) -> pd.DataFrame:
        """Returns the processed feature_group_detail_df with modified column names, using cache if available."""
        if self._feature_group_detail_df_processed_cache is None:
            self._feature_group_detail_df_processed_cache = self._feature_group_detail_df.rename(columns={col: col.replace("_feature", "").replace("_group", "") for col in self._feature_group_detail_df.columns})
        return self._feature_group_detail_df_processed_cache

    # @property
    # def df(self) -> pd.DataFrame:
    #     """DataFrame containing group SHAP values, feature values, and metadata with modified column names."""
    #     return self._get_processed_df()

    @property
    def group_names(self) -> List[str]:
        """List of unique group names."""
        return self._group_names

    @property
    def group_shap_cols(self) -> List[str]:
        """List of group SHAP value column names with modified names."""
        return [col.replace("_shap", "").replace("_group", "") for col in self._group_shap_cols]

    @property
    def group_feature_cols(self) -> List[str]:
        """List of group feature value column names with modified names."""
        return [col.replace("_feature", "").replace("_group", "") for col in self._group_feature_cols]
    
    @property
    def shap_col_names(self) -> List[str]:
        """List of SHAP value column names."""
        return [col.replace("_shap", "") for col in self._shap_col_names]

    @property
    def feature_cols_names(self) -> List[str]:
        """List of feature value column names."""
        return self._feature_cols_names

    @property
    def shap_group_col_names(self) -> List[str]:
        """List of group SHAP value column names with modified names."""
        return self.group_shap_cols

    @property
    def feature_group_cols_names(self) -> List[str]:
        """List of group feature value column names with modified names."""
        return self.group_feature_cols

    @property
    def shap_detail_df(self) -> pd.DataFrame:
        """DataFrame containing SHAP values with modified column names."""
        return self._get_shap_detail_df_processed()

    @property
    def feature_detail_df(self) -> pd.DataFrame:
        """DataFrame containing feature values with modified column names."""
        return self._get_feature_detail_df_processed()

    @property
    def shap_group_detail_df(self) -> pd.DataFrame:
        """DataFrame containing group SHAP values with modified column names."""
        return self._get_shap_group_detail_df_processed()

    @property
    def feature_group_detail_df(self) -> pd.DataFrame:
        """DataFrame containing group feature values with modified column names."""
        return self._get_feature_group_detail_df_processed()

    def shap_hourly_mean_df(self, kg_class: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the hourly mean of SHAP values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of SHAP values.
        """
        if kg_class not in self._shap_hourly_mean_cache:
            df = pd.concat(
                [self._shap_detail_df, self._df[["local_hour", "KGMajorClass"]]], axis=1
            )

            if not kg_class or kg_class == "global":
                result = (
                    df.drop("KGMajorClass", axis=1)
                    .groupby(["local_hour"])
                    .mean()
                    .reset_index()
                )
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

            # Rename columns before caching
            result = result.rename(columns={col: col.replace("_shap", "") for col in result.columns})
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
            df = pd.concat(
                [self._feature_detail_df, self._df[["local_hour", "KGMajorClass"]]],
                axis=1,
            )
            if not kg_class or kg_class == "global":
                result = (
                    df.drop("KGMajorClass", axis=1)
                    .groupby(["local_hour"])
                    .mean()
                    .reset_index()
                )
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

             # Rename columns before caching
            result = result.rename(columns={col: col.replace("_feature", "") for col in result.columns})
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
            df = self._df.copy()
            if not kg_class or kg_class == "global":
                result = (
                    df.drop("KGMajorClass", axis=1)
                    .groupby(["local_hour"])
                    .mean()
                    .reset_index()
                )
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()
            
            # Rename columns before caching
            result = result.rename(columns={col: col.replace("_shap", "").replace("_group", "") for col in result.columns})
            self._shap_group_hourly_mean_cache[kg_class] = result

        return self._shap_group_hourly_mean_cache[kg_class]

    def feature_group_hourly_mean_df(
        self, kg_class: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculates the hourly mean of group feature values.

        Args:
            kg_class (str, optional): KGMajorClass to filter by.

        Returns:
            pd.DataFrame: Hourly mean of group feature values.
        """
        if kg_class not in self._feature_group_hourly_mean_cache:
            df = self._df.copy()
            if not kg_class or kg_class == "global":
                result = (
                    df.drop("KGMajorClass", axis=1)
                    .groupby(["local_hour"])
                    .mean()
                    .reset_index()
                )
            else:
                df = df[df["KGMajorClass"] == kg_class]
                df = df.drop("KGMajorClass", axis=1)
                result = df.groupby("local_hour").mean().reset_index()

            # Rename columns before caching
            result = result.rename(columns={col: col.replace("_feature", "").replace("_group", "") for col in result.columns})
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