import pandas as pd
import numpy as np
from typing import Dict, List

# Initialize lookup_dict as None
lookup_dict: Dict[str, str] = None

def _initialize_lookup():
    """Initialize the lookup dictionary if it hasn't been initialized yet."""
    global lookup_dict
    if lookup_dict is None:
        lookup_df = pd.read_excel(
            "/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx"
        )
        lookup_dict = dict(zip(lookup_df["Variable"], lookup_df["LaTeX"]))

def get_latex_label(feature_name: str) -> str:
    """
    Retrieves the LaTeX label for a given feature based on its feature group.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The corresponding LaTeX label.
    """
    # Initialize lookup dictionary if not already initialized
    _initialize_lookup()
    
    if feature_name == "UHI_diff":
        return "HW-NHW UHII"
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        "delta_": "(Δ)",
        "hw_nohw_diff_": "HW-NHW ",
        "Double_Differencing_": "(Δ)HW-NHW ",
    }
    symbol = ""
    feature_group = feature_name
    for prefix in prefix_to_symbol.keys():
        if feature_name.startswith(prefix):
            feature_group = feature_name[len(prefix) :]
            symbol = prefix_to_symbol[prefix]
            break
    # if feature_group == feature_name:
    #     feature_group += "_Level"

    # Get the LaTeX label from the lookup dictionary
    latex_label = lookup_dict.get(feature_group)

    # Use the original feature group if LaTeX label is not found
    if pd.isna(latex_label) or latex_label == "":
        latex_label = feature_group
    # Combine symbol and LaTeX label
    final_label = f"{symbol}{latex_label}".strip()
    return final_label

def replace_cold_with_continental(kg_main_group: str) -> str:
    """
    Replaces 'Cold' with 'Continental' in the given string.

    Args:
        kg_main_group (str): The input string.

    Returns:
        str: The modified string with 'Cold' replaced by 'Continental'.
    """
    if kg_main_group == "Cold":
        return "Continental"
    return kg_main_group



def get_feature_groups(feature_names: List[str]) -> Dict[str, str]:
    """
    Assign features to groups based on specified rules.

    Args:
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping from feature names to group names.
    """
    prefixes = ("delta_", "hw_nohw_diff_", "Double_Differencing_")
    feature_groups = {}
    for feature in feature_names:
        group = feature
        for prefix in prefixes:
            if feature.startswith(prefix):
                group = feature[len(prefix) :]
                break
        # If feature does not start with any prefix, it is its own group, but name the group feature + "Level"
        if group == feature:
            group = feature + "_Level"
        feature_groups[feature] = group
    return feature_groups