import pandas as pd
import numpy as np
from typing import Dict, List

# Initialize lookup dictionaries as None
lookup_dict: Dict[str, str] = None
unit_dict: Dict[str, str] = None
long_name_dict: Dict[str, str] = None

def _initialize_lookup():
    """Initialize the lookup dictionaries if they haven't been initialized yet."""
    global lookup_dict, unit_dict, long_name_dict
    if lookup_dict is None:
        lookup_df = pd.read_excel(
            "/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx"
        )
        lookup_dict = dict(zip(lookup_df["Variable"], lookup_df["LaTeX"]))
        unit_dict = dict(zip(lookup_df["Variable"], lookup_df["Units"]))
        long_name_dict = dict(zip(lookup_df["Variable"], lookup_df["Long Name"]))

def get_unit(feature_name: str) -> str:
    """
    Retrieves the unit for a given feature based on its feature group.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The corresponding unit.
    """
    # Initialize lookup dictionary if not already initialized
    _initialize_lookup()
    
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        "delta_": "(δ)",
        "hw_nohw_diff_": "Δ",
        "Double_Differencing_": "Δδ",
    }
    
    feature_group = feature_name
    for prefix in prefix_to_symbol.keys():
        if feature_name.startswith(prefix):
            feature_group = feature_name[len(prefix):]
            break

    # Get the unit from the unit dictionary
    unit = unit_dict.get(feature_group)
    
    # Return empty string if no unit found
    if pd.isna(unit) or unit == "":
        return ""
    return unit

def get_long_name_without_unit(feature_name: str) -> str:
    """
    Gets the long name without unit for a feature. Used specifically for plot titles.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The long name without unit.
    """
    # Initialize lookup dictionary if not already initialized
    _initialize_lookup()
    
    # Strip "extbf_" prefix if it exists
    if feature_name.startswith("extbf_"):
        feature_name = feature_name[len("extbf_"):]
    
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        "delta_": "Difference of ",
        "hw_nohw_diff_": "Δ Difference of ",
        "Double_Differencing_": "Δδ Difference of ",
    }
    
    symbol = ""
    feature_group = feature_name
    for prefix in prefix_to_symbol.keys():
        if feature_name.startswith(prefix):
            feature_group = feature_name[len(prefix):]
            symbol = prefix_to_symbol[prefix]
            break

    # Get the long name from the lookup dictionary
    long_name = long_name_dict.get(feature_group)
    
    # Use the original feature group if long name is not found
    if pd.isna(long_name) or long_name == "":
        long_name = feature_group
        
    # Combine symbol and long name
    final_label = f"{symbol}{long_name}"
    final_label = final_label.strip()
    if final_label.startswith(r"\textbf{") and final_label.endswith("}"):
        return fr"\mbox{{{final_label}}}"
    return final_label

def get_label_with_unit(feature_name: str) -> str:
    """
    Gets the LaTeX label with unit for a feature.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The LaTeX label with unit.
    """
    label = get_latex_label(feature_name)
    unit = get_unit(feature_name)
    if unit:
        return f"{label} ({unit})"
    return label

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
    
    # Strip "extbf_" prefix if it exists
    if feature_name.startswith("extbf_"):
        feature_name = feature_name[len("extbf_"):]
    
    if feature_name == "UHI_diff":
        return "Δ UHII"
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        "delta_": "(δ)",
        "hw_nohw_diff_": "Δ",
        "Double_Differencing_": "Δδ",
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
    # If final_label seems to be a \textbf command, wrap it in \mbox for robustness
    if final_label.startswith(r"\textbf{") and final_label.endswith("}"):
        return fr"\mbox{{{final_label}}}"
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

# Define consistent colors for feature groups
FEATURE_COLORS = {
    "FGR":            '#a65628', # Brown
    "FIRA":           '#e41a1c', # Red
    "FSA":            '#984ea3', # Purple
    "FSH":            '#ff7f00', # Orange
    "HEAT_FROM_AC":   '#999999', # Gray / Silver
    "Q2M":            '#377eb8', # Blue
    "Qstor":          '#008080', # Teal
    "SOILWATER_10CM": '#f781bf', # Pink
    "U10":            '#4daf4a', # Green
}

def replace_cold_with_continental(s: str) -> str:
    """
    Replaces 'Cold' with 'Continental' in the given string.

    Args:
        s (str): The input string.

    Returns:
        str: The modified string with 'Cold' replaced by 'Continental'.
    """
    if s == "Cold":
        return "Continental"
    return s