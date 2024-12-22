import mlflow
import pandas as pd
import os
import re
import shap
import matplotlib.pyplot as plt
import numpy as np
from ultimate.paper_figure_feature_contribution_by_hour import get_feature_groups

class Experiment:
    def __init__(self, experiment_name: str, run_id: str, tracking_uri: str = "http://192.168.4.85:8080"):
        """
        Initializes the Experiment object for a specific run of an experiment.

        Args:
            experiment_name (str): Name of the experiment.
            run_id (str): ID of the run within the experiment.
            tracking_uri (str): MLflow tracking URI.
        """
        self.experiment_name: str = experiment_name
        self.run_id: str = run_id
        self.tracking_uri: str = tracking_uri
        self.artifact_uri: str = None
        self.shap_df: pd.DataFrame = None
        self.feature_names: list[str] = None
        self.shap_values: np.ndarray = None
        self.feature_values: np.ndarray = None
        self._setup_mlflow()
        self._load_data()

    def _setup_mlflow(self):
        """Sets up the MLflow tracking URI."""
        mlflow.set_tracking_uri(uri=self.tracking_uri)

    def _load_data(self):
        """Loads data for the specific experiment run."""
        run = mlflow.get_run(self.run_id)
        self.artifact_uri = run.info.artifact_uri.replace(
            "mlflow-artifacts:",
            "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
        )

        shap_values_feather_path: str = os.path.join(
            self.artifact_uri, "shap_values_with_additional_columns.feather"
        )

        if not os.path.exists(shap_values_feather_path):
            raise FileNotFoundError(
                f"SHAP values file not found for run {self.run_id} at {shap_values_feather_path}"
            )

        self.shap_df = pd.read_feather(shap_values_feather_path)

        # Extract feature names and values
        self.feature_names = [
            col.replace("_shap", "")
            for col in self.shap_df.columns
            if col.endswith("_shap")
        ]
        columns_to_drop: list[str] = [
            "global_event_ID",
            "lon",
            "lat",
            "time",
            "KGClass",
            "KGMajorClass",
            "base_value",
        ]
        shap_df_cleaned: pd.DataFrame = self.shap_df.drop(columns=columns_to_drop, errors="ignore")
        self.shap_values = shap_df_cleaned[
            [f + "_shap" for f in self.feature_names]
        ].values
        self.feature_values = self.shap_df[self.feature_names].values

    def generate_summary_shap_plot(self, output_path: str = None):
        """
        Generates a SHAP summary plot for the experiment run.

        Args:
            output_path (str, optional): Path to save the plot. 
                                         If None, uses default artifact directory.
        """
        if output_path is None:
            output_path: str = os.path.join(
                self.artifact_uri, "summary_plots", "feature_summary_plot.png"
            )
        else:
            # Modify output_path to include experiment and run identifiers
            base_dir, filename = os.path.split(output_path)
            filename, ext = os.path.splitext(filename)
            new_filename = f"{filename}_{self.experiment_name}_{self.run_id}{ext}"
            output_path = os.path.join(base_dir, new_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.feature_values,
            feature_names=self.feature_names,
            show=False,
            plot_size=(12, 8),
        )
        plt.title(f"Feature Summary Plot - {self.experiment_name} - {self.run_id}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

    def generate_summary_shap_plot_by_group(self, output_path: str = None):
        """
        Generates a SHAP summary plot, grouped by feature groups, for the experiment run.

        Args:
            output_path (str, optional): Path to save the plot. 
                                         If None, uses default artifact directory.
        """
        if output_path is None:
            output_path: str = os.path.join(
                self.artifact_uri, "summary_plots", "feature_group_summary_plot.png"
            )
        else:
            # Modify output_path to include experiment and run identifiers
            base_dir, filename = os.path.split(output_path)
            filename, ext = os.path.splitext(filename)
            new_filename = f"{filename}_{self.experiment_name}_{self.run_id}{ext}"
            output_path = os.path.join(base_dir, new_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        feature_groups: dict[str, str] = get_feature_groups(self.feature_names)
        group_names: list[str] = list(set(feature_groups.values()))

        group_shap_values: list[np.ndarray] = []
        group_feature_values: list[np.ndarray] = []

        for group in group_names:
            group_features: list[str] = [f for f, g in feature_groups.items() if g == group]
            group_indices: list[int] = [self.feature_names.index(feat) for feat in group_features]

            group_shap: np.ndarray = self.shap_values[
                :, [self.feature_names.index(f) for f in group_features]
            ].sum(axis=1)
            group_shap_values.append(group_shap)

            group_feat: np.ndarray = self.feature_values[:, group_indices].mean(axis=1)
            group_feature_values.append(group_feat)

        group_shap_values: np.ndarray = np.array(group_shap_values).T
        group_feature_values: np.ndarray = np.array(group_feature_values).T

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            group_shap_values,
            group_feature_values,
            feature_names=group_names,
            show=False,
            plot_size=(12, 8),
        )
        plt.title(f"Feature Group Summary Plot - {self.experiment_name} - {self.run_id}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

    def generate_dependency_plots(self, output_dir: str = None):
        """
        Generates SHAP dependency plots for each feature for the experiment run.

        Args:
            output_dir (str, optional): Directory to save the plots. 
                                       If None, uses default artifact directory.
        """
        if output_dir is None:
            output_dir: str = os.path.join(self.artifact_uri, "dependency_plots")
        else:
            # Modify output_dir to include experiment and run identifiers
            output_dir = os.path.join(output_dir, f"{self.experiment_name}_{self.run_id}")

        os.makedirs(output_dir, exist_ok=True)

        for feature_name in self.feature_names:
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(
                feature_name,
                self.shap_values,
                self.feature_values,
                feature_names=self.feature_names,
                display_features=self.feature_values,
                show=False,
            )
            plt.title(f"SHAP Dependence Plot - {feature_name} - {self.experiment_name} - {self.run_id}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{feature_name}_dependence_plot.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close() 