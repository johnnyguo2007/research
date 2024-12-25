import mlflow
import pandas as pd
import os
import re
import shap
import matplotlib.pyplot as plt
import numpy as np
import scipy
import logging
import mlflow.catboost



def get_feature_groups(feature_names):
    """
    Assign features to groups based on specified rules.

    Args:
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping from feature names to group names.
    """
    prefixes = ('delta_', 'hw_nohw_diff_', 'Double_Differencing_')
    feature_groups = {}
    for feature in feature_names:
        group = feature
        for prefix in prefixes:
            if feature.startswith(prefix):
                group = feature[len(prefix):]
                break
        # If feature does not start with any prefix, it is its own group, but name the group feature + "Level"
        if group == feature:
            group = feature + "_Level"
        feature_groups[feature] = group
    return feature_groups

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
        self.model = None
        self._setup_mlflow()
        self._load_data()
        # self._load_model()

    def _setup_mlflow(self):
        """Sets up the MLflow tracking URI."""
        mlflow.set_tracking_uri(uri=self.tracking_uri)

    def _load_data(self):
        """Loads data for the specific experiment run."""
        run = mlflow.get_run(self.run_id)
        logging.warning(
                f"experiment_name  {self.experiment_name} run.info.artifact_uri {run.info.artifact_uri}"
        )
        self.artifact_uri = run.info.artifact_uri.replace(
            "mlflow-artifacts:",
            "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
        )
        logging.warning(f"self.artifact_uri {self.artifact_uri}")

        shap_values_feather_path: str = os.path.join(
            self.artifact_uri, "shap_values_with_additional_columns.feather"
        )

        if not os.path.exists(shap_values_feather_path):
            logging.warning(
                f"SHAP values file not found for run {self.run_id} at {shap_values_feather_path}. "
                "Some functionalities might be unavailable."
            )
            self.shap_df = None
            return

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
        # Only drop columns that exist
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.shap_df.columns]
        shap_df_cleaned: pd.DataFrame = self.shap_df.drop(columns=existing_columns_to_drop)
        self.shap_values = shap_df_cleaned[
            [f + "_shap" for f in self.feature_names]
        ].values
        self.feature_values = self.shap_df[self.feature_names].values

    def _load_model(self, model_subdur: str = "hourly_model"):
        """Loads the model for the specific experiment run and subduration."""
        model_path = os.path.join(self.artifact_uri, model_subdur)
        self.model = mlflow.catboost.load_model(model_path)

    def generate_summary_shap_plot(self):
        """
        Generates a SHAP summary plot for the experiment run.
        """
        base_dir: str = "."
        logging.info("Starting generate_summary_shap_plot method")
        output_path: str = os.path.join(
            self.artifact_uri, base_dir, "feature_summary_plot.png"
        )

        logging.info(f"Creating directory: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.feature_values,
            feature_names=self.feature_names,
            show=False,
            plot_size=(12, 8),
        )
        plt.title(f"Feature Summary Plot - {self.experiment_name}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Generated summary SHAP plot at {output_path}")

    def generate_summary_shap_plot_by_group(self):
        """
        Generates a SHAP summary plot, grouped by feature groups, for the experiment run.
        """
        base_dir: str = "group_summary_plots"
        logging.info("Starting generate_summary_shap_plot_by_group method")
        output_path: str = os.path.join(
            self.artifact_uri, base_dir, "feature_group_summary_plot.png"
        )

        logging.info(f"Creating directory: {os.path.dirname(output_path)}")
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
        plt.title(f"Feature Group Summary Plot - {self.experiment_name}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Generated summary SHAP plot by group at {output_path}")

    def generate_dependency_plots(self):
        """
        Generates SHAP dependency plots for each feature, showing the top 3
        features with the highest interaction, for the experiment run.
        """
        base_dir: str = "x_dependence_plots"
        logging.info("Starting generate_dependency_plots method")

        # Use base_dir directly under artifact_uri
        output_dir = os.path.join(self.artifact_uri, base_dir)
        logging.info(f"Using output directory: {output_dir}")

        for feature_index, target_feature_name in enumerate(self.feature_names):
            logging.info(f"Processing feature: {target_feature_name} (index {feature_index})")

            # Get top 3 interacting features for the target feature
            logging.info("Calculating top 3 interacting features")
            explanation = shap.Explanation(values=self.shap_values, data=self.feature_values, feature_names=self.feature_names)
            interaction_indices = shap.utils.potential_interactions(explanation[:, target_feature_name], explanation)
            
            # Handle cases with fewer than 3 interactions
            top_interaction_indices = interaction_indices[:min(3, len(interaction_indices))]
            top_interacting_features = [
                self.feature_names[i] for i in top_interaction_indices if self.feature_names[i] != target_feature_name
            ]
            logging.info(f"Top interacting features for {target_feature_name}: {top_interacting_features}")

            rank: int = 1
            # Create nested directory structure under 'x_dependence_plots'
            full_nested_path = os.path.join(output_dir, target_feature_name)
            os.makedirs(full_nested_path, exist_ok=True)
            for interacting_feature_name in top_interacting_features:
                logging.info(f"Creating dependency plot for {target_feature_name} vs {interacting_feature_name} (rank {rank})")
                shap.plots.scatter(
                    explanation[:, target_feature_name], color=explanation[:, interacting_feature_name], show=False
                )
                plt.title(
                    f"{target_feature_name} vs {interacting_feature_name} (Interaction Rank: {rank})\n"
                    f"{self.experiment_name}"
                )
        
                # plt.tight_layout()
                
                # Save the plot in the nested directory
                output_path: str = os.path.join(
                    full_nested_path, f"{target_feature_name}_vs_{interacting_feature_name}_dependence_plot.png"
                )
                plt.savefig(
                    output_path,
                    # bbox_inches="tight",
                    # dpi=300,
                )
                plt.close()
                logging.info(f"Saved dependency plot to {output_path}")
                rank += 1

        logging.info("Finished generate_dependency_plots method")

    def generate_marginal_effects_plot(
        self,
        marginal_effects_data=None,
        num_samples: int = 100,
        max_points: int = 20,
        logit: bool = True,
        seed: int = 0,
    ):
        """
        Generates a plot similar to marginal_effects, showing the true marginal causal effects
        and overlays it with SHAP values for comparison.

        Args:
            marginal_effects_data (list, optional): Pre-calculated marginal effects data.
            num_samples (int): Number of samples to use for generating the marginal effects.
            max_points (int): Maximum number of points to plot for each feature.
            logit (bool): Whether to apply logit transformation to the 'Did renew' values.
            seed (int): Seed for random number generation.
        """
        base_dir: str = "marginal_plots"
        logging.info("Starting generate_marginal_effects_plot method")
        output_path = os.path.join(self.artifact_uri, base_dir, "marginal_effects_vs_shap.png")

        logging.info(f"Creating directory: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Calculate marginal effects if not provided
        if marginal_effects_data is None:
            from sandbox.econ import generator, marginal_effects  # Assuming you have access to these

            marginal_effects_data = marginal_effects(
                generator, num_samples, self.feature_names, max_points, logit, seed
            )
        
        plt.figure(figsize=(12, 8))
        shap.plots.scatter(
            self.shap_values,
            ylabel="SHAP value\n(higher means more likely to renew)",
            overlay={"True causal effects": marginal_effects_data},
            feature_names=self.feature_names,
            show=False,
        )
        plt.title(f"Marginal Effects vs. SHAP Values - {self.experiment_name}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Generated marginal effects plot at {output_path}")

    def calculate_marginal_effects(self, columns=None, num_samples=100, max_points=20, logit=True, seed=0):
        """
        Calculates marginal effects based on SHAP values and associated data,
        aligning with the algorithm in econ.py's marginal_effects.

        Args:
            columns (list, optional): List of column names to calculate marginal effects for.
                                      If None, defaults to all SHAP value columns.
            num_samples (int, optional): Number of samples to generate for each column. Defaults to 100.
            max_points (int, optional): Maximum number of points to plot for each feature. Defaults to 20.
            logit (bool, optional): Whether to apply logit transformation to the base value. Defaults to True.
            seed (int, optional): Seed for random number generation. Defaults to 0.

        Returns:
            list: A list of (x, y) pairs, where x represents the unique values of a column
                  and y represents the corresponding centered average SHAP values.
        """
        if self.shap_df is None:
            logging.warning("Warning: SHAP data not loaded. Cannot calculate marginal effects.")
            return []

        if columns is None:
            columns = [col for col in self.shap_df.columns if col.endswith('_shap')]

        np.random.seed(seed)
        results = []

        for col in columns:
            data_col = col.replace('_shap', '')
            if data_col not in self.shap_df.columns:
                logging.warning(f"Data column '{data_col}' not found for SHAP column '{col}'. Skipping.")
                continue

            # Sample data points and select unique values using linspace directly
            x_values = np.linspace(self.shap_df[data_col].min(), self.shap_df[data_col].max(), max_points)

            y_values = []
            for x in x_values:
                # Create a temporary copy of the DataFrame and set the column value
                temp_df = self.shap_df.copy()
                temp_df[data_col] = x

                # Calculate the mean SHAP value plus base value
                avg_shap = temp_df[col].mean()
                val = avg_shap + temp_df["base_value"].mean()

                if logit:
                    val = scipy.special.logit(val)
                y_values.append(val)

            y_values = np.array(y_values)
            y_values = y_values - np.nanmean(y_values)
            results.append((x_values, y_values))

        return results 

    def generate_waterfall_plot(self):
        """
        Generates a SHAP waterfall plot for the first observation in the experiment run.
        """
        base_dir: str = "."
        logging.info("Starting generate_waterfall_plot method")
        output_path = os.path.join(
            self.artifact_uri, base_dir, "waterfall_plot.png"
        )

        logging.info(f"Creating directory: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a SHAP Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[0, :],
            base_values=self.shap_df["base_value"].iloc[0],
            data=self.feature_values[0, :],
            feature_names=self.feature_names
        )

        # Generate the waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Waterfall Plot - {self.experiment_name}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Generated waterfall plot at {output_path}") 