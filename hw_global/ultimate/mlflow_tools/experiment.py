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
        self._load_model()

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

    def _load_model(self, model_subdur: str = "hourly_model"):
        """Loads the model for the specific experiment run and subduration."""
        model_path = os.path.join(self.artifact_uri, model_subdur)
        self.model = mlflow.catboost.load_model(model_path)

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
        Generates SHAP dependency plots for each feature, showing the top 3
        features with the highest interaction, for the experiment run.

        Args:
            output_dir (str, optional): Directory to save the plots.
                                       If None, uses default artifact directory.
        """
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Starting generate_dependency_plots method")

        if output_dir is None:
            output_dir: str = os.path.join(self.artifact_uri, "dependency_plots")
            logging.info(f"Output directory not provided, using default: {output_dir}")
        else:
            # Modify output_dir to include experiment and run identifiers
            output_dir = os.path.join(output_dir, f"{self.experiment_name}_{self.run_id}")
            logging.info(f"Using provided output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

        # Calculate interaction values using TreeExplainer for CatBoost models
        # logging.info("Initializing TreeExplainer for CatBoost model")
        # explainer = shap.TreeExplainer(self.model)
        # logging.info("Calculating SHAP interaction values using shap_interaction_values")
        # interaction_values = explainer.shap_interaction_values(self.feature_values)
        # logging.info(f"Shape of interaction_values: {interaction_values.shape}")

        for feature_index, feature_name in enumerate(self.feature_names):
            logging.info(f"Processing feature: {feature_name} (index {feature_index})")

            # Get top 3 interacting features for the target feature
            logging.info("Calculating top 3 interacting features")
            # top_interactions = np.abs(interaction_values[:, feature_index, :]).mean(0).argsort()[::-1][:3]
            # top_features = [self.feature_names[i] for i in top_interactions if self.feature_names[i] != feature_name]
            explanation = shap.Explanation(values=self.shap_values, data=self.feature_values, feature_names=self.feature_names)
            top_interactions = shap.utils.potential_interactions(explanation[:, feature_name], explanation)
            top_features = [self.feature_names[i] for i in top_interactions[:3] if self.feature_names[i] != feature_name]
            logging.info(f"Top interacting features for {feature_name}: {top_features}")

            rank: int = 1
            # Create nested directory structure under 'x_dependence_plots'
            nested_path = os.path.join('x_dependence_plots', feature_name)
            full_nested_path = os.path.join(self.artifact_uri, nested_path)
            os.makedirs(full_nested_path, exist_ok=True)
            for interacting_feature in top_features:
                logging.info(f"Creating dependency plot for {feature_name} vs {interacting_feature} rank {rank}")
                shap.plots.scatter(explanation[:, feature_name], color=explanation[:, interacting_feature], show=False)
                plt.title(
                    f"SHAP Dependence Plot - {feature_name} vs {interacting_feature} (Interaction Rank: {rank}) - {self.experiment_name}"
                )
                # plt.tight_layout()
                
                # Save the plot in the nested directory
                output_path: str = os.path.join(
                    full_nested_path, f"{feature_name}_vs_{interacting_feature}_dependence_plot.png"
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

    def generate_marginal_effects_plot(self, output_path: str = None, num_samples: int = 100, max_points: int = 20, logit: bool = True, seed: int = 0):
        """
        Generates a plot similar to marginal_effects, showing the true marginal causal effects
        and overlays it with SHAP values for comparison.

        Args:
            output_path (str, optional): Path to save the plot. If None, uses default artifact directory.
            num_samples (int): Number of samples to use for generating the marginal effects.
            max_points (int): Maximum number of points to plot for each feature.
            logit (bool): Whether to apply logit transformation to the 'Did renew' values.
            seed (int): Seed for random number generation.
        """


        if output_path is None:
            output_path = os.path.join(
                self.artifact_uri, "summary_plots", "marginal_effects_vs_shap.png"
            )
        else:
            # Modify output_path to include experiment and run identifiers
            base_dir, filename = os.path.split(output_path)
            filename, ext = os.path.splitext(filename)
            new_filename = f"{filename}_{self.experiment_name}_{self.run_id}{ext}"
            output_path = os.path.join(base_dir, new_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Assuming you have a way to access the 'generator' function from 'econ.py'
        # You might need to adjust this part based on how you can access it
        from sandbox.econ import generator, marginal_effects

        true_effects = marginal_effects(generator, num_samples, self.feature_names, max_points, logit, seed)

        plt.figure(figsize=(12, 8))
        shap.plots.scatter(
            self.shap_values,
            ylabel="SHAP value\n(higher means more likely to renew)",
            overlay={"True causal effects": true_effects},
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f"Marginal Effects vs. SHAP Values - {self.experiment_name} - {self.run_id}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

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
            print("Warning: SHAP data not loaded. Cannot calculate marginal effects.")
            return []

        if columns is None:
            columns = [col for col in self.shap_df.columns if col.endswith('_shap')]

        np.random.seed(seed)
        results = []

        for col in columns:
            data_col = col.replace('_shap', '')
            if data_col not in self.shap_df.columns:
                print(f"Warning: Data column '{data_col}' not found for SHAP column '{col}'. Skipping.")
                continue

            # Sample data points and select unique values
            x_values = np.random.choice(self.shap_df[data_col].values, size=num_samples, replace=True)
            x_values = np.sort(x_values)
            x_values = np.unique([np.nanpercentile(x_values, v, method="nearest") for v in np.linspace(0, 100, max_points)])

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