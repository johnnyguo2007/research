import mlflow
import re
from mlflow_tools.experiment import Experiment
import logging
import pandas as pd
import os


class ExperimentManager:
    def __init__(self, pattern: str, tracking_uri: str = "http://192.168.4.85:8080"):
        """
        Initializes the ExperimentManager.

        Args:
            pattern (str): Regular expression pattern to match experiment names.
            tracking_uri (str): MLflow tracking URI.
        """
        self.pattern: str = pattern
        self.tracking_uri: str = tracking_uri
        self.experiments: list[Experiment] = []
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Sets up the MLflow tracking URI."""
        mlflow.set_tracking_uri(uri=self.tracking_uri)

    def _load_and_process_experiment(
        self,
        experiment_name: str,
        generate_funcs: list[callable],
        output_path: str = None,
        model_subdur: str = "hourly_model",
    ):
        """Loads an experiment, processes it, and then removes it from memory."""
        logging.info(
            f"Announcing the start of processing experiment '{experiment_name}'"
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)

        # Fetch only the latest run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time desc"],
            max_results=1,  # Fetch only the latest run
        )

        if runs.empty:
            logging.warning(f"No runs found for experiment '{experiment_name}'")
            return

        # Process only the latest run
        run = runs.iloc[0]  # Get the first (and only) row
        try:
            exp_obj = Experiment(
                experiment_name=experiment_name,
                run_id=run.run_id,
                tracking_uri=self.tracking_uri,
            )
            exp_obj._load_model(model_subdur)
            
            # Check if shap_df is empty
            if exp_obj.shap_df is None or exp_obj.shap_df.empty:
                logging.warning(f"SHAP data is empty for experiment '{experiment_name}', run '{run.run_id}'. Skipping.")
                return

            for generate_func in generate_funcs:
                try:
                    generate_func(exp_obj)
                except Exception as e:
                    logging.error(
                        f"Error generating plot for experiment '{experiment_name}', run '{run.run_id}': {e}"
                    )
            del exp_obj
        except FileNotFoundError as e:
            logging.error(f"File not found error: {e}")
        except Exception as e:
            logging.error(
                f"Error processing experiment '{experiment_name}', run '{run.run_id}': {e}"
            )

    def process_experiments(
        self,
        generate_types: list[str] = ["summary"],
        output_path: str = None,
        args=None,
    ):
        """
        Processes experiments sequentially based on the specified generation types.

        Args:
            generate_types (list[str]): Types of generation to perform ('summary', 'group_summary', 'dependency', 'all').
            output_path (str, optional): Base output path for generated plots.
        """
        experiment_names: list[str] = [
            exp.name
            for exp in mlflow.search_experiments()
            if re.match(self.pattern, exp.name)
        ]

        if not experiment_names:
            raise ValueError(f"No experiments found matching pattern '{self.pattern}'")

        generate_funcs: list[callable] = []
        if "combine_shap" in generate_types:
            self._combine_shap_values(experiment_names)

        if "all" in generate_types:
            generate_funcs.extend(
                [
                    lambda exp: exp.generate_summary_shap_plot(),
                    lambda exp: exp.generate_summary_shap_plot_by_group(),
                    lambda exp: exp.generate_dependency_plots(),
                    lambda exp: exp.generate_marginal_effects_plot(),
                    lambda exp: exp.generate_waterfall_plot(),
                ]
            )
        other_generate_types = [
            gen_type
            for gen_type in generate_types
            if gen_type not in {"combine_shap", "all"}
        ]
        if other_generate_types:
            for gen_type in other_generate_types:
                if gen_type == "summary":
                    generate_funcs.append(lambda exp: exp.generate_summary_shap_plot())
                elif gen_type == "group_summary":
                    generate_funcs.append(
                        lambda exp: exp.generate_summary_shap_plot_by_group()
                    )
                elif gen_type == "dependency":
                    generate_funcs.append(lambda exp: exp.generate_dependency_plots())
                elif gen_type == "marginal_effects":
                    generate_funcs.append(
                        lambda exp: exp.generate_marginal_effects_plot()
                    )
                elif gen_type == "waterfall":
                    generate_funcs.append(lambda exp: exp.generate_waterfall_plot())
                else:
                    raise ValueError(f"Invalid generate_type: {gen_type}")

        if generate_funcs:
            for experiment_name in experiment_names:
                self._load_and_process_experiment(
                    experiment_name, generate_funcs, output_path, args.model_subdur
                )
        else:
            logging.warning(
                "No generation functions specified. Skipping experiment processing."
            )

    def _combine_shap_values(self, experiment_names: list[str]):
        """
        Combines SHAP values from multiple experiments into a single DataFrame
        and saves it as a new experiment in MLflow.

        Args:
            experiment_names (list[str]): List of experiment names to combine.
        """
        logging.info("Starting _combine_shap_values method")

        combined_shap_df = pd.DataFrame()
        for experiment_name in experiment_names:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logging.warning(f"Experiment '{experiment_name}' not found. Skipping.")
                continue

            # Fetch only the latest run for each experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time desc"],
                max_results=1,  # Fetch only the latest run
            )

            if runs.empty:
                logging.warning(
                    f"No runs found for experiment '{experiment_name}'. Skipping."
                )
                continue

            # Process only the latest run
            run = runs.iloc[0]
            try:
                artifact_uri = run.artifact_uri.replace(
                    "mlflow-artifacts:",
                    "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
                )
                shap_values_path = os.path.join(
                    artifact_uri, "shap_values_with_additional_columns.feather"
                )

                if os.path.exists(shap_values_path):
                    shap_df = pd.read_feather(shap_values_path)
                    combined_shap_df = pd.concat(
                        [combined_shap_df, shap_df], ignore_index=True
                    )
                    logging.info(
                        f"Combined SHAP values from run {run.run_id} of experiment '{experiment_name}'"
                    )
                else:
                    logging.warning(
                        f"SHAP values file not found for run {run.run_id} of experiment '{experiment_name}'. Skipping."
                    )

            except Exception as e:
                logging.error(
                    f"Error processing run {run.run_id} of experiment '{experiment_name}': {e}"
                )

        if combined_shap_df.empty:
            logging.warning("No SHAP values were combined.")
            return

        new_experiment_name = self._generate_combined_experiment_name(experiment_names)

        try:
            new_experiment_id = mlflow.create_experiment(name=new_experiment_name)
            logging.info(
                f"Created new experiment '{new_experiment_name}' with ID {new_experiment_id}"
            )
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                logging.warning(
                    f"Experiment '{new_experiment_name}' already exists. Using existing experiment."
                )
                existing_experiment = mlflow.get_experiment_by_name(new_experiment_name)
                new_experiment_id = existing_experiment.experiment_id
            else:
                logging.error(f"Error creating experiment '{new_experiment_name}': {e}")
                return

        with mlflow.start_run(
            experiment_id=new_experiment_id, run_name="combined_shap_values"
        ) as run:
            artifact_uri = run.info.artifact_uri.replace(
                "mlflow-artifacts:",
                "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
            )
            combined_shap_path = os.path.join(
                artifact_uri, "shap_values_with_additional_columns.feather"
            )

            # Create the necessary directories if they don't exist
            os.makedirs(os.path.dirname(combined_shap_path), exist_ok=True)

            combined_shap_df.to_feather(combined_shap_path)
            logging.info(f"Saved combined SHAP values to {combined_shap_path}")

            mlflow.log_artifact(combined_shap_path)
            logging.info("Logged combined SHAP values as an artifact in MLflow")

        logging.info("Finished _combine_shap_values method")

    def _generate_combined_experiment_name(self, experiment_names: list[str]) -> str:
        """
        Generates a new experiment name based on the common part of the input experiment names.

        Args:
            experiment_names (list[str]): List of experiment names.

        Returns:
            str: New experiment name.
        """
        if not experiment_names:
            return "Combined_SHAP_Values"

        # Remove unwanted characters, anything between square brackets, and trailing digits
        pattern = re.sub(r"\(.*?\)|\[.*?\]|\d+$|[|^$]", "", self.pattern)

        return f"Combined_{pattern}"
