import mlflow
import re
from mlflow_tools.experiment import Experiment

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

    def _load_and_process_experiment(self, experiment_name: str, generate_funcs: list[callable], output_path: str = None, model_subdur: str = "hourly_model"):
        """Loads an experiment, processes it, and then removes it from memory."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time desc"],
        )

        if runs.empty:
            print(f"No runs found for experiment '{experiment_name}'")
            return

        for _, run in runs.iterrows():
            try:
                exp_obj = Experiment(
                    experiment_name=experiment_name,
                    run_id=run.run_id,
                    tracking_uri=self.tracking_uri,
                )
                exp_obj._load_model(model_subdur)
                for generate_func in generate_funcs:
                    generate_func(exp_obj, output_path)  # Call each specified generate function
                del exp_obj  # Remove experiment object from memory
            except FileNotFoundError as e:
                print(e)

    def process_experiments(self, generate_types: list[str] = ["summary"], output_path: str = None, args = None):
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
        if "all" in generate_types:
            generate_funcs.extend([
                lambda exp, path: exp.generate_summary_shap_plot(path),
                lambda exp, path: exp.generate_summary_shap_plot_by_group(path),
                lambda exp, path: exp.generate_dependency_plots(path),
                lambda exp, path: exp.generate_marginal_effects_plot(path)
            ])
        else:
            for gen_type in generate_types:
                if gen_type == "summary":
                    generate_funcs.append(lambda exp, path: exp.generate_summary_shap_plot(path))
                elif gen_type == "group_summary":
                    generate_funcs.append(lambda exp, path: exp.generate_summary_shap_plot_by_group(path))
                elif gen_type == "dependency":
                    generate_funcs.append(lambda exp, path: exp.generate_dependency_plots(path))
                elif gen_type == "marginal_effects":
                    generate_funcs.append(lambda exp, path: exp.generate_marginal_effects_plot(path))
                else:
                    raise ValueError(f"Invalid generate_type: {gen_type}")

        for experiment_name in experiment_names:
            self._load_and_process_experiment(experiment_name, generate_funcs, output_path, args.model_subdur)