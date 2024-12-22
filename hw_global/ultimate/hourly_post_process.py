import argparse
from ultimate.mlflow_tools.experiment_manager import ExperimentManager

def main():
    """
    Main function to process experiments and generate graphs using ExperimentManager.
    """
    parser = argparse.ArgumentParser(description="Process experiments and generate graphs.")
    parser.add_argument("--pattern", required=True, help="Regular expression pattern to match experiment names.")
    parser.add_argument("--generate_types", required=True, help="Space-separated list of generation types (summary, group_summary, dependency, all).")
    parser.add_argument("--tracking_uri", required=True, help="MLflow tracking URI.")
    parser.add_argument("--output_path", required=True, help="Base output path for generated plots.")
    args = parser.parse_args()

    # Convert generate_types string to a list
    generate_types_list = args.generate_types.split()

    # Initialize the ExperimentManager
    manager = ExperimentManager(pattern=args.pattern, tracking_uri=args.tracking_uri)

    # Process experiments and generate plots
    manager.process_experiments(generate_types=generate_types_list, output_path=args.output_path)

if __name__ == "__main__":
    main() 