#!/usr/bin/env python
"""Helper script to update config with best hyperparameters from tuning.

Reads the best parameters from command line arguments and updates the YAML config.
"""

import sys
import yaml
from pathlib import Path


def update_config(config_path: str, stage: str, best_params: dict):
    """Update config file with best hyperparameters.

    Args:
        config_path: Path to YAML config file
        stage: 'encoder' or 'classifier'
        best_params: Dictionary of best parameters from tuning
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update stage-specific parameters
    stage_config = config[stage]
    for key, value in best_params.items():
        if key in stage_config:
            old_value = stage_config[key]
            stage_config[key] = value
            print(f"  {key}: {old_value} → {value}")

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Config updated: {config_path}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python update_config_from_hpo.py <config_path> <stage> <param1>=<value1> <param2>=<value2> ...")
        sys.exit(1)

    config_path = sys.argv[1]
    stage = sys.argv[2]

    # Parse parameters from command line
    best_params = {}
    for arg in sys.argv[3:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to parse as number
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Try to parse as list
                if value.startswith('[') and value.endswith(']'):
                    import ast
                    value = ast.literal_eval(value)
            best_params[key] = value

    print(f"\nUpdating {stage} parameters in {config_path}:")
    update_config(config_path, stage, best_params)


if __name__ == '__main__':
    main()
