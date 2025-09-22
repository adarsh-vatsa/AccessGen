#!/usr/bin/env python3
"""
Script to enrich AWS IAM EC2 registry JSON with official AWS action descriptions.
"""
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_description_mapping(descriptions_file: str) -> Dict[str, str]:
    descriptions = load_json_file(descriptions_file)
    return {item['action']: item['description'] for item in descriptions}


def enrich_ec2_registry(registry_file: str, descriptions_file: str, output_file: str) -> None:
    registry = load_json_file(registry_file)
    description_map = create_description_mapping(descriptions_file)

    actions_processed = 0
    actions_with_descriptions = 0

    if 'ec2' in registry and 'actions' in registry['ec2']:
        for action in registry['ec2']['actions']:
            actions_processed += 1
            action_name = action.get('action', '')

            if action_name in description_map:
                action['description'] = description_map[action_name]
                actions_with_descriptions += 1
            else:
                action['description'] = ""
                print(f"Warning: No description found for action: {action_name}")

    save_json_file(registry, output_file)

    print("Enrichment complete!")
    print(f"Actions processed: {actions_processed}")
    print(f"Actions with descriptions: {actions_with_descriptions}")
    print(f"Actions without descriptions: {actions_processed - actions_with_descriptions}")
    print(f"Enriched JSON saved to: {output_file}")


if __name__ == "__main__":
    # Project root relative paths
    root = Path(__file__).resolve().parents[1]
    registry_file = str(root / "data/aws_iam_registry_ec2.json")
    descriptions_file = str(root / "data/ec2_actions.json")
    output_file = str(root / "enriched_data/aws_iam_registry_ec2_enriched.json")

    os.makedirs(root / "enriched_data", exist_ok=True)
    enrich_ec2_registry(registry_file, descriptions_file, output_file)