#!/usr/bin/env python3
"""
Script to extract EC2 actions and descriptions from AWS documentation.
Saves to project-root data/ec2_actions.json
"""
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup


def extract_ec2_actions():
    url = "https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonec2.html#amazonec2-actions-as-permissions"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    actions_data = []
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        if not rows:
            continue
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            action_cell = cells[0].get_text(strip=True)
            desc_cell = cells[1].get_text(strip=True)
            if not action_cell or not desc_cell:
                continue
            action_name = action_cell.replace('ec2:', '').strip()
            if not action_name or action_name.lower() in ['action', 'actions', 'permission']:
                continue
            actions_data.append({"action": action_name, "description": desc_cell})
        if actions_data:
            break
    return actions_data


def main():
    print("Extracting EC2 actions from AWS documentation...")
    actions = extract_ec2_actions()
    if not actions:
        print("No actions extracted. Please check the URL and page structure.")
        return
    root = Path(__file__).resolve().parents[1]
    output_file = root / "data/ec2_actions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(actions, f, indent=2)
    print(f"Saved {len(actions)} actions to: {output_file}")
    print("Sample:")
    for a in actions[:10]:
        print(f"- {a['action']}: {a['description']}")


if __name__ == "__main__":
    main()