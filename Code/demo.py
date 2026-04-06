#!/usr/bin/env python3
"""
Usage examples:
    # Single repo
    python demo.py --repo github.com/ImageMagick/ImageMagick --checkpoint-month 9 --checkpoint-day 2

    # Multiple repos from CSV
    python demo.py --csv demo.csv --checkpoint-month 9 --checkpoint-day 2 --start-year 2023 --end-year 2025

    # Specify output path
    python demo.py --csv demo.csv --output result.pkl
"""

import os
import sys
import csv
import json
import pickle
import argparse
import shutil
import configparser
from typing import Dict, Any, List, Tuple

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from score_utils import (
    _parse_repo_url,
    _get_default_branch,
    RepoContext,
    collect_metadata,
    collect_test_info,
    collect_developer_info,
    collect_issue_vulnerability_info
)
from metric_utils import our_score_card_metric,process_alarm


# Metric name to threshold key mapping
METRIC_MAP = {
    'code_churn_rate(%, 3week)': 'tau_cc',
    'burstiness(year)': 'tau_bst',
    'commit_frequency(year)': 'tau_cf',
    'test_proportion(%, year)': 'tau_tp',
    'test_code_churn_ratio(3week)': 'tau_ter',
    'core_developer_turnover(year)': 'tau_cdt',
    'truck_factor(year)': 'tau_tf',
    'newcomer_retention(%, year)': 'tau_nr',
    'issue_response_efficiency(day, year)': 'tau_ire',
    'backlog_management_index(%, year)': 'tau_bmi',
    'cve_exposure_increase(year)': 'tau_cer',
    'cve_severity(year)': 'tau_cs'
}


def load_tokens(config_path: str = None) -> Tuple[List[str], str, str]:
    """Load tokens from config file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'utils', 'token_config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    pat_token = config.get('tokens', 'PAT_TOKENS')
    nvd_token = config.get('tokens', 'NVD_TOKEN')
    return pat_token, nvd_token


# Load tokens from config file
PAT_TOKEN, NVD_TOKEN = load_tokens()
TEMP_DIR = './tmp_scan'



def load_alarm_config(config_path: str = None) -> Dict:
    """Load alarm threshold configuration"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'utils', 'alarm_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def read_repos_from_csv(csv_path: str) -> List[str]:
    """Read repository list from CSV file"""
    repos = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row.get('repo', '').strip()
            if repo:
                repos.append(repo)
    return repos



def our_score_card_scan(repo_url: str, check_point_date: Tuple[int, int], token: str,
                        start_year: int = None, end_year: int = None) -> Tuple[Dict, str]:
    """
    Perform comprehensive scan for a GitHub repository

    Returns:
        (result_dict, repo_path)
    """
    import subprocess
    import time

    result = {
        'repo_url': repo_url,
        'scan_period': check_point_date,
        'metadata': {},
        'test_info': {},
        'developer_info': {},
        'issue_vulnerability_info': {},
        'status': 'pending'
    }

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)

    owner, repo = _parse_repo_url(repo_url)
    repo_path = os.path.join(TEMP_DIR, repo)

    # Clone if not exists
    if os.path.exists(repo_path) and os.path.isdir(os.path.join(repo_path, '.git')):
        print(f"Repository already exists: {repo_path}")
    else:
        print(f"Cloning repository: {repo_url}")
        subprocess.run(f'git clone {repo_url} {repo_path}', shell=True, check=True, capture_output=True)

    default_branch = _get_default_branch(owner, repo, token)

    # Initialize context
    print(f"Initializing repository context...")
    ctx = RepoContext(repo_url, repo_path, check_point_date, default_branch, token)

    # Module 1: Metadata
    print(f"[1/4] Collecting metadata...")
    result['metadata'] = collect_metadata(
        repo_path, check_point_date, default_branch, ctx,
        start_year=start_year, end_year=end_year
    )

    # Module 2: Test info
    print(f"[2/4] Collecting test info...")
    result['test_info'] = collect_test_info(
        repo_path, check_point_date, default_branch, ctx,
        start_year=start_year, end_year=end_year
    )

    # Module 3: Developer info
    print(f"[3/4] Collecting developer info...")
    result['developer_info'] = collect_developer_info(
        repo_path, check_point_date, default_branch,
        result['metadata'], ctx,
        start_year=start_year, end_year=end_year
    )

    # Module 4: Issue and vulnerability info
    print(f"[4/4] Collecting issue and vulnerability info...")
    result['issue_vulnerability_info'] = collect_issue_vulnerability_info(
        repo_url, check_point_date, token, NVD_TOKEN, ctx,
        start_year=start_year, end_year=end_year
    )

    result['status'] = 'success'
    return result, repo_path


def get_our_score(repo_name: str, check_point_date: Tuple[int, int],
                  scorecard_dict: Dict, metric_dict: Dict,
                  start_year: int = None, end_year: int = None,
                  token_idx: int = 0) -> Tuple[Dict, Dict]:
    """
    Calculate score and metrics for a repository
    """
    repo_url = f'https://{repo_name}'
    token = PAT_TOKEN

    if repo_name not in scorecard_dict or 'error' in scorecard_dict.get(repo_name, {}):
        print(f"Running scan for: {repo_name}")
        # try:
        result, repo_path = our_score_card_scan(
            repo_url, check_point_date, token,
            start_year=start_year, end_year=end_year
        )
        scorecard_dict[repo_name] = result

        # Clean up cloned repo
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        # except Exception as e:
        #     print(f"Error scanning {repo_name}: {e}")
        #     scorecard_dict[repo_name] = {'error': str(e), 'status': 'failed'}
    else:
        print(f"Scan already exists for: {repo_name}")

    # Calculate metrics
    if repo_name not in metric_dict:
        if scorecard_dict[repo_name].get('status') == 'success':
            metric_dict[repo_name] = our_score_card_metric(scorecard_dict[repo_name], end_year=end_year or 2025)

    return scorecard_dict, metric_dict



def check_alarm(metric_data: Dict, year: int, config: Dict) -> Dict[str, float]:
    """
    Check which metrics exceed thresholds for a given year

    Returns:
        Dict of {metric_name: metric_value} for triggered alarms
    """
    tau_to_metric = {v: k for k, v in METRIC_MAP.items()}

    # Build threshold checks from config
    threshold_checks = []
    for thresh_config in config['thresholds'].values():
        for key, val in thresh_config.items():
            if key.startswith('tau_') and key in tau_to_metric:
                threshold_checks.append({
                    'metric_key': tau_to_metric[key],
                    'threshold': val,
                    'direction': thresh_config['direction']
                })

    years = metric_data.get('years', [])
    if year not in years:
        return {}

    year_idx = years.index(year)
    triggered = {}

    for check in threshold_checks:
        metric_key = check['metric_key']
        values = metric_data.get(metric_key, [])

        if year_idx >= len(values):
            continue

        val = values[year_idx]
        if val is None:
            continue

        threshold = check['threshold']
        direction = check['direction']

        # Check if threshold exceeded
        if (direction == 'g' and val >= threshold) or (direction == 'l' and val <= threshold):
            triggered[metric_key] = val
    triggered = process_alarm(triggered)
    return triggered


def main():
    parser = argparse.ArgumentParser(description='Calculate repository security metrics and check alarms')

    # Input options
    parser.add_argument('--repo', type=str, help='Single repository (e.g., github.com/owner/repo)')
    parser.add_argument('--csv', type=str, default='demo.csv', help='CSV file with repository list')

    # Time options
    parser.add_argument('--checkpoint-month', type=int, default=9, help='Checkpoint month (default: 9)')
    parser.add_argument('--checkpoint-day', type=int, default=2, help='Checkpoint day (default: 2)')
    parser.add_argument('--start-year', type=int, default=2023, help='Start year (default: 2023)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')

    # Output options
    parser.add_argument('--output', type=str, default='demo_result.pkl', help='Output pkl file path')
    parser.add_argument('--config', type=str, help='Alarm config JSON path')

    args = parser.parse_args()

    # Get repo list
    if args.repo:
        repos = [args.repo]
    else:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), csv_path)
        repos = read_repos_from_csv(csv_path)

    print(f"Processing {len(repos)} repositories")
    print(f"Checkpoint: {args.checkpoint_month}/{args.checkpoint_day}")
    print(f"Year range: {args.start_year} - {args.end_year}")

    check_point_date = (args.checkpoint_month, args.checkpoint_day)
    config = load_alarm_config(args.config)

    scorecard_dict = {}
    metric_dict = {}

    # Final result: {year: {metric_name: value}} for triggered alarms
    alarm_result = {year: {} for year in range(args.start_year, args.end_year + 1)}

    for idx, repo_name in enumerate(repos):
        print(f"\n{'='*50}")
        print(f"[{idx+1}/{len(repos)}] Processing: {repo_name}")
        print('='*50)

        scorecard_dict, metric_dict = get_our_score(
            repo_name, check_point_date,
            scorecard_dict, metric_dict,
            start_year=args.start_year,
            end_year=args.end_year,
            token_idx=idx
        )

        # Check alarms for each year
        if repo_name in metric_dict:
            for year in range(args.start_year, args.end_year + 1):
                triggered = check_alarm(metric_dict[repo_name], year, config)
                if triggered:
                    if repo_name not in alarm_result[year]:
                        alarm_result[year][repo_name] = {}
                    alarm_result[year][repo_name] = triggered
                    print(f"  Year {year} alarms: {list(triggered.keys())}")

    # Save results
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    with open(output_path, 'wb') as f:
        pickle.dump(alarm_result, f)

    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")

    # Print summary
    for year in range(args.start_year, args.end_year + 1):
        alarmed_count = len(alarm_result[year])
        print(f"Year {year}: {alarmed_count} repos with alarms")


if __name__ == '__main__':
    main()
