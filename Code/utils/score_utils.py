import os
import requests
import tempfile
import shutil
import subprocess
from typing import Dict, Tuple, Any, List
from datetime import datetime, timedelta, timezone
import statistics
import math
from urllib.parse import urlparse
from collections import defaultdict
import re
from urllib.parse import quote
import nvdlib
from tqdm import trange
GITHUB_API = "https://api.github.com"

def _parse_git_datetime(date_str: str) -> datetime:
    """
    Parse Git datetime string with robust error handling for malformed timezones

    Handles cases like:
    - '2011-09-06T10:54:23+05:18' (valid)
    - '2011-09-06T10:54:23+518:00' (invalid, should be +05:18)
    - '2011-09-06T10:54:23+0518' (valid)
    """
    try:
        # First try standard parsing
        return datetime.fromisoformat(date_str)
    except ValueError:
        # Handle malformed timezone like +518:00 or +1030:00
        # Pattern: YYYY-MM-DDTHH:MM:SS+HHH:MM or +HHMM
        match = re.match(r'^(.+?)([+-])(\d+):(\d+)$', date_str)
        if match:
            base_date, sign, tz_hours, tz_mins = match.groups()

            # Fix malformed timezone hours (e.g., 518 -> 05:18)
            if len(tz_hours) > 2:
                # Extract last 2 digits as real hours, rest as additional minutes
                extra_mins = int(tz_hours[:-2]) if len(tz_hours) > 2 else 0
                tz_hours = tz_hours[-2:]
                tz_mins = str(int(tz_mins) + extra_mins * 60).zfill(2)

            # Reconstruct with corrected timezone
            corrected_date = f"{base_date}{sign}{tz_hours.zfill(2)}:{tz_mins.zfill(2)}"
            try:
                return datetime.fromisoformat(corrected_date)
            except ValueError:
                pass

        # If still fails, try removing timezone and use UTC
        match = re.match(r'^(.+?)([+-]\d+:\d+)$', date_str)
        if match:
            base_date, _ = match.groups()
            try:
                # Parse without timezone and assume UTC
                dt = datetime.fromisoformat(base_date)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Last resort: raise the original error
        raise ValueError(f"Cannot parse datetime string: {date_str}")

def _run_git_command(cmd: str, cwd: str) -> str:
    """Execute git command and return output"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True, errors='ignore')
    return result.stdout.strip()

def _estimate_total_loc_at_commit(repo_path: str, commit_hash: str) -> int:
    """Count total LOC at a specific commit (skip binary blobs)."""
    total = 0
    file_list = _run_git_command(f'git ls-tree -r --name-only {commit_hash}', repo_path)
    for path in [p for p in file_list.split('\n') if p.strip()]:
        try:
            content = _run_git_command(f'git show {commit_hash}:{path}', repo_path)
        except Exception:
            continue
        if '\x00' in content:
            continue
        total += len([l for l in content.split('\n') if l.strip()])
    return total




def _parse_repo_url(repo_url: str) -> Tuple[str, str]:
    """Parse GitHub repository URL to extract owner and repo name"""
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    raise ValueError(f"Invalid repository URL: {repo_url}")


def _get_repo_info(owner: str, repo: str, token: str) -> Dict[str, Any]:
    """Get basic repository information"""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def _get_default_branch(owner: str, repo: str, token: str) -> str:
    """Get repository default branch name"""
    repo_info = _get_repo_info(owner, repo, token)
    return repo_info['default_branch']


class RepoContext:
    """
    Pre-fetch and cache common data needed by all collect functions.
    Avoids redundant Git commands and API requests.
    """
    def __init__(self, repo_url: str, repo_path: str, check_point_date: Tuple[int, int],
                 default_branch: str, token: str):
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.check_point_date = check_point_date
        self.default_branch = default_branch
        self.token = token

        # Parse repo info once
        self.owner, self.repo_name = _parse_repo_url(repo_url)

        # Get current year once
        self.current_year = datetime.now(timezone.utc).year

        # Get first commit and creation year once (for local git-based modules)
        self.first_commit = None
        self.creation_year = None
        self._init_git_info()

        # GraphQL headers (for API-based module 4)
        self.graphql_headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        # REST API headers
        self.api_headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        # Get repository creation info via API (for module 4)
        self.repo_created_at = None
        self.api_creation_year = None
        self._init_api_info()

    def _init_git_info(self):
        """Initialize Git repository info (for modules 1, 2, 3)"""
        try:
            self.first_commit = _run_git_command(
                f'git log --reverse --format=%cd --date=iso-strict {self.default_branch} -- | head -n 1',
                self.repo_path
            )
            self.creation_year = int(self.first_commit.split('-')[0])
            print(f"  Git repo creation year: {self.creation_year}")
        except Exception as e:
            print(f"  Warning: Failed to get git creation info: {e}")
            self.creation_year = self.current_year

    def _init_api_info(self):
        """Initialize API repository info (for module 4)"""
        try:
            graphql_url = "https://api.github.com/graphql"
            repo_query = """
            query($owner: String!, $repo: String!) {
                repository(owner: $owner, name: $repo) {
                    createdAt
                }
            }
            """

            response = requests.post(
                graphql_url,
                json={'query': repo_query, 'variables': {
                    'owner': self.owner,
                    'repo': self.repo_name
                }},
                headers=self.graphql_headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if 'errors' not in data:
                    self.repo_created_at = data['data']['repository']['createdAt']
                    self.api_creation_year = datetime.fromisoformat(
                        self.repo_created_at.replace('Z', '+00:00')
                    ).year
                    print(f"  API repo creation year: {self.api_creation_year}")
        except Exception as e:
            print(f"  Warning: Failed to get API creation info: {e}")
            self.api_creation_year = self.current_year

    def get_year_range(self):
        """Return year range for iteration"""
        return range(self.current_year, self.creation_year - 1, -1)

    def get_api_year_range(self):
        """Return API year range for module 4"""
        return range(self.current_year, self.api_creation_year - 1, -1)


def collect_metadata(repo_path: str, check_point_date: Tuple[int, int], default_branch: str,
                     ctx: 'RepoContext' = None, start_year: int | None = None,
                     end_year: int | None = None) -> Dict[str, list]:
    """
    Collect development and maintenance metadata from local Git repository

    Args:
        repo_path: Local repository path
        check_point_date: Checkpoint date (month, day)
        default_branch: Default branch name
        ctx: RepoContext object (optional, for optimization)

    Returns:
        Dictionary containing yearly statistics
    """
    month, day = check_point_date

    # Use ctx if provided, otherwise fallback to original behavior
    if ctx:
        current_year = ctx.current_year
        creation_year = ctx.creation_year
        first_commit = ctx.first_commit
    else:
        current_year = datetime.now(timezone.utc).year
        first_commit = _run_git_command(
            f'git log --reverse --format=%cd --date=iso-strict {default_branch} -- | head -n 1',
            repo_path
        )
        creation_year = int(first_commit.split('-')[0])

    # Normalize year range [min_year, max_year]
    if start_year is None:
        start_year = creation_year
    if end_year is None:
        end_year = current_year
    min_year = max(creation_year, start_year)
    max_year = min(current_year, end_year)

    # Statistics grouped by year
    yearly_stats = defaultdict(lambda: {
        'commits': 0,
        'additions': 0,
        'deletions': 0,
        'commit_dates': []
    })

    # Get all commits statistics at once
    log_output = _run_git_command(
        f'git log --format=%cd --date=iso-strict --numstat {default_branch} --',
        repo_path
    )

    lines = log_output.split('\n')
    current_date = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '\t' not in line:
            # Date line
            current_date = _parse_git_datetime(line.split()[0])
            # Determine year cycle
            if current_date.month > month or (current_date.month == month and current_date.day >= day):
                year = current_date.year + 1
            else:
                year = current_date.year

            yearly_stats[year]['commits'] += 1
            yearly_stats[year]['commit_dates'].append(current_date)
        else:
            # Stats line: additions deletions filename
            parts = line.split('\t')
            if len(parts) >= 2 and current_date:
                try:
                    add = int(parts[0]) if parts[0] != '-' else 0
                    dele = int(parts[1]) if parts[1] != '-' else 0
                    if current_date.month > month or (current_date.month == month and current_date.day >= day):
                        year = current_date.year + 1
                    else:
                        year = current_date.year
                    yearly_stats[year]['additions'] += add
                    yearly_stats[year]['deletions'] += dele
                except:
                    pass

    result = {
        'additions_per_year': [],
        'deletions_per_year': [],
        'commit_interval_mean_per_year': [],
        'commit_interval_std_per_year': [],
        'burstiness_per_year': [],
        'commit_density_per_year': [],
        'total_loc_start_per_year': [],
        'total_loc_snapshot_per_year': [],
        'years': []
    }

    # Step 1: Get accurate LOC at each year's checkpoint date using git commands
    # Instead of accumulating additions/deletions, we directly query LOC at specific timepoints
    # Optimization: loc_start[year] = loc_end[year-1], so we only need to query once per year
    loc_start_by_year = {}
    loc_end_by_year = {}

    # Get LOC at creation year start (first commit)
    try:
        first_commit_hash = _run_git_command(
            f'git rev-list --max-parents=0 {default_branch} -- | tail -n 1',
            repo_path
        )
        if first_commit_hash:
            creation_loc = _estimate_total_loc_at_commit(repo_path, first_commit_hash)
        else:
            creation_loc = 0
    except Exception:
        creation_loc = 0

    loc_start_by_year[creation_year] = creation_loc

    # For each year, get LOC at year end checkpoint (year, month, day)
    # This becomes the next year's start LOC
    for year in range(creation_year, max_year + 1):
        end_date = f"{year}-{month:02d}-{day:02d}"

        try:
            end_commit_hash = _run_git_command(
                f'git rev-list -n 1 --before="{end_date}" {default_branch} --',
                repo_path
            )
            if end_commit_hash:
                loc_end = _estimate_total_loc_at_commit(repo_path, end_commit_hash)
            else:
                loc_end = 0
        except Exception:
            loc_end = 0

        loc_end_by_year[year] = loc_end

        # Next year's start is this year's end
        if year + 1 <= max_year:
            loc_start_by_year[year + 1] = loc_end

    # Step 2: Compute detailed stats and output ONLY for years in [min_year, max_year]
    for year in range(max_year, min_year - 1, -1):
        stats = yearly_stats.get(year, {
            'commits': 0,
            'additions': 0,
            'deletions': 0,
            'commit_dates': []
        })

        # Commit interval stats for burstiness
        commit_dates = sorted(stats.get('commit_dates', []))
        intervals = []
        for i in range(1, len(commit_dates)):
            delta = (commit_dates[i] - commit_dates[i-1]).total_seconds() / 86400
            intervals.append(max(delta, 0.0))
        if intervals and len(intervals)>=30:# over 30 commits per year
            interval_mean = statistics.mean(intervals)
            interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
            burstiness = (interval_std - interval_mean) / (interval_std + interval_mean) if (interval_std + interval_mean) != 0 else 0.0
        else:
            interval_mean = 0.0
            interval_std = 0.0
            burstiness = -2

        # Commit density: commits / weeks in the year window
        period_start = datetime(year-1, month, day, tzinfo=timezone.utc)
        period_end = datetime(year, month, day, tzinfo=timezone.utc)
        weeks = max((period_end - period_start).days / 7, 1)
        annual_commit=stats.get('commits', 0)
        commit_density = stats.get('commits', 0) / weeks# Commit Frequency

        result['years'].append(year)
        result['additions_per_year'].append(stats.get('additions', 0))
        result['deletions_per_year'].append(stats.get('deletions', 0))
        result['commit_interval_mean_per_year'].append(round(interval_mean, 2))
        result['commit_interval_std_per_year'].append(round(interval_std, 2))
        result['burstiness_per_year'].append(round(burstiness, 4))
        result['commit_density_per_year'].append(round(commit_density, 4))
        result['total_loc_start_per_year'].append(loc_start_by_year.get(year, 0))
        result['total_loc_snapshot_per_year'].append(loc_end_by_year.get(year, 0))

    return result


def _count_functions_in_file(file_content: str, filepath: str) -> int:
    """Count functions in a file based on programming language"""
    import re

    ext = filepath.split('.')[-1].lower() if '.' in filepath else ''
    count = 0

    if ext in ['py']:
        # Python: def function_name(
        count = len([line for line in file_content.split('\n') if line.strip().startswith('def ')])
    elif ext in ['js', 'ts', 'jsx', 'tsx']:
        # JavaScript/TypeScript: function name() or const name = () or name()
        count = len(re.findall(r'\bfunction\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\(', file_content, re.MULTILINE))
    elif ext in ['java', 'cs', 'cpp', 'cc', 'c', 'h', 'hpp']:
        # Java/C#/C++: returnType functionName(args) {
        count = len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*\{', file_content))
    elif ext in ['go']:
        # Go: func functionName(
        count = len(re.findall(r'\bfunc\s+\w+\s*\(', file_content))
    elif ext in ['rb']:
        # Ruby: def function_name
        count = len([line for line in file_content.split('\n') if line.strip().startswith('def ')])
    elif ext in ['php']:
        # PHP: function functionName(
        count = len(re.findall(r'\bfunction\s+\w+\s*\(', file_content))
    elif ext in ['r']:
        # R: functionName <- function( or functionName = function(
        count = len(re.findall(r'\w+\s*<-\s*function\s*\(|\w+\s*=\s*function\s*\(', file_content, re.IGNORECASE))
    elif ext in ['rs']:
        # Rust: fn function_name( or pub fn function_name( or async fn function_name(
        count = len(re.findall(r'\b(?:pub\s+)?(?:async\s+)?fn\s+\w+\s*[<(]', file_content))
    return count


def collect_test_info(repo_path: str, check_point_date: Tuple[int, int], default_branch: str,
                      ctx: 'RepoContext' = None, start_year: int | None = None,
                      end_year: int | None = None) -> Dict[str, list]:
    """
    Collect test file information from repository

    Args:
        repo_path: Local repository path
        check_point_date: Checkpoint date (month, day)
        default_branch: Default branch name
        ctx: RepoContext object (optional, for optimization)

    Returns:
        Dictionary containing yearly test file statistics
    """
    month, day = check_point_date

    # Use ctx if provided, otherwise fallback to original behavior
    if ctx:
        current_year = ctx.current_year
        creation_year = ctx.creation_year
        first_commit = ctx.first_commit
    else:
        current_year = datetime.now(timezone.utc).year
        first_commit = _run_git_command(
            f'git log --reverse --format=%cd --date=iso-strict {default_branch} -- | head -n 1',
            repo_path
        )
        creation_year = int(first_commit.split('-')[0])

    if start_year is None:
        start_year = creation_year
    if end_year is None:
        end_year = current_year
    start_year = max(start_year, creation_year)
    end_year = min(end_year, current_year)

    result = {
        'test_files_count_per_year': [],
        'test_lines_count_per_year': [],
        'test_lines_added_per_year': [],
        'test_lines_deleted_per_year': [],
        'test_lines_start_per_year': [],
        'years': []
    }

    for year in range(end_year, creation_year - 1, -1):
        # Skip years outside the requested range early
        if year < start_year or year > end_year:
            continue

        if year == creation_year:
            year_date = first_commit.split()[0]
        else:
            year_date = f"{year}-{month:02d}-{day:02d}"

        # Get commit hash at checkpoint date
        commit_hash = _run_git_command(
            f'git rev-list -n 1 --before="{year_date}" {default_branch} --',
            repo_path
        )

        if not commit_hash:
            result['years'].append(year)
            result['test_files_count_per_year'].append(0)
            result['test_lines_count_per_year'].append(0)
            result['test_lines_added_per_year'].append(0)
            result['test_lines_deleted_per_year'].append(0)
            continue

        # Get all test files at this commit (case-insensitive search for "test")
        try:
            test_files_output = _run_git_command(
                f'git ls-tree -r --name-only {commit_hash} | grep -i test',
                repo_path
            )
        except Exception as e:
            # No test files found
            print('No test files found.')
            result['years'].append(year)
            result['test_files_count_per_year'].append(0)
            result['test_lines_count_per_year'].append(0)
            result['test_lines_added_per_year'].append(0)
            result['test_lines_deleted_per_year'].append(0)
            continue

        test_files = [f for f in test_files_output.split('\n') if f.strip()]
        test_files_count = len(test_files)

        # Count lines in test files
        total_lines = 0

        for test_file in test_files:
            # Skip binary files by checking file extension
            binary_exts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.pdf',
                            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib',
                            '.class', '.jar', '.war', '.pyc', '.pyo', '.whl']
            if any(test_file.lower().endswith(ext) for ext in binary_exts):
                continue

            try:
                # First check if it's a blob (file) and not a tree (directory)
                obj_type = _run_git_command(f'git cat-file -t {commit_hash}:{test_file}', repo_path)
                if obj_type.strip() != 'blob':
                    # Skip directories and other non-file objects
                    continue

                file_content = _run_git_command(f'git show {commit_hash}:{test_file}', repo_path)

                # Skip if content appears to be binary (contains null bytes)
                if '\x00' in file_content:
                    continue
            except Exception as e:
                # Skip if any git command fails (e.g., submodules, invalid paths)
                continue

            lines = len([l for l in file_content.split('\n') if l.strip()])
            total_lines += lines

        # Calculate yearly additions/deletions for test files
        if year == creation_year:
            since_date = first_commit.split()[0]
        else:
            since_date = f"{year-1}-{month:02d}-{day:02d}"

        # Get test file changes in this year
        log_output = _run_git_command(
            f'git log --since="{since_date}" --until="{year_date}" --numstat {default_branch} -- "*test*" "*Test*"',
            repo_path
        )

        test_additions = 0
        test_deletions = 0

        for line in log_output.split('\n'):
            if not line.strip() or '\t' not in line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    add = int(parts[0]) if parts[0] != '-' else 0
                    dele = int(parts[1]) if parts[1] != '-' else 0
                    test_additions += add
                    test_deletions += dele
                except:
                    pass

        result['years'].append(year)
        result['test_files_count_per_year'].append(test_files_count)
        result['test_lines_count_per_year'].append(total_lines)
        result['test_lines_added_per_year'].append(test_additions)
        result['test_lines_deleted_per_year'].append(test_deletions)

    # Derive start-of-year test LOC by shifting next year's snapshot (first year uses itself)
    start_lines = []
    for i in range(len(result['years'])):
        if i + 1 < len(result['test_lines_count_per_year']):
            start_lines.append(result['test_lines_count_per_year'][i+1])
        else:
            start_lines.append(result['test_lines_count_per_year'][i])
    result['test_lines_start_per_year'] = start_lines

    return result


def _is_bot_developer(author_name: str, author_email: str) -> bool:
    """Check if developer is a bot or automation tool"""
    bot_keywords = ['bot', 'auto', 'claude', 'copilot', 'github-actions', 'dependabot',
                     'renovate', 'greenkeeper', 'mergify', 'semantic-release']
    combined = f"{author_name} {author_email}".lower()
    return any(keyword in combined for keyword in bot_keywords)


def _calculate_truck_factor(commit_counts: Dict[str, int]) -> int:
    """
    Calculate Bus Factor - minimum number of developers who made 50% of commits

    Args:
        commit_counts: Dictionary mapping developer names to commit counts

    Returns:
        Bus factor (number of developers)
    """
    if not commit_counts:
        return 0

    # Sort developers by commit count (descending)
    sorted_devs = sorted(commit_counts.items(), key=lambda x: x[1], reverse=True)
    total_commits = sum(commit_counts.values())
    threshold = total_commits * 0.5

    cumulative = 0
    truck_factor = 0

    for dev, count in sorted_devs:
        cumulative += count
        truck_factor += 1
        if cumulative >= threshold:
            break

    return truck_factor


def _core_developers_percentile(commit_counts: Dict[str, int], percentile: float = 0.8) -> list[str]:
    """Return developers whose commit count is at/above the given percentile threshold."""
    if not commit_counts:
        return []
    counts = sorted(commit_counts.values())
    idx = max(0, math.ceil(len(counts) * percentile) - 1)
    threshold = counts[idx]
    return [dev for dev, cnt in commit_counts.items() if cnt >= threshold]


def collect_developer_info(repo_path: str, check_point_date: Tuple[int, int], default_branch: str,
                           metadata_result: Dict[str, list], ctx: 'RepoContext' = None,
                           start_year: int | None = None,
                           end_year: int | None = None) -> Dict[str, Any]:
    """
    Collect developer information from repository

    Args:
        repo_path: Local repository path
        check_point_date: Checkpoint date (month, day)
        default_branch: Default branch name
        metadata_result: Result from collect_metadata containing yearly commit stats
        ctx: RepoContext object (optional, for optimization)

    Returns:
        Dictionary containing yearly developer statistics
    """
    month, day = check_point_date

    # Use ctx if provided, otherwise fallback to original behavior
    if ctx:
        current_year = ctx.current_year
        creation_year = ctx.creation_year
        first_commit = ctx.first_commit
    else:
        current_year = datetime.now(timezone.utc).year
        first_commit = _run_git_command(
            f'git log --reverse --format=%cd --date=iso-strict {default_branch} -- | head -n 1',
            repo_path
        )
        creation_year = int(first_commit.split('-')[0])

    # Get all commits with author info (reverse order: oldest first, to track first appearance correctly)
    log_output = _run_git_command(
        f'git log --reverse --format="%cd|%an|%ae" --date=iso-strict {default_branch} --',
        repo_path
    )

    # Set effective year bounds (prefer metadata years if provided)
    md_years = metadata_result.get('years', [])
    if start_year is None and md_years:
        start_year = min(md_years)
    if end_year is None and md_years:
        end_year = max(md_years)
    if start_year is None:
        start_year = creation_year
    if end_year is None:
        end_year = current_year
    start_year = max(start_year, creation_year)
    end_year = min(end_year, current_year)

    # Parse all commits
    all_developers = set()
    yearly_commits = defaultdict(lambda: defaultdict(int))
    developer_first_year = {}
    total_commit_counts = defaultdict(int)

    for line in log_output.split('\n'):
        if not line.strip():
            continue

        parts = line.split('|')
        if len(parts) < 3:
            continue

        try:
            commit_date = _parse_git_datetime(parts[0].split()[0])
        except Exception as e:
            print(e)
            continue
        author_name = parts[1]
        author_email = parts[2]

        # Skip bot developers
        if _is_bot_developer(author_name, author_email):
            continue

        # Determine year cycle
        if commit_date.month > month or (commit_date.month == month and commit_date.day >= day):
            year = commit_date.year + 1
        else:
            year = commit_date.year

        dev_id = f"{author_name}|{author_email}"
        all_developers.add(dev_id)
        yearly_commits[year][dev_id] += 1
        total_commit_counts[dev_id] += 1

        # Track first appearance
        if dev_id not in developer_first_year:
            developer_first_year[dev_id] = year

    # Identify core developers across selected years (80th percentile per year bucket)
    # Cache core developers for each year to avoid redundant computation
    core_devs_by_year = {}
    core_dev_candidates = set()
    for y in yearly_commits.keys():
        if y < start_year or y > end_year:
            continue
        core_devs_this_year = set(_core_developers_percentile(yearly_commits[y], percentile=0.8))
        core_devs_by_year[y] = core_devs_this_year
        core_dev_candidates.update(core_devs_this_year)

    # Build yearly statistics
    result = {
        'new_developers_per_year': [],
        'truck_factor_per_year': [],
        'core_developers_per_year': [],
        'core_dev_turnover_per_year': [],
        'newcomer_retention_next_year_per_year': [],
        'core_dev_over_50_ratio_per_year': [],
        'years': []
    }

    # Now iterate in reverse order for output
    for year in range(end_year, start_year - 1, -1):
        # New developers in this year
        new_devs = [dev for dev, first_year in developer_first_year.items() if first_year == year]
        new_devs_count = len(new_devs)

        # Active developers in this year
        active_devs = list(yearly_commits[year].keys())

        # Bus factor for this year
        truck_factor = _calculate_truck_factor(yearly_commits[year])

        # Core developers: use cached result to avoid redundant computation
        # https://ieeexplore.ieee.org/abstract/document/7985659/
        core_devs = core_devs_by_year.get(year, set())
        # Turnover: proportion of previous year's core developers who left this year
        # Note: Cannot calculate turnover for start_year (no data for year-1)
        if year > start_year:
            prev_year_core = core_devs_by_year.get(year - 1, set())
            turnover = (len(prev_year_core - core_devs) / len(prev_year_core)) if prev_year_core else 0
        else:
            turnover = None

        # Newcomer retention: newcomers of previous year that are active in this year
        # Note: Cannot calculate retention for start_year (no data for year-1)
        if year > start_year:
            prev_year_new_devs = [dev for dev, first_year in developer_first_year.items() if first_year == year - 1]
            this_year_active = set(yearly_commits.get(year, {}).keys())
            retained_newcomers = len([d for d in prev_year_new_devs if d in this_year_active])
            prev_year_new_devs_count = len(prev_year_new_devs)
            newcomer_retention = retained_newcomers / prev_year_new_devs_count if prev_year_new_devs_count > 0 else 0
        else:
            # For start_year, we cannot calculate retention (no data for year-1)
            newcomer_retention = None

        # Core dev over-50 ratio (disabled: age estimation no longer performed)
        core_over_50_ratio = None

        # Append results
        result['years'].append(year)
        result['new_developers_per_year'].append(new_devs_count)
        result['truck_factor_per_year'].append(truck_factor)
        result['core_developers_per_year'].append(list(core_devs))
        if turnover is None:
            result['core_dev_turnover_per_year'].append(None)
        else:
            result['core_dev_turnover_per_year'].append(round(turnover* 100, 4))
        if newcomer_retention is None:
            result['newcomer_retention_next_year_per_year'].append(None)
        else:
            result['newcomer_retention_next_year_per_year'].append(round(newcomer_retention, 4))
        if core_over_50_ratio is None:
            result['core_dev_over_50_ratio_per_year'].append(None)
        else:
            result['core_dev_over_50_ratio_per_year'].append(round(core_over_50_ratio* 100, 4))

    # Overall statistics
    result['total_developers'] = len(all_developers)
    result['significant_developers_count'] = len(core_dev_candidates)

    return result



def _gh_headers(github_token: str | None) -> dict:
    h = {'Accept': 'application/vnd.github.v3+json'}
    if github_token:
        h['Authorization'] = f'token {github_token}'
    return h

def _looks_like_display_name(name: str) -> bool:
    # GitHub login rules: ^[A-Za-z0-9-]+$, cannot start/end with -, no consecutive --
    # We do rough filtering: if it has spaces or non-alphanumeric/hyphen chars, treat as display name
    return any(c.isspace() for c in name) or not all(c.isalnum() or c == '-' for c in name)

def _fetch_user_by_login(login: str, github_token: str | None, timeout=8):
    url = f"{GITHUB_API}/users/{quote(login)}"
    r = requests.get(url, headers=_gh_headers(github_token), timeout=timeout)
    if r.status_code == 200:
        return r.json()
    return None

def _best_candidate_from_search(items: list[dict], display_name: str) -> str | None:
    """
    Pick a login from search results:
    1) Exact name match (case-insensitive, ignoring extra spaces)
    2) Otherwise pick highest score
    """
    def norm(s: str) -> str:
        return " ".join(s.split()).casefold()

    target = norm(display_name)
    exact = [it for it in items if it.get('login') and it.get('score') is not None]
    # If result includes 'name' field (sometimes search/users doesn't), prefer exact match; otherwise use score
    exact_name_hits = []
    for it in exact:
        # Try to supplement 'name' via /users/{login} for strict matching
        it['__name_norm'] = None
    for it in exact[:8]:  # Check at most 8 candidates to avoid excessive requests
        # Lazy fetch candidate profile to check name
        prof = _fetch_user_by_login(it['login'], github_token=None)
        if prof and prof.get('name'):
            it['__name_norm'] = norm(prof['name'])
            if it['__name_norm'] == target:
                exact_name_hits.append(it)

    if exact_name_hits:
        # Multiple exact matches: pick the one with more followers (more likely "main" account)
        return sorted(exact_name_hits, key=lambda x: x.get('followers', 0), reverse=True)[0]['login']

    # Otherwise pick by highest score
    if exact:
        return sorted(exact, key=lambda x: x['score'], reverse=True)[0]['login']
    return None

def _resolve_login_from_display_name(display_name: str, github_token: str | None, timeout=10) -> str | None:
    # Search for display name using search/users?q=... in:name
    q = f"{display_name} in:name"
    url = f"{GITHUB_API}/search/users"
    r = requests.get(url, params={'q': q}, headers=_gh_headers(github_token), timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    items = data.get('items') or []
    if not items:
        return None
    return _best_candidate_from_search(items, display_name)

def _batch_fetch_users_by_logins_graphql(logins: List[str], github_token: str | None = None, timeout: int = 30, batch_size: int = 50) -> Dict[str, dict]:
    """
    Batch fetch multiple GitHub user profiles using GraphQL API (chunked into smaller batches)

    Args:
        logins: List of GitHub usernames
        github_token: GitHub API token
        timeout: Request timeout
        batch_size: Number of logins per GraphQL request (default 50 to stay within rate limits)

    Returns:
        Dict mapping login to profile dict
    """
    if not logins:
        return {}

    all_results = {}

    # Split logins into chunks
    for chunk_start in range(0, len(logins), batch_size):
        chunk_logins = logins[chunk_start:chunk_start + batch_size]

        # Build GraphQL query with aliases for each user
        queries = []
        for i, login in enumerate(chunk_logins):
            # Escape special characters
            escaped_login = login.replace('"', '\\"')
            queries.append(f'''
            user{i}: user(login: "{escaped_login}") {{
                login
                id
                name
                email
                avatarUrl
                bio
                location
                company
                createdAt
                publicRepos: repositories {{ totalCount }}
                followers {{ totalCount }}
            }}
            ''')

        graphql_query = "query { " + " ".join(queries) + " }"

        try:
            url = "https://api.github.com/graphql"
            headers = _gh_headers(github_token)
            headers['Content-Type'] = 'application/json'

            response = requests.post(
                url,
                json={'query': graphql_query},
                headers=headers,
                timeout=timeout
            )

            if response.status_code != 200:
                print(f"Warning: GraphQL batch fetch chunk {chunk_start//batch_size + 1} failed with status {response.status_code}")
                continue

            data = response.json()

            # Process results
            for i, login in enumerate(chunk_logins):
                user_data = data.get('data', {}).get(f'user{i}') if 'data' in data else None
                if user_data:
                    all_results[login] = {
                        'login': user_data.get('login'),
                        'id': user_data.get('id'),
                        'name': user_data.get('name'),
                        'email': user_data.get('email'),
                        'avatar_url': user_data.get('avatarUrl'),
                        'bio': user_data.get('bio'),
                        'location': user_data.get('location'),
                        'company': user_data.get('company'),
                        'created_at': user_data.get('createdAt'),
                        'public_repos': user_data.get('publicRepos', {}).get('totalCount', 0),
                        'followers': user_data.get('followers', {}).get('totalCount', 0)
                    }

        except Exception as e:
            print(f"Warning: Failed to process login chunk {chunk_start//batch_size + 1}: {e}")
            continue

    print(f"Login fetch: {len(logins)} queried, {len(all_results)} found")
    return all_results


def _batch_search_users_by_emails(emails: List[str], github_token: str | None = None, timeout: int = 30, batch_size: int = 50) -> Dict[str, dict]:
    """
    Batch search GitHub users by multiple emails using GraphQL (chunked into smaller batches)

    Args:
        emails: List of email addresses
        github_token: GitHub API token
        timeout: Request timeout
        batch_size: Number of emails per GraphQL request (default 50 to avoid rate limits)

    Returns:
        Dict mapping email to profile dict (first match for each email)
    """
    if not emails:
        return {}

    all_results = {}

    # Split emails into chunks to avoid GraphQL complexity limits
    for chunk_start in range(0, len(emails), batch_size):
        chunk_emails = emails[chunk_start:chunk_start + batch_size]

        # Build GraphQL query for this chunk
        queries = []
        for i, email in enumerate(chunk_emails):
            # Escape special characters in email for GraphQL
            escaped_email = email.replace('"', '\\"').replace('\\', '\\\\')
            queries.append(f'''
            search{i}: search(query: "{escaped_email} in:email", type: USER, first: 1) {{
                edges {{
                    node {{
                        ... on User {{
                            login
                            id
                            name
                            email
                            avatarUrl
                            bio
                            location
                            company
                            createdAt
                            publicRepos: repositories {{ totalCount }}
                            followers {{ totalCount }}
                        }}
                    }}
                }}
            }}
            ''')

        graphql_query = "query { " + " ".join(queries) + " }"

        try:
            url = "https://api.github.com/graphql"
            headers = _gh_headers(github_token)
            headers['Content-Type'] = 'application/json'

            response = requests.post(
                url,
                json={'query': graphql_query},
                headers=headers,
                timeout=timeout
            )

            if response.status_code != 200:
                print(f"Warning: GraphQL batch search chunk {chunk_start//batch_size + 1} failed with status {response.status_code}")
                if response.status_code == 403:
                    print(f"  Response: {response.text[:200]}")
                continue

            data = response.json()

            # Process results
            for i, email in enumerate(chunk_emails):
                search_result = data.get('data', {}).get(f'search{i}') if 'data' in data else None
                if not search_result:
                    continue

                edges = search_result.get('edges', [])

                if edges and len(edges) > 0:
                    user_data = edges[0].get('node', {})
                    if user_data:
                        # Convert GraphQL format to REST API format
                        all_results[email] = {
                            'login': user_data.get('login'),
                            'id': user_data.get('id'),
                            'name': user_data.get('name'),
                            'email': user_data.get('email'),
                            'avatar_url': user_data.get('avatarUrl'),
                            'bio': user_data.get('bio'),
                            'location': user_data.get('location'),
                            'company': user_data.get('company'),
                            'created_at': user_data.get('createdAt'),
                            'public_repos': user_data.get('publicRepos', {}).get('totalCount', 0),
                            'followers': user_data.get('followers', {}).get('totalCount', 0)
                        }

        except Exception as e:
            print(f"Warning: Failed to process email chunk {chunk_start//batch_size + 1}: {e}")
            continue

    print(f"Email search: {len(emails)} queried, {len(all_results)} found")
    return all_results


def _search_user_by_email(email: str, github_token: str | None = None, timeout: int = 10) -> dict | None:
    """
    Search GitHub user by email address

    Args:
        email: Email address
        github_token: GitHub API token
        timeout: Request timeout

    Returns:
        User profile dict or None
    """
    try:
        # Use GitHub search API with email query
        q = f"{email} in:email"
        url = f"{GITHUB_API}/search/users"
        r = requests.get(url, params={'q': q}, headers=_gh_headers(github_token), timeout=timeout)

        if r.status_code != 200:
            return None

        data = r.json() or {}
        items = data.get('items') or []

        if not items:
            return None

        # Get the first (most relevant) user's full profile
        login = items[0].get('login')
        if login:
            return _fetch_user_by_login(login, github_token)

        return None
    except Exception as e:
        print(f"Warning: Failed to search user by email {email}: {e}")
        return None


def _batch_search_users_by_names(names: List[str], github_token: str | None = None, timeout: int = 30, batch_size: int = 50) -> Dict[str, dict]:
    """
    Batch search GitHub users by display names using GraphQL (chunked into smaller batches)

    Args:
        names: List of display names
        github_token: GitHub API token
        timeout: Request timeout
        batch_size: Number of names per GraphQL request (default 30 due to search complexity)

    Returns:
        Dict mapping display name to profile dict (first match for each name)
    """
    if not names:
        return {}

    all_results = {}

    # Split names into chunks (smaller batch size because search is more complex)
    for chunk_start in range(0, len(names), batch_size):
        chunk_names = names[chunk_start:chunk_start + batch_size]

        # Build GraphQL query for name searches
        queries = []
        for i, name in enumerate(chunk_names):
            # Escape special characters
            escaped_name = name.replace('"', '\\"').replace('\\', '\\\\')
            queries.append(f'''
            search{i}: search(query: "{escaped_name} in:name", type: USER, first: 1) {{
                edges {{
                    node {{
                        ... on User {{
                            login
                            id
                            name
                            email
                            avatarUrl
                            bio
                            location
                            company
                            createdAt
                            publicRepos: repositories {{ totalCount }}
                            followers {{ totalCount }}
                        }}
                    }}
                }}
            }}
            ''')

        graphql_query = "query { " + " ".join(queries) + " }"

        try:
            url = "https://api.github.com/graphql"
            headers = _gh_headers(github_token)
            headers['Content-Type'] = 'application/json'

            response = requests.post(
                url,
                json={'query': graphql_query},
                headers=headers,
                timeout=timeout
            )

            if response.status_code != 200:
                print(f"Warning: GraphQL name search chunk {chunk_start//batch_size + 1} failed with status {response.status_code}")
                continue

            data = response.json()

            # Process results
            for i, name in enumerate(chunk_names):
                search_result = data.get('data', {}).get(f'search{i}') if 'data' in data else None
                if not search_result:
                    continue

                edges = search_result.get('edges', [])

                if edges and len(edges) > 0:
                    user_data = edges[0].get('node', {})
                    if user_data:
                        # Convert GraphQL format to REST API format
                        all_results[name] = {
                            'login': user_data.get('login'),
                            'id': user_data.get('id'),
                            'name': user_data.get('name'),
                            'email': user_data.get('email'),
                            'avatar_url': user_data.get('avatarUrl'),
                            'bio': user_data.get('bio'),
                            'location': user_data.get('location'),
                            'company': user_data.get('company'),
                            'created_at': user_data.get('createdAt'),
                            'public_repos': user_data.get('publicRepos', {}).get('totalCount', 0),
                            'followers': user_data.get('followers', {}).get('totalCount', 0)
                        }

        except Exception as e:
            print(f"Warning: Failed to process name chunk {chunk_start//batch_size + 1}: {e}")
            continue

    print(f"Name search: {len(names)} queried, {len(all_results)} found")
    return all_results


def get_github_profiles_batch(developers: List[Dict[str, str]], github_token: str | None = None) -> Dict[str, dict]:
    """
    Batch fetch GitHub profiles for multiple developers using GraphQL (efficient, single request)

    Args:
        developers: List of dicts with keys: 'dev_id', 'author_name', 'author_email'
        github_token: GitHub API token

    Returns:
        Dict mapping dev_id to profile dict (or None if not found)
    """
    if not developers:
        return {}

    results = {}

    # Step 1: Batch search by emails (single GraphQL request)
    emails_to_dev_ids = {}  # email -> list of dev_ids
    for dev in developers:
        email = dev['author_email']
        if email and '@' in email:
            if email not in emails_to_dev_ids:
                emails_to_dev_ids[email] = []
            emails_to_dev_ids[email].append(dev['dev_id'])

    if emails_to_dev_ids:
        email_results = _batch_search_users_by_emails(list(emails_to_dev_ids.keys()), github_token)
        for email, profile in email_results.items():
            for dev_id in emails_to_dev_ids[email]:
                results[dev_id] = profile

    # Step 2: For remaining developers without email match, try batch fetch by login
    logins_to_dev_ids = {}  # login -> dev_id
    for dev in developers:
        dev_id = dev['dev_id']
        if dev_id in results:  # Already found by email
            continue

        author_name = dev['author_name']
        # Only try as login if it doesn't look like a display name
        if not _looks_like_display_name(author_name):
            logins_to_dev_ids[author_name] = dev_id

    if logins_to_dev_ids:
        login_results = _batch_fetch_users_by_logins_graphql(list(logins_to_dev_ids.keys()), github_token)
        for login, profile in login_results.items():
            dev_id = logins_to_dev_ids[login]
            results[dev_id] = profile

    # Step 3: For still not found, batch search by display names
    names_to_dev_ids = {}  # name -> dev_id
    for dev in developers:
        dev_id = dev['dev_id']
        if dev_id in results:  # Already found
            continue

        author_name = dev['author_name']
        names_to_dev_ids[author_name] = dev_id

    if names_to_dev_ids:
        name_results = _batch_search_users_by_names(list(names_to_dev_ids.keys()), github_token)
        for name, profile in name_results.items():
            dev_id = names_to_dev_ids[name]
            results[dev_id] = profile

    # Step 4: Mark remaining as None
    for dev in developers:
        dev_id = dev['dev_id']
        if dev_id not in results:
            results[dev_id] = None

    return results


def get_github_profile(author_name: str, github_token: str | None = None) -> dict | None:
    """
    Get GitHub profile by author name.
    author_name can be either a login or a display name (with spaces).
    Returns profile JSON (with avatar_url/name/id etc.), or None on failure.
    """
    # 1) First try treating it as a login directly
    profile = None
    if not _looks_like_display_name(author_name):
        profile = _fetch_user_by_login(author_name, github_token)
        if profile:  # Hit, return immediately
            return profile

    # 2) Treat as display name -> search to get login -> then fetch profile
    login = _resolve_login_from_display_name(author_name, github_token)
    if login:
        profile = _fetch_user_by_login(login, github_token)
        if profile:
            return profile

    return None


def nvd_search(keyword: str, api_key: str = NVD_TOKEN, delay: int = 1) -> list:
    """
    Search CVE vulnerabilities from NVD database using nvdlib

    Args:
        keyword: Search keyword
        api_key: NVD API key
        delay: Delay between requests in seconds

    Returns:
        List of dictionaries containing:
        [
            {
                'cve_id': 'CVE-2024-0001',
                'published': '2024-01-15',
                'cvss': ['v3.1', 7.5]  # [version, score]
            },
            ...
        ]
    """

    # Search CVE using nvdlib
    r = nvdlib.searchCVE(keywordSearch=keyword, key=api_key, delay=delay)

    results = []
    for cve in r:
        # Extract CVE ID
        cve_id = cve.id

        # Extract published date
        published = getattr(cve, 'published', None)
        if published:
            published = str(published).split('T')[0]  # Extract date part (YYYY-MM-DD)

        # Extract CVSS score with version priority: v3.1 > v3.0 > v2.0
        cvss = None

        # Method 1: Try v31score, v30score, v2score attributes
        if hasattr(cve, 'v31score') and cve.v31score is not None:
            cvss = ['v3.1', cve.v31score]
        elif hasattr(cve, 'v30score') and cve.v30score is not None:
            cvss = ['v3.0', cve.v30score]
        elif hasattr(cve, 'v2score') and cve.v2score is not None:
            cvss = ['v2.0', cve.v2score]
        # Method 2: Try accessing metrics attribute (newer nvdlib)
        elif hasattr(cve, 'metrics') and cve.metrics:
            metrics = cve.metrics
            if hasattr(metrics, 'cvssMetricV31') and metrics.cvssMetricV31:
                cvss = ['v3.1', metrics.cvssMetricV31[0].cvssData.baseScore]
            elif hasattr(metrics, 'cvssMetricV30') and metrics.cvssMetricV30:
                cvss = ['v3.0', metrics.cvssMetricV30[0].cvssData.baseScore]
            elif hasattr(metrics, 'cvssMetricV2') and metrics.cvssMetricV2:
                cvss = ['v2.0', metrics.cvssMetricV2[0].cvssData.baseScore]
        # Method 3: Try score attribute
        elif hasattr(cve, 'score') and cve.score:
            score_data = cve.score
            if isinstance(score_data, list) and len(score_data) > 0:
                cvss_info = score_data[0]
                if hasattr(cvss_info, 'baseScore') and hasattr(cvss_info, 'version'):
                    cvss = [cvss_info.version, cvss_info.baseScore]
        results.append({
            'cve_id': cve_id,
            'published': published,
            'cvss': cvss
        })

    return results


def collect_issue_vulnerability_info(repo_url: str, check_point_date: Tuple[int, int], token: str,
                                     nvd_api_key: str = None, ctx: 'RepoContext' = None,
                                     start_year: int | None = None, end_year: int | None = None) -> Dict[str, Any]:
    """
    Collect issue and vulnerability information using GraphQL aggregation

    Args:
        repo_url: GitHub repository URL
        check_point_date: Checkpoint date (month, day)
        token: GitHub Personal Access Token
        nvd_api_key: NVD API key for CVE search
        ctx: RepoContext object (optional, for optimization)

    Returns:
        Dictionary containing yearly statistics:
        - issues_per_year: Total issues submitted per year
        - unresolved_issues_per_year: Issues unresolved by checkpoint date
        - unresolved_issues_ratio_per_year: Ratio of unresolved issues
        - avg_issue_resolution_days_per_year: Average days to close issues
        - cve_count_per_year: CVE vulnerabilities reported per year
        - avg_cvss_per_year: Average CVSS score per year
        - years: List of years
        - historical_avg_cve_count: Historical average CVE count per year
        - historical_avg_cvss: Historical average CVSS score
    """
    

    month, day = check_point_date

    # Use ctx if provided, otherwise fallback to original behavior
    if ctx:
        current_year = ctx.current_year
        owner = ctx.owner
        repo_name = ctx.repo_name
        if ctx.api_creation_year==None:  # fallback solution
            creation_year=ctx.creation_year
        else:
            creation_year = ctx.api_creation_year
        graphql_url = "https://api.github.com/graphql"
        headers = ctx.graphql_headers
        print(f"Collecting issues for {owner}/{repo_name} using GraphQL aggregation (with context)...")
    else:
        current_year = datetime.now(timezone.utc).year
        owner, repo_name = _parse_repo_url(repo_url)
        print(f"Collecting issues for {owner}/{repo_name} using GraphQL aggregation...")

        graphql_url = "https://api.github.com/graphql"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        # Get repository creation year
        repo_query = """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) {
                createdAt
            }
        }
        """

        repo_response = requests.post(
            graphql_url,
            json={'query': repo_query, 'variables': {'owner': owner, 'repo': repo_name}},
            headers=headers,
            timeout=30
        )

        if repo_response.status_code != 200:
            print(f"Warning: Failed to get repo info")
            return {}

        repo_data = repo_response.json()
        if 'errors' in repo_data:
            print(f"GraphQL errors: {repo_data['errors']}")
            return {}

        created_at = repo_data['data']['repository']['createdAt']
        creation_year = datetime.fromisoformat(created_at.replace('Z', '+00:00')).year

    # Normalize year range
    if start_year is None:
        start_year = creation_year
    if end_year is None:
        end_year = current_year
    start_year = max(start_year, creation_year)
    end_year = min(end_year, current_year)

    repo = repo_name

    # Issue cutoff date: 2025-09-01
    ISSUE_CUTOFF_DATE = datetime(2025, month, day, tzinfo=timezone.utc)
    cutoff_date_str = ISSUE_CUTOFF_DATE.strftime('%Y-%m-%d')

    # GraphQL query for counting issues (created/open/closed in a period)
    count_query = """
    query ($qAll: String!, $qStillOpen: String!, $qClosedYear: String!) {
        all:       search(query: $qAll, type: ISSUE)       { issueCount }
        stillOpen: search(query: $qStillOpen, type: ISSUE) { issueCount }
        closedYear: search(query: $qClosedYear, type: ISSUE) { issueCount }
    }
    """

    # Query for fetching closed issues to calculate resolution time
    closed_issues_query = """
    query($owner:String!, $name:String!, $cursor:String, $since:DateTime!) {
        repository(owner:$owner, name:$name) {
            issues(first:100, after:$cursor, states:CLOSED, filterBy:{since:$since}) {
                nodes {
                    createdAt
                    closedAt
                    stateReason
                }
                pageInfo { hasNextPage endCursor }
            }
        }
    }
    """

    # Initialize result storage
    yearly_stats = defaultdict(lambda: {
        'total_issues': 0,
        'unresolved_issues': 0,
        'resolution_times': [],
        'closed_in_year': 0
    })

    # Process each year
    for year in range(start_year, end_year + 1):
        # Calculate date range for this year cycle
        # Year cycle: from checkpoint of (year-1) to checkpoint of year
        date_from = f"{year-1}-{month:02d}-{day:02d}"
        date_to = f"{year}-{month:02d}-{day:02d}"

        # Don't process beyond cutoff date
        if datetime.strptime(date_from, '%Y-%m-%d').replace(tzinfo=timezone.utc) > ISSUE_CUTOFF_DATE:
            continue

        # Adjust date_to if it exceeds cutoff
        if datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc) > ISSUE_CUTOFF_DATE:
            date_to = cutoff_date_str

        # print(f"  Processing year {year} ({date_from} to {date_to})...")

        # # 1. Count total issues created in this period
        # q_all = f"repo:{owner}/{repo} is:issue created:{date_from}..{date_to}"

        # # 2. Count issues that were still open at checkpoint (date_to)
        # # These are issues created before/at date_to AND still open at date_to
        # # We need to find issues that were either never closed, or closed after date_to
        # q_still_open = f"repo:{owner}/{repo} is:issue created:..{date_to} is:open"
        # q_closed_after = f"repo:{owner}/{repo} is:issue created:..{date_to} closed:>{date_to}"

        # Issues created in period; still open at checkpoint; closed in period (regardless of created date)
        q_created_period = f"repo:{owner}/{repo} is:issue created:{date_from}..{date_to}"
        q_created_period_still_open = f"repo:{owner}/{repo} is:issue state:open created:{date_from}..{date_to}"
        q_closed_in_period = f"repo:{owner}/{repo} is:issue closed:{date_from}..{date_to}"

        count_response = requests.post(
            graphql_url,
            json={'query': count_query, 'variables': {
                'qAll': q_created_period,
                'qStillOpen': q_created_period_still_open,
                'qClosedYear': q_closed_in_period
            }},
            headers=headers,
            timeout=30
        )

        if count_response.status_code != 200:
            print(f"    Warning: Failed to count issues for year {year}")
            continue

        count_data = count_response.json()
        if 'errors' in count_data:
            print(f"    GraphQL errors: {count_data['errors']}")
            continue

        total_issues = count_data['data']['all']['issueCount']           # opened this period
        still_open = count_data['data']['stillOpen']['issueCount']       # opened in period & still open
        closed_in_year = count_data['data']['closedYear']['issueCount']  # closed in period (any created date)

        # Now we need to add issues that were closed after checkpoint
        # Fetch all closed issues in this period to calculate both unresolved count and avg resolution time
        since_iso = date_from + "T00:00:00Z"
        cursor = None
        closed_in_period = []

        while True:
            variables = {"owner": owner, "name": repo, "cursor": cursor, "since": since_iso}
            response = requests.post(
                graphql_url,
                json={"query": closed_issues_query, "variables": variables},
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Warning: Failed to fetch closed issues")
                break

            data = response.json()
            if 'errors' in data:
                print(f"GraphQL errors: {data['errors']}")
                break

            issues_data = data['data']['repository']['issues']
            for node in issues_data['nodes']:
                created_at_str = node['createdAt']
                closed_at_str = node.get('closedAt')
                state_reason = node.get('stateReason', '')

                if not closed_at_str:
                    continue

                created_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                closed_date = datetime.fromisoformat(closed_at_str.replace('Z', '+00:00'))
                checkpoint_date = datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)

                # Only consider issues created in this period and before cutoff
                if date_from <= created_at_str[:10] <= date_to and created_date <= ISSUE_CUTOFF_DATE:
                    closed_in_period.append({
                        'created': created_date,
                        'closed': closed_date,
                        'state_reason': state_reason
                    })

            if not issues_data['pageInfo']['hasNextPage']:
                break
            cursor = issues_data['pageInfo']['endCursor']

        # Count issues closed after checkpoint (these are unresolved at checkpoint)
        closed_after_checkpoint = 0
        for issue in closed_in_period:
            checkpoint_date = datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            if issue['closed'] > checkpoint_date:
                closed_after_checkpoint += 1

        # Calculate unresolved issues at checkpoint
        unresolved_issues = still_open + closed_after_checkpoint

        # Calculate average resolution time (excluding NOT_PLANNED)# only for issues open in this period
        for issue in closed_in_period:
            if issue['state_reason'] != 'NOT_PLANNED':
                resolution_days = (issue['closed'] - issue['created']).total_seconds() / 3600 / 24
                yearly_stats[year]['resolution_times'].append(resolution_days)

        yearly_stats[year]['total_issues'] = total_issues
        yearly_stats[year]['unresolved_issues'] = unresolved_issues
        yearly_stats[year]['closed_in_year'] = closed_in_year

        # print(f"    Year {year}: Total={total_issues}, Unresolved={unresolved_issues}, Closed samples={len(closed_in_period)}")


    # Search for CVEs using nvd_search
    print(f"Searching for CVEs related to '{repo_name}'...")
    try:
        cve_results = nvd_search(repo_name, api_key=nvd_api_key, delay=6)
    except Exception as e:
        print(e)
        cve_results=[]

    # Process CVE results by year
    cve_yearly_stats = defaultdict(lambda: {
        'cve_count': 0,
        'cvss_scores': []
    })

    for cve in cve_results:
        published = cve.get('published', '')
        if published:
            cve_year = int(published.split('-')[0])
            cve_yearly_stats[cve_year]['cve_count'] += 1

            cvss = cve.get('cvss')
            if cvss and len(cvss) == 2:
                score = cvss[1]
                if score is not None:
                    cve_yearly_stats[cve_year]['cvss_scores'].append(score)

    # Build result dictionary
    result = {
        'issues_per_year': [],
        'unresolved_issues_per_year': [],
        'unresolved_issues_ratio_per_year': [],
        'avg_issue_resolution_days_per_year': [],
        'closed_issues_per_year': [],
        'cve_count_per_year': [],
        'avg_cvss_per_year': [],
        'years': []
    }

    # Calculate statistics for each year (output in reverse order)
    total_cves = 0
    total_cvss_scores = []

    for year in range(end_year, start_year - 1, -1):
        stats = yearly_stats.get(year, {
            'total_issues': 0,
            'unresolved_issues': 0,
            'resolution_times': []
        })

        cve_stats = cve_yearly_stats.get(year, {
            'cve_count': 0,
            'cvss_scores': []
        })

        # Issue statistics
        total_issues = stats.get('total_issues', 0)
        unresolved_issues = stats.get('unresolved_issues', 0)
        unresolved_ratio = unresolved_issues / total_issues if total_issues > 0 else 0

        resolution_times = stats.get('resolution_times', [])
        avg_resolution_days = statistics.mean(resolution_times) if resolution_times else 0
        closed_in_year = stats.get('closed_in_year', 0)

        # CVE statistics
        cve_count = cve_stats.get('cve_count', 0)
        cvss_scores = cve_stats.get('cvss_scores', [])
        avg_cvss = statistics.mean(cvss_scores) if cvss_scores else 0

        # Accumulate for historical averages
        total_cves += cve_count
        total_cvss_scores.extend(cvss_scores)

        # Append to result
        result['years'].append(year)
        result['issues_per_year'].append(total_issues)
        result['unresolved_issues_per_year'].append(unresolved_issues)
        result['unresolved_issues_ratio_per_year'].append(round(unresolved_ratio, 4))
        result['avg_issue_resolution_days_per_year'].append(round(avg_resolution_days, 2))
        result['closed_issues_per_year'].append(closed_in_year)
        result['cve_count_per_year'].append(cve_count)
        result['avg_cvss_per_year'].append(round(avg_cvss, 2))

    # Calculate historical averages
    total_years = end_year - start_year + 1
    result['historical_avg_cve_count'] = round(total_cves / total_years, 2) if total_years > 0 else 0
    result['historical_avg_cvss'] = round(statistics.mean(total_cvss_scores), 2) if total_cvss_scores else 0

    return result


def collect_issue_vulnerability_info_nocve(repo_url: str, check_point_date: Tuple[int, int], token: str,
                                           ctx: 'RepoContext' = None,
                                           start_year: int | None = None, end_year: int | None = None) -> Dict[str, Any]:
    """
    Collect issue information using GraphQL aggregation (without CVE data)

    Args:
        repo_url: GitHub repository URL
        check_point_date: Checkpoint date (month, day)
        token: GitHub Personal Access Token
        ctx: RepoContext object (optional, for optimization)

    Returns:
        Dictionary containing yearly statistics:
        - issues_per_year: Total issues submitted per year
        - unresolved_issues_per_year: Issues unresolved by checkpoint date
        - unresolved_issues_ratio_per_year: Ratio of unresolved issues
        - avg_issue_resolution_days_per_year: Average days to close issues
        - years: List of years
    """

    month, day = check_point_date

    # Use ctx if provided, otherwise fallback to original behavior
    if ctx:
        current_year = ctx.current_year
        owner = ctx.owner
        repo_name = ctx.repo_name
        if ctx.api_creation_year == None:  # degrade solution
            creation_year = ctx.creation_year
        else:
            creation_year = ctx.api_creation_year
        graphql_url = "https://api.github.com/graphql"
        headers = ctx.graphql_headers
        print(f"Collecting issues for {owner}/{repo_name} using GraphQL aggregation (with context, no CVE)...")
    else:
        current_year = datetime.now(timezone.utc).year
        owner, repo_name = _parse_repo_url(repo_url)
        print(f"Collecting issues for {owner}/{repo_name} using GraphQL aggregation (no CVE)...")

        graphql_url = "https://api.github.com/graphql"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        # Get repository creation year
        repo_query = """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) {
                createdAt
            }
        }
        """

        repo_response = requests.post(
            graphql_url,
            json={'query': repo_query, 'variables': {'owner': owner, 'repo': repo_name}},
            headers=headers,
            timeout=30
        )

        if repo_response.status_code != 200:
            print(f"Warning: Failed to get repo info")
            return {}

        repo_data = repo_response.json()
        if 'errors' in repo_data:
            print(f"GraphQL errors: {repo_data['errors']}")
            return {}

        created_at = repo_data['data']['repository']['createdAt']
        creation_year = datetime.fromisoformat(created_at.replace('Z', '+00:00')).year

    # Normalize year range
    if start_year is None:
        start_year = creation_year
    if end_year is None:
        end_year = current_year
    start_year = max(start_year, creation_year)
    end_year = min(end_year, current_year)

    repo = repo_name

    # Issue cutoff date: 2025-09-01
    ISSUE_CUTOFF_DATE = datetime(2025, month, day, tzinfo=timezone.utc)
    cutoff_date_str = ISSUE_CUTOFF_DATE.strftime('%Y-%m-%d')

    # GraphQL query for counting issues (created/open/closed in a period)
    count_query = """
    query ($qAll: String!, $qStillOpen: String!, $qClosedYear: String!) {
        all:       search(query: $qAll, type: ISSUE)       { issueCount }
        stillOpen: search(query: $qStillOpen, type: ISSUE) { issueCount }
        closedYear: search(query: $qClosedYear, type: ISSUE) { issueCount }
    }
    """

    # Query for fetching closed issues to calculate resolution time
    closed_issues_query = """
    query($owner:String!, $name:String!, $cursor:String, $since:DateTime!) {
        repository(owner:$owner, name:$name) {
            issues(first:100, after:$cursor, states:CLOSED, filterBy:{since:$since}) {
                nodes {
                    createdAt
                    closedAt
                    stateReason
                }
                pageInfo { hasNextPage endCursor }
            }
        }
    }
    """

    # Initialize result storage
    yearly_stats = defaultdict(lambda: {
        'total_issues': 0,
        'unresolved_issues': 0,
        'resolution_times': [],
        'closed_in_year': 0
    })

    # Process each year
    for year in range(start_year, end_year + 1):
        # Calculate date range for this year cycle
        # Year cycle: from checkpoint of (year-1) to checkpoint of year
        date_from = f"{year-1}-{month:02d}-{day:02d}"
        date_to = f"{year}-{month:02d}-{day:02d}"

        # Don't process beyond cutoff date
        if datetime.strptime(date_from, '%Y-%m-%d').replace(tzinfo=timezone.utc) > ISSUE_CUTOFF_DATE:
            continue

        # Adjust date_to if it exceeds cutoff
        if datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc) > ISSUE_CUTOFF_DATE:
            date_to = cutoff_date_str

        # Issues created in period; still open at checkpoint; closed in period (regardless of created date)
        q_created_period = f"repo:{owner}/{repo} is:issue created:{date_from}..{date_to}"
        q_created_period_still_open = f"repo:{owner}/{repo} is:issue state:open created:{date_from}..{date_to}"
        q_closed_in_period = f"repo:{owner}/{repo} is:issue closed:{date_from}..{date_to}"

        count_response = requests.post(
            graphql_url,
            json={'query': count_query, 'variables': {
                'qAll': q_created_period,
                'qStillOpen': q_created_period_still_open,
                'qClosedYear': q_closed_in_period
            }},
            headers=headers,
            timeout=30
        )

        if count_response.status_code != 200:
            print(f"    Warning: Failed to count issues for year {year}")
            continue

        count_data = count_response.json()
        if 'errors' in count_data:
            print(f"    GraphQL errors: {count_data['errors']}")
            continue

        total_issues = count_data['data']['all']['issueCount']           # opened this period
        still_open = count_data['data']['stillOpen']['issueCount']       # opened in period & still open
        closed_in_year = count_data['data']['closedYear']['issueCount']  # closed in period (any created date)

        # Now we need to add issues that were closed after checkpoint
        # Fetch all closed issues in this period to calculate both unresolved count and avg resolution time
        since_iso = date_from + "T00:00:00Z"
        cursor = None
        closed_in_period = []

        while True:
            variables = {"owner": owner, "name": repo, "cursor": cursor, "since": since_iso}
            response = requests.post(
                graphql_url,
                json={"query": closed_issues_query, "variables": variables},
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Warning: Failed to fetch closed issues")
                break

            data = response.json()
            if 'errors' in data:
                print(f"GraphQL errors: {data['errors']}")
                break

            issues_data = data['data']['repository']['issues']
            for node in issues_data['nodes']:
                created_at_str = node['createdAt']
                closed_at_str = node.get('closedAt')
                state_reason = node.get('stateReason', '')

                if not closed_at_str:
                    continue

                created_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                closed_date = datetime.fromisoformat(closed_at_str.replace('Z', '+00:00'))
                checkpoint_date = datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)

                # Only consider issues created in this period and before cutoff
                if date_from <= created_at_str[:10] <= date_to and created_date <= ISSUE_CUTOFF_DATE:
                    closed_in_period.append({
                        'created': created_date,
                        'closed': closed_date,
                        'state_reason': state_reason
                    })

            if not issues_data['pageInfo']['hasNextPage']:
                break
            cursor = issues_data['pageInfo']['endCursor']

        # Count issues closed after checkpoint (these are unresolved at checkpoint)
        closed_after_checkpoint = 0
        for issue in closed_in_period:
            checkpoint_date = datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            if issue['closed'] > checkpoint_date:
                closed_after_checkpoint += 1

        # Calculate unresolved issues at checkpoint
        unresolved_issues = still_open + closed_after_checkpoint

        # Calculate average resolution time (excluding NOT_PLANNED) only for issues open in this period
        for issue in closed_in_period:
            if issue['state_reason'] != 'NOT_PLANNED':
                resolution_days = (issue['closed'] - issue['created']).total_seconds() / 3600 / 24
                yearly_stats[year]['resolution_times'].append(resolution_days)

        yearly_stats[year]['total_issues'] = total_issues
        yearly_stats[year]['unresolved_issues'] = unresolved_issues
        yearly_stats[year]['closed_in_year'] = closed_in_year

    # Build result dictionary (without CVE data)
    result = {
        'issues_per_year': [],
        'unresolved_issues_per_year': [],
        'unresolved_issues_ratio_per_year': [],
        'avg_issue_resolution_days_per_year': [],
        'closed_issues_per_year': [],
        'years': []
    }

    # Calculate statistics for each year (output in reverse order)
    for year in range(end_year, start_year - 1, -1):
        stats = yearly_stats.get(year, {
            'total_issues': 0,
            'unresolved_issues': 0,
            'resolution_times': []
        })

        # Issue statistics
        total_issues = stats.get('total_issues', 0)#TODO: solve the potential error
        unresolved_issues = stats.get('unresolved_issues', 0)
        unresolved_ratio = unresolved_issues / total_issues if total_issues > 0 else 0

        resolution_times = stats.get('resolution_times', [])
        avg_resolution_days = statistics.mean(resolution_times) if resolution_times else 0
        closed_in_year = stats.get('closed_in_year', 0)

        # Append to result
        result['years'].append(year)
        result['issues_per_year'].append(total_issues)
        result['unresolved_issues_per_year'].append(unresolved_issues)
        result['unresolved_issues_ratio_per_year'].append(round(unresolved_ratio, 4))
        result['avg_issue_resolution_days_per_year'].append(round(avg_resolution_days, 2))
        result['closed_issues_per_year'].append(closed_in_year)

    return result


def get_high_star_repos_count(min_stars: int = 1000, github_token: str = None) -> int:
    """
    Query the number of public repositories with stars greater than the specified threshold using GraphQL

    Args:
        min_stars: Minimum star count threshold (default: 1000)
        github_token: GitHub Personal Access Token for API requests

    Returns:
        int: Total count of repositories matching the criteria
    """
    graphql_url = "https://api.github.com/graphql"

    # Prepare headers
    headers = {
        'Content-Type': 'application/json'
    }
    if github_token:
        headers['Authorization'] = f'Bearer {github_token}'

    # GraphQL query to get repository count
    query = """
    query($searchQuery: String!) {
        search(type: REPOSITORY, query: $searchQuery, first: 1) {
            repositoryCount
        }
    }
    """

    # Build search query: public repositories, stars > min_stars, not forks
    search_query = f"is:public stars:>{min_stars} fork:false"

    try:
        response = requests.post(
            graphql_url,
            json={
                'query': query,
                'variables': {'searchQuery': search_query}
            },
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if 'errors' not in data:
                repo_count = data['data']['search']['repositoryCount']
                print(f"Found {repo_count} public repositories with >{min_stars} stars (excluding forks)")
                return repo_count
            else:
                print(f"GraphQL errors: {data['errors']}")
                return 0
        else:
            print(f"HTTP error: {response.status_code}")
            return 0

    except Exception as e:
        print(f"Error querying repository count: {e}")
        return 0


if __name__ == "__main__":
    # Example usage of get_high_star_repos_count
    # Replace with your actual GitHub token
    GITHUB_TOKEN = PAT_TOKEN

    # Query repositories with >1000 stars
    count_1000 = get_high_star_repos_count(min_stars=1000, github_token=GITHUB_TOKEN)
    print(f"Repositories with >1000 stars: {count_1000}")

    # Query repositories with >5000 stars
    count_5000 = get_high_star_repos_count(min_stars=5000, github_token=GITHUB_TOKEN)
    print(f"Repositories with >5000 stars: {count_5000}")

    # Query repositories with >10000 stars
    count_10000 = get_high_star_repos_count(min_stars=10000, github_token=GITHUB_TOKEN)
    print(f"Repositories with >10000 stars: {count_10000}")
    print(1)
