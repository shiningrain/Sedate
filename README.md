# SEDATE Framework

## TL;DR

We propose **SEDATE** , a multi-dimensional scoring framework that supplements the mainstream OpenSSF Scorecard.
SEDATE introduces 12 new metrics (e.g., core contributor retention rate, backlog management index) that quantify OSS security from the perspectives of developer maintenance status and project dynamic evolution.

We conduct a large-scale evaluation on 9,114 high-impact OSS projects, including 8,492 widespread OSS projects and 622 built-in libraries from 10 popular Unix-Like OS distributions.
Guided by SEDATE, we responsibly discover and report 47 new vulnerabilities in three flagged projects, resulting in 32 CVEs. We also interview 14 experts from both academia and industry, whose feedback corroborates our design.

This repository contains the implementation of SEDATE, experimental results (including a CVE list), and interview materials.

## Repo Structure

```
├── Code/
│   ├── demo.py                 # Main entry script for computing SEDATE metrics
│   ├── demo.csv                # Example input repository list
│   ├── requirement.txt         # Python dependencies
│   └── utils/
│       ├── score_utils.py      # Utility functions for data collection
│       ├── metric_utils.py     # Metric calculation functions
│       ├── alarm_config.json   # Threshold configuration for risk alerts
│       └── token_config.cfg    # API token configuration
├── Motivation/
│   ├── 1_raw_annotation.xlsx   # Raw expert annotations (sheets: raw-vulnerable, raw-control)
│   └── 1_all_annotation.csv    # Final labels of all 176 projects, aligned with the four findings
├── Experiment/
│   ├── rq1/                    # RQ1: Dataset, scripts, and visualization results
│   ├── rq2/                    # RQ2: Analysis scripts and heatmap visualizations
│   ├── rq3&rq4/                # RQ3/RQ4: Prediction, ablation, and sensitivity results
│   └── rq5/                    # RQ5: Vulnerability and CVE list
├── Interview/
│   ├── Interview Questions.pdf # Semi-structured interview questions
│   ├── Invitation Letters.pdf  # Participant invitation letters
│   └── Consent Form.pdf        # IRB consent form
└── README.md
```

## Setup

Our code is implemented on Python 3.9+. To install all dependencies, run the following command:

```bash
pip install -r Code/requirement.txt
```

Before running the tool, configure your API tokens in `Code/utils/token_config.cfg`:
- `PAT_TOKENS`: GitHub Personal Access Token for accessing repository metadata
- `NVD_TOKEN`: NVD API token for querying vulnerability data

### Usage

The `demo.py` script computes SEDATE metrics for GitHub repositories and checks for risk alerts.

**Single repository:**
```bash
python Code/demo.py --repo github.com/owner/repo --checkpoint-month 9 --checkpoint-day 2
```

**Multiple repositories from CSV:**
```bash
python Code/demo.py --csv demo.csv --checkpoint-month 9 --checkpoint-day 2 --start-year 2023 --end-year 2025
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--repo` | Single repository URL (e.g., `github.com/owner/repo`) | - |
| `--csv` | CSV file with repository list (requires `repo` column) | `demo.csv` |
| `--checkpoint-month` | Checkpoint month for evaluation | 9 |
| `--checkpoint-day` | Checkpoint day for evaluation | 2 |
| `--start-year` | Start year for time-series analysis | 2023 |
| `--end-year` | End year for time-series analysis | 2025 |
| `--output` | Output pickle file path | `demo_result.pkl` |
| `--config` | Custom alarm threshold configuration | `utils/alarm_config.json` |

The script outputs a pickle file containing risk alerts for each repository across the specified year range.

## Motivation

To understand the limitations of current scorecard metrics, we conduct a preliminary empirical study on 176 OSS projects.
The *vulnerable group* consists of 88 projects that meet all three criteria:
- **High Perceived Security**: at the checkpoint time t0 (09-02-2024), the project's latest release achieves an OpenSSF Scorecard (v5.3.0) rating no lower than 7.
- **High Exposed Vulnerability**: at least one high-severity CVE (CVSS score ≥ 7) is disclosed between t0 and t1 (09-02-2025).
- **Critical Supply Chain Impact**: the project is either pre-installed in Ubuntu 24.04 or has at least 100 GitHub stars with verified downstream adoption.

As a comparative baseline, the *control group* consists of 88 projects randomly selected from the same source that satisfy the first and third criteria but disclose no high-severity CVE between t0 and t1.

Three experts with at least five years of experience in software engineering and security analyze the source code, tests, GitHub metadata, and Scorecard details of all 176 projects during the observation period (09-02-2023 to 09-02-2024), following open coding and open card sorting.
Two experts independently annotate every project while blinded to its group assignment, and the third expert mediates the remaining discrepancies.
The inter-rater agreement over all 176 projects reaches a Cohen's kappa of 0.82, and the manual analysis takes over 240 person-hours.
The analysis identifies four key findings revealing gaps between scorecard ratings and real security posture:

1. **Finding 1**: 25/88 projects in the vulnerable group exhibit inactive developer engagement and community participation.
2. **Finding 2**: 42/88 projects in the vulnerable group expose insufficient testing capabilities.
3. **Finding 3**: 77/88 projects in the vulnerable group face critical developer issues (e.g., knowledge concentration and maintainer attrition).
4. **Finding 4**: 71/88 projects in the vulnerable group exhibit deficiencies in issue and vulnerability management.

The raw annotations of both groups are available in [Motivation/1_raw_annotation.xlsx](Motivation/1_raw_annotation.xlsx) (sheets `raw-vulnerable` and `raw-control`).
The final labels aligned with the four findings are available in [Motivation/1_all_annotation.csv](Motivation/1_all_annotation.csv), where the first 88 rows are the vulnerable group and the last 88 rows are the control group.

## Experimental Results

### RQ1

This experiment evaluates SEDATE on 9,114 high-impact OSS projects to analyze the alarm distribution across different scores.

**Available resources:**
- [Experiment/rq1/dataset.csv](Experiment/rq1/dataset.csv) - Complete experimental dataset
- [Experiment/rq1/rq1.py](Experiment/rq1/rq1.py) - Analysis script
- [Experiment/rq1/1_rq1.pkl](Experiment/rq1/1_rq1.pkl) - Processed analysis results
- PDF visualizations: bucket distribution charts and alarm rate trends

### RQ2

This experiment studies the root causes distribution of high-risk projects flagged by SEDATE.

**Available resources:**
- [Experiment/rq2/rq2.py](Experiment/rq2/rq2.py) - Analysis script
- [Experiment/rq2/2_rq2.pkl](Experiment/rq2/2_rq2.pkl) - Processed analysis results
- PDF visualizations: heatmaps showing metric occurrence and share distributions

### RQ3 and RQ4

These experiments evaluate prediction and ablation using the reported alerts in RQ1, and examine sensitivity to the 12 metric thresholds.

**Available results:**
- [Experiment/rq3&rq4/1_rq3.pkl](Experiment/rq3%26rq4/1_rq3.pkl) - Table IV results, including each method's detected repositories by year and its evaluation metrics.
- [Experiment/rq3&rq4/1_rq4.pkl](Experiment/rq3%26rq4/1_rq4.pkl) - Two dictionaries, `ablation` and `sensitivity`. Ablation includes the annual repository sets for each Ablated/Alone setting and Table V statistics; sensitivity includes Rec., Add., Recall, and Precision at each tested threshold.
- [Experiment/rq3&rq4/4_rq4_sensitivity.pdf](Experiment/rq3%26rq4/4_rq4_sensitivity.pdf) - Sensitivity plots of Add. and Rec. under different thresholds of the 12 metrics.


### RQ5

To validate the practical value of SEDATE, this RQ randomly selects five projects with high OpenSSF Scorecard ratings (over 5, some exceeding 7) but triggering SEDATE alerts and conducts a vulnerability discovery experiment.

**Results:**
- Discovered and reported **47 previously unknown vulnerabilities**
- Obtained **32 CVEs** (and still increasing)

The list of vulnerabilities can be found in [Experiment/rq5/Vulnerability+CVE list.csv](Experiment/rq5/Vulnerability+CVE%20list.csv) (We have processed the content to avoid potential security risks).


## Interview Study

To understand real-world needs for OSS assessment, we conduct semi-structured interviews with 14 experts from both academia and industry, including:
- Leaders and practitioners within OSS organizations
- Security experts from large technology companies
- Faculty members and graduate students from universities

All participants have a minimum of 3 years of experience in OSS contribution or maintenance and prior familiarity with OpenSSF Scorecard.


**Interview materials:**
- [Interview/Interview Questions.pdf](Interview/Interview%20Questions.pdf) - Semi-structured interview protocol
- [Interview/Invitation Letters.pdf](Interview/Invitation%20Letters.pdf) - Participant recruitment materials
- [Interview/Consent Form.pdf](Interview/Consent%20Form.pdf) - IRB consent form
