# HiveSpace Reservation No-Show Dataset

## Project Overview

This project generates a synthetic dataset of 1,000 coworking space reservations for HiveSpace. The dataset is designed to support future analysis and machine learning experiments for predicting reservation no-shows.

## Dataset Information

The generated dataset contains 1,000 records with the following columns:

| Column | Description |
|---|---|
| `member_id` | Unique-style identifier assigned to each member |
| `reservation_date` | Date of the reservation |
| `time_slot` | Reservation time category (Morning, Afternoon, Evening) |
| `member_tenure_days` | Number of days the member has been associated with HiveSpace |
| `no_show` | Reservation outcome: 0 = attended, 1 = no-show |

## Data Generation Logic

- Generates 1,000 synthetic reservations.
- Assigns member IDs and reservation dates.
- Uses three reservation time slots.
- Generates member tenure between 1 and 1,200 days.
- Assigns a higher base no-show probability to newer members.
- Base no-show probability decreases gradually from approximately 35% for brand-new members to 25% after 365 days (one year).
- **For members with more than 365 days of tenure, the base no-show probability remains at 25% rather than decreasing further.**
- Includes time-slot adjustments and random variation to avoid overly strong correlations.
- Generates binary no-show outcomes using probability-based random sampling.

## No-Show Distribution

The generated dataset contains:

- Attended reservations: 728 (72.8%)
- No-show reservations: 272 (27.2%)

The distribution is not balanced artificially. No-show outcomes are generated through probability-based random sampling.

## Requirements

- Python 3.11+
- Pandas
- NumPy

## Installation

Clone the repository:

```bash
git clone https://github.com/Muhammad-Hayad/hivespace-reservation-no-show-data.git
cd hivespace-reservation-no-show-data
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install pandas numpy
```

## Generate the Dataset

Run:

```bash
python generate_data.py
```

The script generates `reservations.csv` in the project directory and displays a dataset preview, no-show distribution, and no-show rates by tenure group.

## Project Structure

```text
hivespace-reservation-no-show-data/
│
├── generate_data.py
├── reservations.csv
├── README.md
└── .gitignore
```

## Purpose

This synthetic dataset provides a starting point for exploring reservation no-show patterns and developing predictive models for coworking space operations.