# Clinical Characteristics and Treatment Outcomes of Primary Cardiac Sarcomas: A Retrospective Analysis

## Repository Structure

```
CCTO_PCS/
├── main.py                  # Main entry point (simplified)
├── FiguresOuter.py          # Core figure generation (legacy)
├── src/                     # Modular code
│   ├── config.py            # Style settings
│   ├── data_loader.py       # Data loading
│   ├── utils.py             # Utilities
│   └── __init__.py
├── requirements.txt         # Dependencies
├── Data/                    # Data files (not tracked)
├── Figures/                 # Output figures (not tracked)
└── Tables/                  # Output statistics (not tracked)
```

## Figures

**Figure 1**: Prognostic factors and patient characteristics
- A: Cox regression forest plot (univariate/multivariate)
- B: Distant metastases vs OS
- C: Baseline LDH vs OS

**Figure 2**: Treatment outcomes (All, Surgery, Adjuvant, Chemotherapy, Targeted, Immunotherapy)

**Figure 3**: Clinical characteristics
- A: Age vs OS (scatter)
- B: Age groups vs OS (KM curves)
- C: Pathological subtype vs OS (KM, Angiosarcoma vs Others)
- D: Pathological subtype boxplot with pairwise comparisons

## Installation

```bash
git clone https://github.com/EnyuY/CCTO_PCS.git
cd CCTO_PCS
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Outputs: `Figures/*.pdf`, `Figures/*.png`, `Tables/*.csv`, `Tables/supplemental_table_1.xlsx`

## Statistical Methods

- Kaplan-Meier survival curves with log-rank test
- Cox proportional hazards regression (univariate/multivariate, p<0.10 threshold)
- Mann-Whitney U / Kruskal-Wallis tests
- Fixed random seed (42) for reproducibility

## Dependencies

pandas (>=1.5.0), numpy (>=1.23.0), matplotlib (>=3.5.0), scipy (>=1.9.0), lifelines (>=0.27.0), openpyxl (>=3.0.0)

See `requirements.txt` for complete list.


