# Clinical Characteristics and Treatment Outcomes of Primary Cardiac Sarcomas: A Retrospective Analysis

## Repository Structure

```
CCTO_PCS/
├── FiguresOuter.py           # Main script (Figure 1, 2, 3)
├── Figure1C_COX.py           # Cox regression (Figure 1C)
├── requirements.txt          # Dependencies
├── Data/                     # Data files (not tracked)
├── Figures/                  # Output figures (not tracked)
└── Tables/                   # Output statistics (not tracked)
```

## Figures

**Figure 1**: Patient characteristics and prognostic factors (Pathological subtype, LDH, Cox forest plot)

**Figure 2**: Treatment outcomes (All patients, Surgery, Adjuvant, Chemotherapy, Targeted, Immunotherapy)

**Figure 3**: Clinical characteristics (Age, Metastases boxplots/KM curves, Pathological subtypes)

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
python FiguresOuter.py
```

Outputs: `Figures/*.pdf`, `Figures/*.png`, `Tables/*.csv`

## Statistical Methods

- Kaplan-Meier survival curves with log-rank test
- Cox proportional hazards regression (univariate/multivariate, p<0.10 threshold)
- Mann-Whitney U / Kruskal-Wallis tests
- Fixed random seed (42) for reproducibility

## Dependencies

pandas (>=1.5.0), numpy (>=1.23.0), matplotlib (>=3.5.0), scipy (>=1.9.0), lifelines (>=0.27.0), openpyxl (>=3.0.0)

See `requirements.txt` for complete list.


