# Bass Model Homework: Diffusion of Apple AirPods Pro 3

Marketing Analytics, Robert Shakhnazarian. Look-alike innovation: Google Translate. Scope: global.

The analysis estimates Bass diffusion parameters (p, q) from Google Trends data for Google Translate, sizes the market potential (M) for AirPods Pro 3 with a Fermi estimate, and forecasts adopters per period from the launch date (19 Sep 2025).

## Structure
```
bass_airpods_pro3.ipynb   main notebook (report text, code and outputs)
helper_functions.py       Google Trends loader, Bass model, NLS/OLS fitting, bootstrap, adopters tables
requirements.txt          Python dependencies
data/                     input data and tables written by the notebook (see data/README.md)
img/                      figures produced by the notebook
report/                   final report (report.pdf, report.html) and references.md
```

## How to reproduce
1. `pip install -r requirements.txt`
2. Open `bass_airpods_pro3.ipynb` in Jupyter from the repository folder and run all cells. The input file `data/google_translate_trends.csv` is included.

To rebuild the HTML report: `jupyter nbconvert --to html bass_airpods_pro3.ipynb --output-dir report --output report`. The PDF was produced by printing that HTML to PDF.

## Data
Google Trends, worldwide interest in "Google Translate", 2004 to present (monthly): https://trends.google.com/trends/explore?date=all&q=Google%20Translate. Statista only provided cross-sectional data for this product, so Google Trends is used as a proxy for adoption activity. All sources are listed in `report/references.md`.
