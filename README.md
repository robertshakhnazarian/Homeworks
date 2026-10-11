# Bass Model Homework: Diffusion of Apple AirPods Pro 3

Marketing Analytics, Robert Shakhnazarian. Look-alike innovation: Google Translate. Scope: global.

## Structure
```
bass_airpods_pro3.ipynb   main notebook (report + code)
helper_functions.py       Google Trends loader, Bass model, NLS/OLS fitting, bootstrap, adopters tables
requirements.txt
data/                     input CSV (you add) and CSVs written by the notebook, see data/README.md
img/                      figures written by the notebook
report/                   report.pdf (you generate, see below) and references.md
```

## Run
1. `pip install -r requirements.txt`
2. Download the Google Trends CSV (link in `data/README.md`) and save it as `data/google_translate_trends.csv`.
3. `jupyter nbconvert --to notebook --execute --inplace bass_airpods_pro3.ipynb`

## Build the PDF
Pick one:
- `jupyter nbconvert --to pdf bass_airpods_pro3.ipynb --output-dir report --output report` (needs a LaTeX install)
- Open the executed notebook in Jupyter and use *File > Save and Export Notebook As > PDF*
- `jupyter nbconvert --to html bass_airpods_pro3.ipynb`, open the HTML in a browser, and print to PDF (keeps the equations)

Save the result as `report/report.pdf`. The notebook itself is the report source (instead of a separate `report_source.md`).
