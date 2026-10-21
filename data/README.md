# Data Source

RCRA (Resource Conservation and Recovery Act) data for this prediction model comes from the EPA ECHO (Enforcement and Compliance History Online) data downloads: https://echo.epa.gov/tools/data-downloads

## File
- **Filename:** `PIPELINE_RCRA_00_COMPLETE.csv` (about 145 MB)
- **Location:** place it in this `data/` folder. The file is not included in the repo because it exceeds GitHub's 100 MB file limit.
- **Downloaded:** October 10, 2026
- **Where to find it on ECHO:** [(https://echo.epa.gov/tools/data-downloads#hazardouswaste) / RCRA Pipline Dataset (ZIP)]

## Reproducibility note
Results depend on this snapshot. ECHO updates its files, so a newer download may give different numbers. If you use a newer download, update `LABEL_LAG_CUTOFF` in the notebook's settings cell. I used 2026-09-01 for an October 10 download, because evaluations close to the download date often do not have violations determined yet.

## Disclaimer
This is public EPA data. This project is not affiliated with or endorsed by the EPA.