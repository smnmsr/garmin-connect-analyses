# Garmin Connect Analyses

Analyse your Garmin Connect activities with Python.

## Setup

1. Create a Personal API key in [Garmin Connect](https://connect.garmin.com/app/settings/personalApiToken).
2. Add it to `.env`:

   ```
   GARMIN_CONNECT_API_KEY="GCDC_..."
   ```

3. `uv sync`

## Run

```
uv run scripts/power_vs_hr.py
```

Output goes to `output/`.

## Structure

```
src/garmin_connect_analyses/   shared code: Garmin API, FIT decoding
scripts/                       one analysis per file
output/                        generated files (git-ignored)
data/                          downloaded Garmin files (git-ignored)
docs/                          definitions and units of analysis features
```
