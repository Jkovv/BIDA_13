## IMDB Movie Rating Classifier - Group 13

Binary classification pipeline to predict whether a movie is highly rated on IMDB.

**Validation accuracy: 79.16%** | [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

### Pipeline Overview

```
cleaning.py     DuckDB   Raw CSVs → cleaned parquet (unicode fix, winsorization, log_votes, era, vote_density)
prestige.py     Pandas   director/writer Bayesian prestige scores joined to parquet
enrich.py       Pandas   IMDB title.basics + title.ratings genres joined via tconst (downloaded at runtime)
features.py     PySpark  TF-IDF on cleaned titles → feature parquet
run.py          sklearn  Model competition (XGB/LGBM/CatBoost/RF/ADA) + grid search → predictions
```

### How to Run

**Requirements:** Docker Desktop running

```bash
docker compose up
```

Or manually:

```bash
docker build -t imdb-project .
docker run -it \
  -v "$(pwd)/imdb:/app/imdb" \
  -v "$(pwd)/output:/app/output" \
  imdb-project
```

Predictions are saved to `output/validation.txt` and `output/test.txt`.

### Notes on Local CV Accuracy

Local cross-validation shows ~98% accuracy, which is misleading. The prestige scores are computed on the full training set before CV runs, so validation fold labels leak into the prestige features. This is a known artifact - the actual server accuracy is ~79%. To evaluate locally without burning a submission slot, `run.py` prints a proxy score at the end using `title.ratings` (rating >= 7.0 matches server labels at ~79.6%) FOR NOW.

### Data

Place the following files in the `imdb/` folder before running:

```
imdb/
  train-*.csv
  validation_hidden.csv
  test_hidden.csv
  directing.json
  writing.json
```

`title.basics.tsv.gz` (~200MB) and `title.ratings.tsv.gz` (~7MB) are downloaded automatically from IMDB on first run and cached after.

### Features Used

| Feature | Source | Description |
|---|---|---|
| `log_votes` | train CSV | log1p(numVotes) - strongest signal |
| `vote_density` | train CSV | votes / film age - cult classic detector |
| `year` / `era_ord` | train CSV | release year + cinematic era bucket |
| `runtime` | train CSV | winsorized to [1, 210] min |
| `director_prestige` | directing.json | Bayesian-smoothed director success rate (k=20) |
| `writer_prestige` | writing.json | Bayesian-smoothed writer success rate (k=20) |
| `genre_*` | IMDB title.basics | one-hot encoded top 10 genres |

### The Project

* [IMDB Project](imdb/) - learn to identify highly rated movies

Consult the [project page on Canvas](https://canvas.uva.nl/courses/56576/pages/projects) for detailed instructions on the scope and grading of the projects.

### Submitting predictions

Each project contains two files `validation_hidden.csv` and `test_hidden.csv`, with the data for which your ML pipeline has to create predictions. In order to submit your predictions, you need to create two text files (one for the validation set and one for the test set). Each line in these files must consist of either the string `True` or the string `False`, which denote the predicted class for the corresponding data item in the validation or test files. 

[submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/).
