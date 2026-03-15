## IMDB Movie Rating Classifier - Group 13

Binary classification pipeline to predict whether a movie is highly rated on IMDB.

**Validation accuracy: 79.16%** | [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

### Pipeline Overview

```
cleaning.py     DuckDB   Raw CSVs → cleaned parquet (unicode fix, winsorization, log_votes, era, vote_density)
prestige.py     Pandas   director/writer Bayesian prestige scores joined to parquet
enrich.py       Pandas   IMDB title.basics genres joined via tconst (downloaded at runtime)
features.py     PySpark  TF-IDF on cleaned titles → feature parquet
run.py          sklearn  Model competition (XGB/LGBM/CatBoost/RF/ADA) → predictions
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

`title.basics.tsv.gz` is downloaded automatically from IMDB on first run (~200MB, cached after).

### Features Used

| Feature | Source | Description |
|---|---|---|
| `log_votes` | train CSV | log1p(numVotes) - strongest signal |
| `vote_density` | train CSV | votes / film age - cult classic detector |
| `year` / `era_ord` | train CSV | release year + cinematic era bucket |
| `runtime` | train CSV | winsorized to [1, 210] min |
| `director_prestige` | directing.json | Bayesian-smoothed director success rate |
| `writer_prestige` | writing.json | Bayesian-smoothed writer success rate |
| `genre_*` | IMDB title.basics | one-hot encoded top 10 genres |

### The Projects

* [IMDB Project](imdb/) - learn to identify highly rated movies
* [Reviews Project](reviews/) - learn to identify helpful product reviews
* [DBLP Project](dblp/) - learn to identify duplicate entries in a bibliography

Consult the [project page on Canvas](https://canvas.uva.nl/courses/56576/pages/projects) for detailed instructions on the scope and grading of the projects.

### Submitting predictions

Each project contains two files `validation_hidden.csv` and `test_hidden.csv`, with the data for which your ML pipeline has to create predictions. In order to submit your predictions, you need to create two text files (one for the validation set and one for the test set). Each line in these files must consist of either the string `True` or the string `False`, which denote the predicted class for the corresponding data item in the validation or test files. 

In order to submit predictions for your team, you have to use our [submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/). The access credentials for the submission server will be given out by the TAs in next week's lab. For each submission, the submission server will compute the accuracy on the validation set and the test set. However, only the accuracy on the validation set will be shown (and used to generate the leaderboard). For each project, there is a _random-baseline_ submission, which shows the accuracy achieved by random guessing, and a _ta-baseline_ submission, which shows the accuracy of a minimal submission created by one of the TAs. Each team can submit only five times per day.
