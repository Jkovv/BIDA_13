import duckdb
import os
import unicodedata
import ftfy

def clean_txt(text):
    if text is None: return None
    # fix encoding corruption first, then strip diacritics
    fixed = ftfy.fix_text(str(text))
    nfkd = unicodedata.normalize('NFKD', fixed.lower().strip())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def run():
    con = duckdb.connect()
    con.create_function("clean_txt", clean_txt, ['VARCHAR'], 'VARCHAR')

    # Load all training CSVs
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv')")

    # Winsorize runtime and cap votes to handle outliers
    con.execute("""
        CREATE TABLE clean_base AS SELECT DISTINCT
            tconst, clean_txt(primaryTitle) as title_clean,
            TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
            LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
            LEAST(numVotes, 400000) as votes,
            label
        FROM raw
    """)

    # Impute missing runtime with decade median, compute log_votes + new features
    con.execute("""
        CREATE TABLE final_export AS SELECT
            tconst, title_clean, year, votes, label,
            COALESCE(runtime, MEDIAN(runtime) OVER (PARTITION BY (year/10*10))) as runtime,
            LOG(votes + 1) as log_votes,
            -- vote density: engagement relative to film age (cult classic detector)
            CASE WHEN year IS NOT NULL AND year < 2026
                THEN votes * 1.0 / (2026 - year)
                ELSE NULL
            END as vote_density,
            -- cinematic era buckets (survivorship bias is real for older films)
            CASE
                WHEN year < 1930 THEN 'silent'
                WHEN year < 1950 THEN 'golden_age'
                WHEN year < 1970 THEN 'classic'
                WHEN year < 1990 THEN 'new_hollywood'
                WHEN year < 2005 THEN 'modern'
                ELSE 'contemporary'
            END as era
        FROM clean_base
    """)

    os.makedirs('/app/processed', exist_ok=True)
    con.execute("COPY final_export TO '/app/processed/train.parquet' (FORMAT PARQUET)")

    # Process hidden sets
    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}')")
            con.execute(f"""
                COPY (SELECT
                    tconst,
                    clean_txt(primaryTitle) as title_clean,
                    TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
                    LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
                    LEAST(numVotes, 400000) as votes,
                    LOG(LEAST(numVotes, 400000) + 1) as log_votes,
                    CASE WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) IS NOT NULL
                        THEN LEAST(numVotes, 400000) * 1.0 / (2026 - TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER))
                        ELSE NULL
                    END as vote_density,
                    CASE
                        WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) < 1930 THEN 'silent'
                        WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) < 1950 THEN 'golden_age'
                        WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) < 1970 THEN 'classic'
                        WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) < 1990 THEN 'new_hollywood'
                        WHEN TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) < 2005 THEN 'modern'
                        ELSE 'contemporary'
                    END as era
                FROM {split}_raw) TO '/app/processed/{split}.parquet' (FORMAT PARQUET)
            """)
    con.close()

if __name__ == "__main__":
    run()
