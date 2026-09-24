# Module 1 — Data Pipeline (`/data_pipeline`)

Scrape → clean → convert → store → query against
[books.toscrape.com](https://books.toscrape.com/) (a public scraping-practice
site: no login, no API key, no paid tier).

## Install / Run

```bash
pip install requests beautifulsoup4 pandas
python data_pipeline/pipeline.py
```

One command does everything: scrape → clean → convert → load SQLite → run the
SQL queries → prove pandas equivalence. Artifacts written next to the script:

| File | What it is |
|---|---|
| `books.db` | SQLite database (schema below), rebuilt from scratch each run |
| `books.csv` | Cleaned dataset snapshot (159 rows) |
| `sql_queries_and_output.txt` | All 6 SQL queries + their printed output |
| `pandas_equivalence.txt` | `read_sql` vs `pd.merge` side-by-side proof |

## What the script does

1. **Scrape** — `requests` + `BeautifulSoup` over **6 categories** (Travel,
   Mystery, Science, Fiction, Classics, History), following each category's
   pagination. Captures `title`, `price` (GBP as listed), `star_rating` (text,
   e.g. `"Three"`), `availability` (text), `category`.
   HTTP status codes are checked explicitly (non-200 → logged, page skipped);
   request exceptions are caught so one bad page never crashes the run.
2. **Clean** — see decisions below.
3. **Convert** — `price_inr = price_gbp × 105.50`.
4. **Load** — normalized 2-table schema via `sqlite3`.
5. **Query** — 6 SQL queries, output saved to
   `sql_queries_and_output.txt`.
6. **Prove** — 2 query results read back with `pd.read_sql_query`; the JOIN
   result reproduced independently with `pd.merge` and asserted equal.

## Result summary

- **159 books** scraped (≥ 60 required) across **6 categories** (≥ 3 required).
- Zero rows dropped, zero values median-imputed on this run (site parsed
  cleanly); the imputation/drop logic still runs so messy rows can't crash it.

## Cleaning / parsing decisions

| Field | Rule |
|---|---|
| `price_gbp` | Regex the first number out of the `£xx.xx` text → `float`. |
| `rating` | `{"One":1 … "Five":5}` map on the star-rating class → `int` 1–5. |
| `in_stock` | `"in stock"` (case-insensitive) in availability → `1`, else `0` (SQLite has no native bool; `1 = True`). |
| Unparseable **numeric** (price/rating) | **Median imputation** — a single messy cell shouldn't delete an otherwise valid catalog row. Median computed on the non-null values of that column. |
| Unparseable **identity** (title/category/availability) | **Drop the row** — without these the record can't be keyed or attributed to a category, so there is nothing worth imputing. |
| `price_inr` | `round(price_gbp × 105.50, 2)`. |

## Currency conversion — required fixed baseline

**1 GBP = 105.50 INR** — an artificial, project-defined constant for this
assignment. It is **not** a live or historical market rate, needs **no
lookup, no API, no date reference, and no network access**. This fixed-rate
baseline is the required graded path for `price_inr`.
(The optional keyless currency-API stretch is not used and does not affect
this submission.)

## Schema (two tables, PK/FK)

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

`books.category_id` → `categories.category_id` is the required PK/FK
relationship. Category names are stored once; books reference them by id.

## SQL queries (6 total, all clauses + JOIN covered)

Full text + output: `sql_queries_and_output.txt`.

| # | Covers | Query |
|---|---|---|
| Q1 | `SELECT` / `WHERE` | In-stock books rated ≥ 4, top 10 |
| Q2 | `ORDER BY` + `LIMIT` | 10 most expensive books |
| Q3 | `DISTINCT` | Distinct ratings present |
| Q4 | `IN` + `BETWEEN` | £10–£30 books in Travel or Mystery |
| Q5 | **`JOIN`** | 10 highest-rated books per category (window function) |
| Q6 | **`JOIN`** + `GROUP BY` | Avg GBP/INR price + count per category |

## pandas verification

- Q5 and Q6 each re-read with `pd.read_sql_query(...)`.
- Both **reproduced without SQL** using `pd.merge` on the in-memory
  `books` / `categories` DataFrames (groupby/`ROW_NUMBER` equivalents for
  Q5, `groupby.agg` for Q6).
- Asserted `.equals()` → **both True**; side-by-side printed in
  `pandas_equivalence.txt`.
