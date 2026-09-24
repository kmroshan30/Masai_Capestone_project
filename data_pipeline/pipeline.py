"""Zepto Module 1 - Data Pipeline.

Scrape books.toscrape.com (public scraping-practice site), clean the data,
convert GBP -> INR at the fixed baseline rate 1 GBP = 105.50 INR, load into a
normalized SQLite schema (categories 1-N books), run >=5 SQL queries covering
SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN and a JOIN, then prove the
JOIN is reproducible in pandas via read_sql + merge.

Run:  python data_pipeline/pipeline.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Windows console defaults to cp1252; scraped titles contain UTF-8 chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://books.toscrape.com/"
DB_PATH = Path(__file__).parent / "books.db"
CSV_PATH = Path(__file__).parent / "books.csv"
QUERIES_PATH = Path(__file__).parent / "sql_queries_and_output.txt"

# Required, graded fixed-rate baseline. Project-defined constant; no API,
# no date reference, no network lookup ever needed for this conversion.
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

# At least 3 categories, scraped across all their pagination pages.
# Slugs discovered from the sidebar of https://books.toscrape.com/
# (enough categories are included so the final dataset is comfortably >= 60
#  books while still spanning >= 3 categories)
CATEGORIES = [
    ("Travel", "travel_2"),
    ("Mystery", "mystery_3"),
    ("Science", "science_22"),
    ("Fiction", "fiction_10"),
    ("Classics", "classics_6"),
    ("History", "history_32"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0; "
        "educational scraping-practice exercise)"
    )
}


# --------------------------------------------------------------------------- #
# 1. SCRAPE
# --------------------------------------------------------------------------- #
def fetch(url: str, session: requests.Session) -> BeautifulSoup | None:
    """GET a page with explicit status-code handling; return parsed soup."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:  # network failure
        print(f"  ! request failed for {url}: {exc}")
        return None
    if resp.status_code != 200:
        print(f"  ! HTTP {resp.status_code} for {url}")
        return None
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup, category: str) -> list[dict]:
    """Extract raw book rows from one listing page."""
    rows: list[dict] = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"] if article.h3 and article.h3.a else None

        # price lives in <p class="price_color"> e.g. "&#163;51.77"
        price_raw = article.select_one("p.price_color")
        price_text = price_raw.get_text(strip=True) if price_raw else None

        # star rating lives in <p class="star-rating Three">
        star_el = article.select_one("p.star-rating")
        rating_text = None
        if star_el is not None:
            classes = star_el.get("class", [])
            for cls in classes:
                if cls != "star-rating":
                    rating_text = cls
                    break

        # availability lives in <p class="instock availability>
        avail_el = article.select_one("p.instock.availability")
        availability = avail_el.get_text(strip=True) if avail_el else None

        rows.append(
            {
                "title": title,
                "price_text": price_text,
                "star_rating": rating_text,
                "availability": availability,
                "category": category,
            }
        )
    return rows


def scrape_all() -> pd.DataFrame:
    """Scrape every pagination page of each selected category."""
    session = requests.Session()
    all_rows: list[dict] = []

    for category, slug in CATEGORIES:
        url = f"{BASE_URL}catalogue/category/books/{slug}/index.html"
        print(f"[scrape] category={category} -> {url}")

        page_soups: list[BeautifulSoup] = []
        current_url = url
        soup = fetch(current_url, session)
        if soup is None:
            print(f"  ! skipping category {category}")
            continue
        page_soups.append(soup)

        # follow pagination ("next" button) within the category
        while True:
            next_el = soup.select_one("li.next > a")
            if next_el is None:
                break
            next_url = urljoin(current_url, next_el["href"])
            if not next_url.startswith(f"{BASE_URL}catalogue/category/"):
                break  # never escape into the full catalogue
            time.sleep(0.25)
            soup = fetch(next_url, session)
            if soup is None:
                break
            current_url = next_url
            page_soups.append(soup)

        category_rows: list[dict] = []
        for page_soup in page_soups:
            category_rows.extend(parse_listing_page(page_soup, category))
        print(f"  captured {len(category_rows)} books in '{category}'")
        all_rows.extend(category_rows)

    print(f"[scrape] total raw rows: {len(all_rows)}")
    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------- #
# 2. CLEAN
# --------------------------------------------------------------------------- #
def clean_price(value) -> float:
    """Strip currency symbol -> float. Returns None if unparseable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def clean_rating(value) -> int | None:
    """'Three' -> 3. Returns None if unexpected text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return RATING_MAP.get(str(value).strip().title())


def clean_in_stock(value) -> int | None:
    """availability text -> 1/0 integer (SQLite has no native bool)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return 1 if "in stock" in str(value).lower() else 0


def median_impute(series: pd.Series) -> pd.Series:
    """Fill unparseable numeric entries with the column median."""
    median = series.dropna().median()
    return series.fillna(median)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- price -----------------------------------------------------------
    out["price_gbp"] = out["price_text"].apply(clean_price)

    # --- rating ----------------------------------------------------------
    out["rating"] = out["star_rating"].apply(clean_rating)

    # --- availability ----------------------------------------------------
    out["in_stock"] = out["availability"].apply(clean_in_stock)

    # --- title / category hygiene ---------------------------------------
    out["title"] = out["title"].astype("string").str.strip()
    out["category"] = out["category"].astype("string").str.strip()

    # --- missing-value policy (stated, deterministic) --------------------
    # Titles/categories/availability are essential identity fields -> if a
    # row cannot identify WHAT was scraped, drop it (it cannot be keyed or
    # attributed to a category).
    before = len(out)
    out = out.dropna(subset=["title", "category", "in_stock"])
    dropped_identity = before - len(out)

    # Numeric fields (price, rating): median-impute so no row is lost to a
    # single messy parse - per the assignment's median-imputation guidance.
    imputed_price = int(out["price_gbp"].isna().sum())
    imputed_rating = int(out["rating"].isna().sum())
    out["price_gbp"] = median_impute(out["price_gbp"])
    out["rating"] = median_impute(out["rating"]).astype(int)

    # Final safety net: if median imputation still failed (e.g. all-missing),
    # drop those rows rather than crash the pipeline.
    out = out.dropna(subset=["price_gbp", "rating"])
    out["rating"] = out["rating"].astype(int)
    out["price_gbp"] = out["price_gbp"].astype(float)
    out["in_stock"] = out["in_stock"].astype(int)  # 1 = True, 0 = False

    # --- fixed-rate conversion (required baseline) -----------------------
    out["price_inr"] = (out["price_gbp"] * GBP_TO_INR).round(2)

    out = out.reset_index(drop=True)
    print(
        f"[clean] rows: {dropped_identity} dropped (identity fields missing), "
        f"price median-imputed: {imputed_price}, "
        f"rating median-imputed: {imputed_rating}, "
        f"final rows: {len(out)}"
    )
    print(
        "[clean] dtypes -> "
        f"price_gbp={out['price_gbp'].dtype}, "
        f"price_inr={out['price_inr'].dtype}, "
        f"rating={out['rating'].dtype} (int), "
        f"in_stock={out['in_stock'].dtype} (0/1 bool-like)"
    )
    return out


# --------------------------------------------------------------------------- #
# 3. LOAD (normalized schema: categories 1-N books)
# --------------------------------------------------------------------------- #
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""

INSERT_CATEGORY_SQL = (
    "INSERT OR IGNORE INTO categories(category_name) VALUES (?)"
)

INSERT_BOOK_SQL = """
INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
VALUES (?, ?, ?, ?, ?, ?)
"""


def load(df: pd.DataFrame, db_path: Path = DB_PATH) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch every run

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    # categories (PK)
    unique_categories = sorted(df["category"].dropna().unique().tolist())
    conn.executemany(INSERT_CATEGORY_SQL, [(c,) for c in unique_categories])

    # map name -> id (FK resolution)
    cat_ids = dict(conn.execute("SELECT category_name, category_id FROM categories"))

    # books (FK -> categories.category_id)
    records = [
        (
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating),
            int(row.in_stock),
            cat_ids[row.category],
        )
        for row in df.itertuples()
    ]
    conn.executemany(INSERT_BOOK_SQL, records)
    conn.commit()

    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    n_cats = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    print(f"[load] inserted {n_books} books across {n_cats} categories -> {db_path}")
    return conn


# --------------------------------------------------------------------------- #
# 4. QUERY (>=5 queries covering every required clause + one JOIN)
# --------------------------------------------------------------------------- #
QUERIES: list[tuple[str, str]] = [
    (
        "Q1 - SELECT/WHERE: in-stock books rated 4 or 5",
        """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, price_gbp DESC
        LIMIT 10;
        """,
    ),
    (
        "Q2 - ORDER BY + LIMIT: 10 most expensive books (GBP)",
        """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """,
    ),
    (
        "Q3 - DISTINCT: distinct ratings present in the catalogue",
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
        """,
    ),
    (
        "Q4 - IN + BETWEEN: price band within a set of categories",
        """
        SELECT title, price_gbp, rating, category_id
        FROM books
        WHERE price_gbp BETWEEN 10.0 AND 30.0
          AND category_id IN (SELECT category_id FROM categories
                              WHERE category_name IN ('Travel', 'Mystery'))
        ORDER BY price_gbp;
        """,
    ),
    (
        "Q5 - JOIN: 10 highest-rated books per category (via window fn)",
        """
        SELECT category_name, title, price_gbp, rating
        FROM (
            SELECT c.category_name, b.title, b.price_gbp, b.rating,
                   ROW_NUMBER() OVER (
                       PARTITION BY b.category_id
                       ORDER BY b.rating DESC, b.price_gbp DESC, b.title
                   ) AS rn
            FROM books b
            JOIN categories c ON c.category_id = b.category_id
        )
        WHERE rn <= 10
        ORDER BY category_name, rn;
        """,
    ),
    (
        "Q6 - JOIN (simple, aggregate): avg price + book count per category",
        """
        SELECT c.category_name,
               COUNT(*)          AS book_count,
               ROUND(AVG(b.price_gbp), 4) AS avg_price_gbp,
               ROUND(AVG(b.price_inr), 2) AS avg_price_inr
        FROM categories c
        JOIN books b ON b.category_id = c.category_id
        GROUP BY c.category_id
        ORDER BY book_count DESC;
        """,
    ),
]


def run_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    lines: list[str] = []
    for label, sql in QUERIES:
        lines.append("=" * 78)
        lines.append(label)
        lines.append("-" * 78)
        lines.append(sql.strip())
        lines.append("")
        df = pd.read_sql_query(sql, conn)
        results[label] = df
        lines.append(df.to_string(index=False))
        lines.append(f"\n({len(df)} rows)\n")
        print(f"\n{label}\n{df.to_string(index=False)}")

    QUERIES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[query] saved queries + output -> {QUERIES_PATH}")
    return results


# --------------------------------------------------------------------------- #
# 5. PANDAS: read_sql + merge equivalence proof
# --------------------------------------------------------------------------- #
def pandas_equivalence(conn: sqlite3.Connection) -> None:
    books_df = pd.read_sql_query("SELECT * FROM books", conn)
    cats_df = pd.read_sql_query("SELECT * FROM categories", conn)

    # --- read_sql on two of the queries ---------------------------------
    join_label = QUERIES[4][0]  # Q5 JOIN
    agg_label = QUERIES[5][0]   # Q6 JOIN aggregate
    df_join_sql = pd.read_sql_query(QUERIES[4][1], conn)
    df_agg_sql = pd.read_sql_query(QUERIES[5][1], conn)
    print(f"\n[pandas] read_sql -> '{join_label}': {df_join_sql.shape}")
    print(f"[pandas] read_sql -> '{agg_label}': {df_agg_sql.shape}")

    # --- reproduce Q6 (simple JOIN/aggregate) with pd.merge --------------
    merged = books_df.merge(
        cats_df, on="category_id", how="inner", validate="many_to_one"
    )
    df_agg_merge = (
        merged.groupby("category_name", as_index=False)
        .agg(
            book_count=("book_id", "count"),
            avg_price_gbp=("price_gbp", "mean"),
            avg_price_inr=("price_inr", "mean"),
        )
        .round({"avg_price_gbp": 4, "avg_price_inr": 2})
        .sort_values("book_count", ascending=False)
        .reset_index(drop=True)
    )

    # --- reproduce Q5 (top-10 per category JOIN) with pd.merge ------------
    merged2 = books_df.merge(
        cats_df, on="category_id", how="inner", validate="many_to_one"
    )
    merged2["rn"] = (
        merged2.groupby("category_id")["rating"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    # rank() ties differ from ROW_NUMBER ties: make deterministic like SQL
    merged2 = merged2.sort_values(
        ["category_id", "rating", "price_gbp", "title"],
        ascending=[True, False, False, True],
    )
    merged2["rn"] = merged2.groupby("category_id").cumcount() + 1
    df_join_merge = (
        merged2[merged2["rn"] <= 10][
            ["category_name", "title", "price_gbp", "rating"]
        ]
        .reset_index(drop=True)
    )

    print("\n[pandas] side-by-side: Q5 JOIN via SQL read_sql vs pd.merge")
    print(df_join_sql.head(30).to_string(index=False))
    print("-" * 60)
    print(df_join_merge.head(30).to_string(index=False))

    eq_join = df_join_sql.reset_index(drop=True).equals(
        df_join_merge.reset_index(drop=True)
    )
    eq_agg = df_agg_sql.reset_index(drop=True).equals(
        df_agg_merge.reset_index(drop=True)
    )
    print(f"\n[pandas] Q5 join output identical (read_sql vs merge): {eq_join}")
    print(f"[pandas] Q6 agg output identical (read_sql vs merge):  {eq_agg}")

    # persist proof
    proof = Path(__file__).parent / "pandas_equivalence.txt"
    proof.write_text(
        "Q5 JOIN: read_sql vs pd.merge identical = {}\n"
        "Q6 AGG:  read_sql vs pd.merge identical = {}\n\n"
        "--- Q5 via read_sql ---\n{}\n\n--- Q5 via merge ---\n{}\n\n"
        "--- Q6 via read_sql ---\n{}\n\n--- Q6 via merge ---\n{}\n".format(
            eq_join,
            eq_agg,
            df_join_sql.to_string(index=False),
            df_join_merge.to_string(index=False),
            df_agg_sql.to_string(index=False),
            df_agg_merge.to_string(index=False),
        ),
        encoding="utf-8",
    )
    print(f"[pandas] equivalence proof -> {proof}")

    if not (eq_join and eq_agg):
        raise SystemExit("EQUIVALENCE CHECK FAILED - do not ship this output")
    print("[pandas] EQUIVALENCE VERIFIED: SQL join == pandas merge")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 78)
    print("Zepto Module 1 - Data Pipeline")
    print(f"Fixed baseline conversion rate: 1 GBP = {GBP_TO_INR} INR")
    print("=" * 78)

    raw = scrape_all()
    if raw.empty:
        raise SystemExit("Scraping produced 0 rows - aborting.")

    cleaned = clean(raw)
    if len(cleaned) < 60:
        raise SystemExit(
            f"Only {len(cleaned)} rows scraped; acceptance needs >= 60."
        )
    n_cat = cleaned["category"].nunique()
    if n_cat < 3:
        raise SystemExit(f"Only {n_cat} categories; acceptance needs >= 3.")
    print(f"[check] OK: {len(cleaned)} books across {n_cat} categories")

    # save a CSV snapshot for inspection
    cleaned.to_csv(CSV_PATH, index=False)
    print(f"[save] cleaned dataset -> {CSV_PATH}")

    conn = load(cleaned)
    try:
        run_queries(conn)
        pandas_equivalence(conn)
    finally:
        conn.close()

    print("\n[pipeline] DONE")


if __name__ == "__main__":
    main()
