import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")

BASE_URL = "https://api.openalex.org"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "Data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "indian_institutions_works.csv"

# Maximum number of works collected per institution.
# Increase later if needed.
MAX_WORKS_PER_INSTITUTION = 5000

# Number of results returned per API request.
PER_PAGE = 200

# Optional: only collect papers from these years.
# Set to None to collect all years.
MIN_YEAR = 2015
MAX_YEAR = 2026


# ============================================================
# Institutions from your VIDWAN/OpenAlex search
# ============================================================

INSTITUTIONS = [
    "Indian Institute of Technology Bombay",
    "Indian Institute of Technology Kanpur",
    "Indian Institute of Technology Delhi",
    "Indian Institute of Technology Kharagpur",
    "Indian Institute of Technology Madras",
    "Indian Institute of Technology Roorkee",
    "Indian Institute of Science Bangalore",
    "Indian Institute of Technology Guwahati",
    "Indian Institute of Technology Dhanbad",
    "Indian Institute of Technology BHU",
    "Vellore Institute of Technology University",
    "Birla Institute of Technology and Science",
    "Narsee Monjee Institute of Management Studies",
]


# ============================================================
# Helper functions
# ============================================================

session = requests.Session()


def api_get(endpoint, params):
    """
    Make a GET request to OpenAlex with basic retry handling.
    """

    if not API_KEY:
        raise RuntimeError(
            "OPENALEX_API_KEY not found. Put it in the project .env file."
        )

    params = dict(params)
    params["api_key"] = API_KEY

    for attempt in range(3):
        try:
            response = session.get(
                f"{BASE_URL}{endpoint}",
                params=params,
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            response.raise_for_status()

        except requests.RequestException as e:
            if attempt == 2:
                raise

            wait = 3 * (attempt + 1)
            print(f"Request failed: {e}")
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)

    raise RuntimeError("OpenAlex request failed after retries.")


def find_institution(name):
    """
    Search OpenAlex and select the best matching Indian institution.
    """

    data = api_get(
        "/institutions",
        {
            "search": name,
            "per-page": 10,
        },
    )

    results = data.get("results", [])

    if not results:
        return None

    # Prefer an exact display-name match.
    exact = [
        r for r in results
        if r.get("display_name", "").lower() == name.lower()
    ]

    if exact:
        return exact[0]

    # Otherwise use the first OpenAlex result.
    return results[0]


def reconstruct_abstract(inverted_index):
    """
    Reconstruct normal abstract text from OpenAlex's
    inverted-index representation.
    """

    if not inverted_index:
        return ""

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda x: x[0])

    return " ".join(word for _, word in words)


def extract_topics(work):
    """
    Extract topic names from an OpenAlex work.
    """

    topics = []

    for topic in work.get("topics", []) or []:
        name = topic.get("display_name")

        if name:
            topics.append(name)

    return "; ".join(dict.fromkeys(topics))


def extract_keywords(work):
    """
    Extract keyword names from an OpenAlex work.
    """

    keywords = []

    for keyword in work.get("keywords", []) or []:
        name = keyword.get("display_name")

        if name:
            keywords.append(name)

    return "; ".join(dict.fromkeys(keywords))


def extract_authors(work, target_institution_id):
    """
    Extract only authorships connected to the target institution.

    A paper can have authors from many institutions.
    We only keep researcher relationships where the author
    has an affiliation with the institution currently being collected.
    """

    rows = []

    for authorship in work.get("authorships", []) or []:

        author = authorship.get("author") or {}

        author_id = author.get("id")
        author_name = author.get("display_name")

        institutions = authorship.get("institutions", []) or []

        institution_ids = {
            inst.get("id")
            for inst in institutions
            if inst.get("id")
        }

        if target_institution_id in institution_ids:

            rows.append({
                "author_id": author_id or "",
                "author_name": author_name or "",
            })

    return rows


# ============================================================
# Main collection
# ============================================================

def main():

    all_rows = []

    institution_summary = []

    print("\nFinding OpenAlex institution IDs...\n")

    resolved_institutions = []

    for name in INSTITUTIONS:

        institution = find_institution(name)

        if institution is None:

            print(f"NOT FOUND: {name}")
            continue

        institution_id = institution.get("id")
        display_name = institution.get("display_name", "")

        print(f"{name}")
        print(f"  → {display_name}")
        print(f"  → {institution_id}")
        print(
            f"  → OpenAlex works: "
            f"{institution.get('works_count', 'unknown')}"
        )
        print()

        resolved_institutions.append({
            "requested_name": name,
            "institution_id": institution_id,
            "display_name": display_name,
        })

    print("=" * 80)
    print("Starting collection...")
    print("=" * 80)

    # ========================================================
    # Collect works institution by institution
    # ========================================================

    for institution in resolved_institutions:

        institution_id = institution["institution_id"]
        institution_name = institution["display_name"]

        print("\n")
        print("=" * 80)
        print(f"INSTITUTION: {institution_name}")
        print("=" * 80)

        cursor = "*"
        collected = 0

        while cursor and collected < MAX_WORKS_PER_INSTITUTION:

            remaining = MAX_WORKS_PER_INSTITUTION - collected

            per_page = min(
                PER_PAGE,
                remaining
            )

            filters = [
                f"institutions.id:{institution_id.split('/')[-1]}"
            ]

            if MIN_YEAR is not None:
                filters.append(
                    f"from_publication_date:{MIN_YEAR}-01-01"
                )

            if MAX_YEAR is not None:
                filters.append(
                    f"to_publication_date:{MAX_YEAR}-12-31"
                )

            data = api_get(
                "/works",
                {
                    "filter": ",".join(filters),
                    "per-page": per_page,
                    "cursor": cursor,
                    "sort": "publication_date:desc",
                },
            )

            works = data.get("results", [])

            if not works:
                break

            for work in works:

                work_id = work.get("id", "")
                title = work.get("display_name", "") or ""

                abstract = reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )

                topics = extract_topics(work)

                keywords = extract_keywords(work)

                publication_year = (
                    work.get("publication_year")
                )

                doi = work.get("doi") or ""

                authors = extract_authors(
                    work,
                    institution_id
                )

                # If no institution-specific authorship was returned,
                # keep the paper with an empty researcher relationship.
                if not authors:

                    all_rows.append({
                        "institution_id": institution_id,
                        "institution_name": institution_name,
                        "work_id": work_id,
                        "author_id": "",
                        "author_name": "",
                        "title": title,
                        "abstract": abstract,
                        "topics": topics,
                        "keywords": keywords,
                        "publication_year": publication_year,
                        "doi": doi,
                    })

                else:

                    for author in authors:

                        all_rows.append({
                            "institution_id": institution_id,
                            "institution_name": institution_name,
                            "work_id": work_id,
                            "author_id": author["author_id"],
                            "author_name": author["author_name"],
                            "title": title,
                            "abstract": abstract,
                            "topics": topics,
                            "keywords": keywords,
                            "publication_year": publication_year,
                            "doi": doi,
                        })

            collected += len(works)

            print(
                f"Collected {collected:,} works..."
            )

            cursor = (
                data.get("meta", {})
                .get("next_cursor")
            )

            if not cursor:
                break

        institution_summary.append({
            "institution_id": institution_id,
            "institution_name": institution_name,
            "works_collected": collected,
        })

        print(
            f"Finished {institution_name}: "
            f"{collected:,} works"
        )

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError(
            "No data was collected."
        )

    # Remove exact duplicate relationships.
    df = df.drop_duplicates(
        subset=[
            "institution_id",
            "work_id",
            "author_id",
        ]
    )

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # Print final statistics
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print(f"Rows: {len(df):,}")

    print(
        "Unique institutions:",
        df["institution_id"].nunique()
    )

    print(
        "Unique papers:",
        df["work_id"].nunique()
    )

    print(
        "Unique researchers:",
        df.loc[
            df["author_id"] != "",
            "author_id"
        ].nunique()
    )

    print(
        "Missing abstracts:",
        (df["abstract"] == "").sum()
    )

    print(
        "Missing author IDs:",
        (df["author_id"] == "").sum()
    )

    print("\nInstitution summary:")

    summary_df = pd.DataFrame(
        institution_summary
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()