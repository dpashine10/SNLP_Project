import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENALEX_API_KEY is not set in .env")

BASE_URL = "https://api.openalex.org/works"

SEARCH_TOPICS = [
    "natural language processing",
    "machine learning",
    "artificial intelligence",
    "generative AI",
    "information retrieval",
    "knowledge graphs",
]

PER_PAGE = 100
MAX_PAGES_PER_TOPIC = 5


def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index.
    Convert it back into normal text.
    """
    if not inverted_index:
        return ""

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda x: x[0])

    return " ".join(word for _, word in words)


def get_works(search_term, max_pages=5):
    works = []

    cursor = "*"

    for _ in range(max_pages):

        params = {
            "api_key": API_KEY,
            "search": search_term,
            "per-page": PER_PAGE,
            "cursor": cursor,
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            break

        for work in results:

            title = work.get("display_name") or ""

            abstract = reconstruct_abstract(
                work.get("abstract_inverted_index")
            )

            authors = []

            for authorship in work.get("authorships", []):
                author = authorship.get("author")

                if author:
                    authors.append({
                        "author_id": author.get("id"),
                        "author_name": author.get("display_name"),
                    })

            topics = []

            for topic in work.get("topics") or []:
                if topic.get("display_name"):
                    topics.append(topic["display_name"])

            keywords = []

            for keyword in work.get("keywords") or []:
                if keyword.get("display_name"):
                    keywords.append(keyword["display_name"])

            works.append({
                "work_id": work.get("id"),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "topics": topics,
                "keywords": keywords,
                "publication_year": work.get("publication_year"),
            })

        cursor = data.get("meta", {}).get("next_cursor")

        if not cursor:
            break

        time.sleep(0.2)

    return works


def main():

    all_works = []

    for topic in SEARCH_TOPICS:

        print(f"\nCollecting: {topic}")

        works = get_works(
            topic,
            max_pages=MAX_PAGES_PER_TOPIC,
        )

        print(f"Collected: {len(works)}")

        all_works.extend(works)

    # Remove duplicate works
    unique = {}

    for work in all_works:
        unique[work["work_id"]] = work

    all_works = list(unique.values())

    rows = []

    for work in tqdm(all_works, desc="Preparing dataset"):

        for author in work["authors"]:

            rows.append({
                "work_id": work["work_id"],
                "author_id": author["author_id"],
                "author_name": author["author_name"],
                "title": work["title"],
                "abstract": work["abstract"],
                "topics": "|".join(work["topics"]),
                "keywords": "|".join(work["keywords"]),
                "publication_year": work["publication_year"],
            })

    df = pd.DataFrame(rows)

    output = "/Users/divy/College/NLP/Project/Data/raw/rawdata.csv"

    df.to_csv(output, index=False)

    print("\nDataset created")
    print("Rows:", len(df))
    print("Unique researchers:", df["author_id"].nunique())
    print("Unique papers:", df["work_id"].nunique())
    print("Saved to:", output)


if __name__ == "__main__":
    main()