"""Collect Wikipedia articles about football tactical concepts."""

import hashlib
import wikipedia


WIKI_PAGES = [
    # Tactical systems
    "Catenaccio",
    "Tiki-taka",
    "Total Football",
    "Gegenpressing",
    # Formations
    "Formation (association football)",
    "4-4-2 (association football)",
    "4-3-3 (association football)",
    "3-5-2 (association football)",
    "4-2-3-1 (association football)",
    "4-1-4-1 (association football)",
    # Player roles
    "Sweeper (association football)",
    "False 9",
    "Playmaker",
    "Libero (association football)",
    "Regista",
    "Trequartista",
    "Mezzala",
    "Wing-back",
    "Box-to-box midfielder",
    # Tactical concepts
    "Offside trap",
    "Pressing (association football)",
    "Counter-attack",
    "Zonal marking",
    "Man-to-man marking (association football)",
    "Park the bus (association football)",
    "Route one",
    "Long ball",
    "Set piece (association football)",
    "Throw-in",
    # Influential managers and their tactical legacies
    "Johan Cruyff",
    "Arrigo Sacchi",
    "Pep Guardiola",
    "Jurgen Klopp",
    "Jose Mourinho",
    "Marcelo Bielsa",
    "Carlo Ancelotti",
    "Rinus Michels",
    "Helenio Herrera",
]


def collect_wiki():
    """Download Wikipedia articles for football tactical concepts."""
    documents = []

    for page_title in WIKI_PAGES:
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            content = page.content
        except wikipedia.DisambiguationError as e:
            # pick the first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                content = page.content
            except Exception:
                print(f"  Skipping '{page_title}': disambiguation failed")
                continue
        except wikipedia.PageError:
            print(f"  Skipping '{page_title}': page not found")
            continue
        except Exception as e:
            print(f"  Skipping '{page_title}': {e}")
            continue

        if len(content) < 200:
            print(f"  Skipping '{page_title}': too short")
            continue

        doc_id = hashlib.md5(f"wiki-{page_title}".encode()).hexdigest()[:12]

        documents.append({
            "doc_id": doc_id,
            "source": "wikipedia",
            "title": page.title,
            "content": content,
            "metadata": {
                "url": page.url,
                "page_title": page_title,
            },
        })
        print(f"  Collected: {page.title} ({len(content)} chars)")

    return documents


if __name__ == "__main__":
    docs = collect_wiki()
    print(f"\nTotal Wikipedia documents: {len(docs)}")
    if docs:
        print(f"Sample:\n{docs[0]['content'][:500]}")
