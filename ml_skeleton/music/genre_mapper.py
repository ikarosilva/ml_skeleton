"""Genre mapping utility for multi-label classification.

Maps raw genre strings from Clementine database to 7 broad categories.
Handles "/" separator for multi-genre songs (e.g., "Rock/Alternative" -> [rock, rock]).

Categories:
    - rock: rock, metal, punk, grunge, alternative, indie, blues, soul, funk
    - pop: pop, dance, disco, synth, new wave
    - electronic: electronic, techno, house, trance, ambient, edm, dubstep
    - hiphop: hip-hop, rap, r&b, urban, trap
    - jazz_classical: jazz, classical, orchestra, symphony, instrumental, new age
    - country: country, bluegrass, americana, folk, western
    - latin_world: latin, reggae, world, african, caribbean, soundtrack
"""

from typing import List, Optional
import torch


# 7 broad genre categories (index order matters for multi-label encoding)
GENRE_CATEGORIES = [
    "rock",
    "pop",
    "electronic",
    "hiphop",
    "jazz_classical",
    "country",
    "latin_world",
]

NUM_GENRES = len(GENRE_CATEGORIES)

# Mapping patterns to categories (case-insensitive substring match)
GENRE_PATTERNS = {
    "rock": [
        "rock", "metal", "punk", "grunge", "alternative", "indie",
        "hard rock", "progressive", "psychedelic", "garage",
        "blues", "rhythm and blues", "soul", "motown", "funk"
    ],
    "pop": [
        "pop", "dance", "disco", "synth", "new wave", "europop"
    ],
    "electronic": [
        "electronic", "electronica", "techno", "house", "trance",
        "ambient", "edm", "dubstep", "drum and bass", "dnb",
        "breakbeat", "idm", "downtempo", "chillout", "psytrance"
    ],
    "hiphop": [
        "hip hop", "hip-hop", "hiphop", "rap", "r&b", "rnb",
        "urban", "trap", "grime", "boom bap"
    ],
    "jazz_classical": [
        "jazz", "swing", "bebop", "fusion", "smooth jazz", "big band",
        "classical", "orchestra", "symphony", "opera", "baroque",
        "chamber", "choral", "concerto", "sonata",
        "instrumental", "acoustic", "new age", "easy listening", "lounge"
    ],
    "country": [
        "country", "bluegrass", "americana", "folk", "western"
    ],
    "latin_world": [
        "latin", "salsa", "merengue", "bachata", "reggaeton",
        "bossa nova", "samba", "tango", "cumbia", "mariachi",
        "latin pop", "tropical", "brazilian", "mpb",
        "reggae", "ska", "dub", "dancehall", "roots",
        "world", "african", "celtic", "indian", "asian",
        "middle eastern", "ethnic", "tribal",
        "soundtrack", "film", "movie", "score", "game", "musical"
    ],
}


def get_genre_category(genre_str: str) -> Optional[str]:
    """Map a single genre string to its category.

    Args:
        genre_str: Raw genre string (e.g., "Rock", "Alternative Rock")

    Returns:
        Category name or None if no match
    """
    if not genre_str:
        return None

    genre_lower = genre_str.lower().strip()

    for category, patterns in GENRE_PATTERNS.items():
        for pattern in patterns:
            if pattern in genre_lower:
                return category

    return None


def parse_genre_string(genre_str: str) -> List[str]:
    """Parse a genre string that may contain multiple genres separated by '/'.

    Args:
        genre_str: Raw genre string (e.g., "Rock/Alternative", "Electronic")

    Returns:
        List of category names (deduplicated)
    """
    if not genre_str:
        return []

    # Split by "/" and map each part
    parts = genre_str.split("/")
    categories = set()

    for part in parts:
        category = get_genre_category(part.strip())
        if category:
            categories.add(category)

    return list(categories)


def genre_to_multilabel(genre_str: str) -> torch.Tensor:
    """Convert genre string to multi-label binary tensor.

    Args:
        genre_str: Raw genre string (e.g., "Rock/Alternative")

    Returns:
        Binary tensor of shape (NUM_GENRES,) with 1s for matching categories
    """
    categories = parse_genre_string(genre_str)

    label = torch.zeros(NUM_GENRES, dtype=torch.float32)
    for cat in categories:
        if cat in GENRE_CATEGORIES:
            idx = GENRE_CATEGORIES.index(cat)
            label[idx] = 1.0

    return label


def batch_genre_to_multilabel(genre_strings: List[str]) -> torch.Tensor:
    """Convert batch of genre strings to multi-label tensor.

    Args:
        genre_strings: List of raw genre strings

    Returns:
        Binary tensor of shape (batch_size, NUM_GENRES)
    """
    labels = torch.stack([genre_to_multilabel(g) for g in genre_strings])
    return labels


def get_category_index(category: str) -> int:
    """Get the index of a category in GENRE_CATEGORIES.

    Args:
        category: Category name

    Returns:
        Index or -1 if not found
    """
    try:
        return GENRE_CATEGORIES.index(category)
    except ValueError:
        return -1


def get_category_name(index: int) -> str:
    """Get category name from index.

    Args:
        index: Category index

    Returns:
        Category name or "unknown"
    """
    if 0 <= index < NUM_GENRES:
        return GENRE_CATEGORIES[index]
    return "unknown"
