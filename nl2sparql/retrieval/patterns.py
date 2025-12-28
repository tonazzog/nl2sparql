"""Pattern inference from natural language questions."""

import re
from typing import Optional

# Keywords that indicate specific query patterns (Italian + English)
PATTERN_KEYWORDS = {
    "EMOTION_LEXICON": {
        "keywords": [
            # Italian
            "emozione", "emozioni", "emotivo", "emotivi", "emotiva", "emotive",
            "gioia", "tristezza", "paura", "rabbia", "amore", "sorpresa",
            "sentimento", "sentimenti", "affettivo", "affettiva",
            "positivo", "positiva", "negativo", "negativa", "polarita",
            # English
            "emotion", "emotions", "emotional", "feeling", "feelings",
            "joy", "sadness", "fear", "anger", "love", "surprise",
            "sentiment", "sentiments", "affective",
            "positive", "negative", "polarity",
        ],
        "weight": 1.0,
    },
    "TRANSLATION": {
        "keywords": [
            # Italian
            "traduzione", "traduzioni", "tradurre", "tradotto", "tradotta",
            "siciliano", "siciliana", "siciliani", "siciliane",
            "parmigiano", "parmigiana", "parmigiani", "parmigiane",
            "dialetto", "dialetti", "dialettale", "dialettali",
            "corrispondente", "corrispondenti", "equivalente", "equivalenti",
            # English
            "translation", "translations", "translate", "translated",
            "sicilian", "dialect", "dialects", "dialectal",
            "corresponding", "equivalent", "equivalents",
        ],
        "weight": 1.0,
    },
    "MULTI_TRANSLATION": {
        "keywords": [
            # Italian
            "entrambi i dialetti", "tutti i dialetti", "piu dialetti",
            "siciliano e parmigiano", "diverse traduzioni",
            # English
            "both dialects", "all dialects", "multiple dialects",
            "sicilian and parmigiano", "different translations",
        ],
        "weight": 0.9,
    },
    "SENSE_DEFINITION": {
        "keywords": [
            # Italian
            "definizione", "definizioni", "significato", "significati",
            "senso", "sensi", "accezione", "accezioni",
            "cosa significa", "cosa vuol dire", "definito come",
            # English
            "definition", "definitions", "meaning", "meanings",
            "sense", "senses", "what does it mean", "what means",
            "defined as",
        ],
        "weight": 1.0,
    },
    "SENSE_COUNT": {
        "keywords": [
            # Italian
            "quanti sensi", "numero di sensi", "contare i sensi",
            "quante accezioni", "polisemia", "polisemico",
            # English
            "how many senses", "number of senses", "count senses",
            "how many meanings", "polysemy", "polysemous",
        ],
        "weight": 0.9,
    },
    "SEMANTIC_RELATION": {
        "keywords": [
            # Italian
            "iperonimo", "iperonimi", "iponimo", "iponimi",
            "meronimo", "meronimi", "olonimo", "olonimi",
            "piu generale", "piu specifico", "tipo di", "parte di",
            "relazione semantica", "relazioni semantiche",
            "gerarchia", "gerarchico", "tassonomia",
            # English
            "hypernym", "hypernyms", "hyponym", "hyponyms",
            "meronym", "meronyms", "holonym", "holonyms",
            "more general", "more specific", "type of", "kind of", "part of",
            "semantic relation", "semantic relations",
            "hierarchy", "hierarchical", "taxonomy",
        ],
        "weight": 1.0,
    },
    "POS_FILTER": {
        "keywords": [
            # Italian
            "sostantivo", "sostantivi", "nome", "nomi",
            "verbo", "verbi", "aggettivo", "aggettivi",
            "avverbio", "avverbi", "preposizione", "preposizioni",
            "pronome", "pronomi", "articolo", "articoli",
            # English
            "noun", "nouns", "verb", "verbs",
            "adjective", "adjectives", "adverb", "adverbs",
            "preposition", "prepositions", "pronoun", "pronouns",
            "article", "articles", "part of speech", "pos",
        ],
        "weight": 0.8,
    },
    "MORPHO_REGEX": {
        "keywords": [
            # Italian
            "inizia con", "iniziano con", "finisce con", "finiscono con",
            "contiene", "contengono", "pattern", "regex",
            "prefisso", "suffisso", "infisso",
            "terminazione", "desinenza",
            # English
            "starts with", "start with", "begins with", "begin with",
            "ends with", "end with", "contains", "contain",
            "prefix", "suffix", "infix",
            "ending", "termination",
        ],
        "weight": 0.9,
    },
    "COUNT_ENTITIES": {
        "keywords": [
            # Italian
            "quanti", "quante", "conta", "contare",
            "numero di", "totale", "quantita",
            # English
            "how many", "count", "counting",
            "number of", "total", "quantity",
        ],
        "weight": 0.7,
    },
    "META_GRAPH": {
        "keywords": [
            # Italian
            "grafo", "grafi", "dataset", "ontologia",
            "schema", "struttura", "metadati",
            # English
            "graph", "graphs", "ontology",
            "structure", "metadata",
        ],
        "weight": 0.6,
    },
    "SERVICE_INTEGRATION": {
        "keywords": [
            # Italian
            "compl-it", "compli", "federated", "federata",
            "servizio esterno",
            # English
            "external service", "remote service",
        ],
        "weight": 0.8,
    },
    "COMPOSITIONAL": {
        "keywords": [
            # Italian
            "tutti gli", "tutte le", "tutti i",
            "che sono", "che hanno", "che esprimono",
            "tipi di", "generi di", "categorie di",
            "parti di", "componenti di",
            "in comune", "relazione tra",
            "che non sono", "che non hanno",
            # English
            "all the", "all of the", "every",
            "that are", "that have", "that express", "which are", "which have",
            "types of", "kinds of", "categories of",
            "parts of", "components of",
            "in common", "relationship between", "relation between",
            "that are not", "that don't have", "which are not",
        ],
        "weight": 0.9,
    },
}


def infer_patterns(
    question: str,
    threshold: float = 0.3,
) -> dict[str, float]:
    """
    Infer query patterns from a natural language question.

    Args:
        question: The natural language question
        threshold: Minimum score to include a pattern

    Returns:
        Dictionary mapping pattern names to confidence scores
    """
    question_lower = question.lower()
    patterns = {}

    for pattern_name, config in PATTERN_KEYWORDS.items():
        keywords = config["keywords"]
        base_weight = config["weight"]

        # Count keyword matches
        matches = 0
        for keyword in keywords:
            if keyword in question_lower:
                matches += 1

        if matches > 0:
            # Score based on number of matches, normalized
            score = min(1.0, (matches / 3) * base_weight)
            if score >= threshold:
                patterns[pattern_name] = score

    return patterns


def get_top_patterns(
    patterns: dict[str, float],
    top_k: int = 3,
) -> list[str]:
    """
    Get the top-k patterns by score.

    Args:
        patterns: Pattern scores dictionary
        top_k: Number of top patterns to return

    Returns:
        List of pattern names sorted by score
    """
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_patterns[:top_k]]


def extract_entity_terms(question: str) -> list[str]:
    """
    Extract potential entity terms (specific words to query) from a question.

    Args:
        question: The natural language question

    Returns:
        List of potential entity terms
    """
    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', question)
    quoted.extend(re.findall(r"'([^']+)'", question))

    # Look for terms after indicator phrases (Italian + English)
    indicators = [
        # Italian
        r"parola\s+(\w+)",
        r"lemma\s+(\w+)",
        r"termine\s+(\w+)",
        r"vocabolo\s+(\w+)",
        r"traduzione di\s+(\w+)",
        r"definizione di\s+(\w+)",
        r"significato di\s+(\w+)",
        # English
        r"word\s+(\w+)",
        r"term\s+(\w+)",
        r"translation of\s+(\w+)",
        r"definition of\s+(\w+)",
        r"meaning of\s+(\w+)",
    ]

    for pattern in indicators:
        matches = re.findall(pattern, question.lower())
        quoted.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for term in quoted:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique.append(term)

    return unique
