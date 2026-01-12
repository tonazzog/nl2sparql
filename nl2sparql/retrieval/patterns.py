"""Pattern inference from natural language questions."""

import re
from typing import Optional

import numpy as np

# Ontology namespace prefixes for boosting
ONTOLOGY_NAMESPACES = {
    "complit": "http://w3id.org/complit/",
    "lexinfo": "http://www.lexinfo.net/ontology/3.0/lexinfo#",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "vartrans": "http://www.w3.org/ns/lemon/vartrans#",
    "marl": "http://www.gsi.upm.es/ontologies/marl/ns#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "lila": "http://lila-erc.eu/ontologies/lila/",
    "elita": "http://w3id.org/elita/",
    "lime": "http://www.w3.org/ns/lemon/lime#",
    "synsem": "http://www.w3.org/ns/lemon/synsem#",
}

# Pattern to ontology mapping with boost factors
# primary: 1.5x boost, secondary: 1.2x boost
PATTERN_ONTOLOGY_BOOST = {
    "EMOTION_LEXICON": {
        "primary": ["elita","marl"],
        "secondary": ["complit"],
    },
    "TRANSLATION": {
        "primary": ["vartrans"],
        "secondary": [],
    },
    "MULTI_TRANSLATION": {
        "primary": ["vartrans"],
        "secondary": [],
    },
    "SENSE_DEFINITION": {
        "primary": ["complit", "ontolex"],
        "secondary": ["skos", "lexinfo"],
    },
    "SENSE_COUNT": {
        "primary": ["complit", "ontolex"],
        "secondary": [],
    },
    "SEMANTIC_RELATION": {
        "primary": ["complit", "lexinfo"],
        "secondary": ["skos"],
    },
    "POS_FILTER": {
        "primary": ["lila", "lexinfo"],
        "secondary": ["ontolex"],
    },
    "MORPHO_REGEX": {
        "primary": ["lila","lexinfo"],
        "secondary": ["ontolex"],
    },
    "COUNT_ENTITIES": {
        "primary": [],
        "secondary": [],
    },
    "META_GRAPH": {
        "primary": ["lime"],
        "secondary": ["ontolex"],
    },
    "SERVICE_INTEGRATION": {
        "primary": ["complit"],
        "secondary": [],
    },
    "COMPOSITIONAL": {
        "primary": [],
        "secondary": [],
    },
    "ETYMOLOGY": {
        "primary": ["lexinfo"],
        "secondary": [],
    },
    "LEXICAL_RELATION": {
        "primary": ["complit", "lexinfo"],
        "secondary": ["vartrans"],
    },
    "DOMAIN_REGISTER": {
        "primary": ["lexinfo"],
        "secondary": [],
    },
    "SYNTACTIC_FRAME": {
        "primary": ["synsem", "lexinfo"],
        "secondary": [],
    },
    "LEXICAL_FORM": {
        "primary": ["ontolex", "lexinfo"],
        "secondary": [],
    }
}

# Boost factors
PRIMARY_BOOST = 1.5
SECONDARY_BOOST = 1.2


def get_ontology_boosts(
    patterns: dict[str, float],
) -> dict[str, float]:
    """
    Calculate ontology boost factors based on detected patterns.

    Args:
        patterns: Dictionary mapping pattern names to confidence scores

    Returns:
        Dictionary mapping ontology names to boost factors
    """
    boosts: dict[str, float] = {}

    for pattern_name, confidence in patterns.items():
        if pattern_name not in PATTERN_ONTOLOGY_BOOST:
            continue

        mapping = PATTERN_ONTOLOGY_BOOST[pattern_name]

        # Apply primary boosts (weighted by pattern confidence)
        for onto in mapping["primary"]:
            current = boosts.get(onto, 1.0)
            # Combine boosts multiplicatively, weighted by confidence
            boost_factor = 1.0 + (PRIMARY_BOOST - 1.0) * confidence
            boosts[onto] = max(current, boost_factor)

        # Apply secondary boosts
        for onto in mapping["secondary"]:
            current = boosts.get(onto, 1.0)
            boost_factor = 1.0 + (SECONDARY_BOOST - 1.0) * confidence
            boosts[onto] = max(current, boost_factor)

    return boosts


def get_ontology_for_uri(uri: str) -> str | None:
    """
    Get the ontology name for a given URI.

    Args:
        uri: The full URI of an ontology entry

    Returns:
        Ontology name or None if not recognized
    """
    for name, namespace in ONTOLOGY_NAMESPACES.items():
        if uri.startswith(namespace):
            return name
    return None


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
    "ETYMOLOGY": {
        "keywords": [
            # Italian
            "etimologia", "etimologie", "etimologico", "etimologica",
            "origine", "origini", "derivato da", "derivata da",
            "provenienza", "radice", "radici",
            "storia della parola", "evoluzione lessicale",
            # English
            "etymology", "etymologies", "etymological",
            "origin", "origins", "derived from", "derives from",
            "root", "roots", "word history", "lexical evolution",
        ],
        "weight": 1.0,
    },
    "LEXICAL_RELATION": {
        "keywords": [
            # SYNOMYM
            # Italian
            "sinonimo", "sinonimi", "sinonimia",
            "stesso significato", "significato simile",
            "parola equivalente", "parole equivalenti",
            "analogo", "analoghi", "analoga", "analoghe",
            # English
            "synonym", "synonyms", "synonymy",
            "same meaning", "similar meaning",
            "equivalent word", "equivalent words",
            "analogous",
             
             # ANTONYM
             # Italian
            "antonimo", "antonimi", "antonimia",
            "contrario", "contrari", "contraria", "contrarie",
            "opposto", "opposti", "opposta", "opposte",
            "significato opposto", "significato contrario",
            # English
            "antonym", "antonyms", "antonymy",
            "contrary", "contraries",
            "opposite", "opposites",
            "opposite meaning", "contrary meaning",
        ],
        "weight": 1.0,
    },
    "DOMAIN_REGISTER": {
        "keywords": [
            # Italian
            "registro", "registri", "registro linguistico",
            "dominio", "domini", "ambito", "ambiti",
            "tecnico", "tecnica", "tecnici", "tecniche",
            "formale", "informale", "colloquiale",
            "settore", "settori", "specialistico",
            "gergo", "slang", "linguaggio settoriale",
            # English
            "register", "registers", "linguistic register",
            "domain", "domains", "field", "fields",
            "technical", "formal", "informal", "colloquial",
            "sector", "specialized", "jargon", "slang",
        ],
        "weight": 0.9,
    },
    "SYNTACTIC_FRAME": {
        "keywords": [
            # Italian
            "frame sintattico", "struttura sintattica",
            "struttura argomentale", "argomenti del verbo",
            "soggetto", "oggetto diretto", "oggetto indiretto",
            "transitivo", "intransitivo", "ditransitivo",
            "valenza", "valenze", "ruolo tematico",
            # English
            "syntactic frame", "syntactic structure",
            "argument structure", "verb arguments",
            "subject", "direct object", "indirect object",
            "transitive", "intransitive", "ditransitive",
            "valency", "thematic role", "theta role",
        ],
        "weight": 0.9,
    },
    "LEXICAL_FORM": {
        "keywords": [
            # Italian
            "forma", "forme", "forma flessa", "forme flesse",
            "coniugazione", "coniugazioni", "declinazione", "declinazioni",
            "flessione", "flessioni", "paradigma",
            "forma canonica", "lemma", "lemmi",
            "singolare", "plurale", "maschile", "femminile",
            # English
            "form", "forms", "inflected form", "inflected forms",
            "conjugation", "conjugations", "declension", "declensions",
            "inflection", "inflections", "paradigm",
            "canonical form", "lemma", "lemmas",
            "singular", "plural", "masculine", "feminine",
        ],
        "weight": 0.9,
    }
}


# Pattern prototypes for semantic similarity matching
# These are representative example sentences that capture the semantic intent
# of each pattern, including variations that keyword matching would miss
PATTERN_PROTOTYPES = {
    "EMOTION_LEXICON": [
        # Core emotion queries
        "Quali parole esprimono tristezza?",
        "Trova i lemmi associati alla gioia",
        "Parole con emozione negativa",
        "Lemmi che trasmettono paura o rabbia",
        "Vocaboli legati a sentimenti positivi",
        "Termini che esprimono amore",
        "Parole associate alla sorpresa",
        "Lemmi con connotazione emotiva",
        # Polarity variants
        "Parole con polarità negativa",
        "Termini con sentimento positivo",
        "Lemmi con valenza affettiva",
    ],
    "TRANSLATION": [
        # Direct translation queries
        "Come si dice casa in siciliano?",
        "Traduzione in dialetto parmigiano",
        "Qual è l'equivalente siciliano di acqua?",
        "Trova la versione dialettale di mangiare",
        "Corrispondente in dialetto",
        # Dialect-specific
        "Parole siciliane per indicare il pane",
        "Come si traduce in dialetto?",
        "Forma dialettale di questa parola",
        "Equivalente nel dialetto locale",
    ],
    "MULTI_TRANSLATION": [
        "Traduzioni in tutti i dialetti disponibili",
        "Come si dice in siciliano e parmigiano?",
        "Confronta le traduzioni dialettali",
        "Trova equivalenti in entrambi i dialetti",
        "Versioni in diversi dialetti",
    ],
    "SENSE_DEFINITION": [
        # Definition queries
        "Qual è la definizione di libertà?",
        "Cosa significa la parola democrazia?",
        "Spiega il significato di questo termine",
        "Trova la definizione lessicale",
        "Che cosa vuol dire questa parola?",
        # Sense queries
        "Quali sono i sensi della parola banco?",
        "Mostra le accezioni di questo lemma",
        "I diversi significati di pesca",
        "Sensi lessicali del termine",
    ],
    "SENSE_COUNT": [
        "Quanti sensi ha la parola chiave?",
        "Numero di significati di questo lemma",
        "Conta le accezioni",
        "Parole con più di tre sensi",
        "Lemmi polisemici con molti significati",
    ],
    "SEMANTIC_RELATION": [
        # Hypernym/hyponym (hierarchy)
        "Qual è l'iperonimo di cane?",
        "Trova gli iponimi di veicolo",
        "Concetto più generale di automobile",
        "Parole più specifiche di animale",
        "Tipi di frutta",
        "Categorie di veicoli",
        # Meronym/holonym (part-whole)
        "Parti del corpo umano",
        "Componenti di una casa",
        "Elementi che compongono qualcosa",
        "Di cosa è fatto questo oggetto?",
        "Quali sono le parti di un fiore?",
        # General semantic relations
        "Relazioni semantiche tra parole",
        "Gerarchia lessicale",
        "Termini correlati semanticamente",
        "Parole nella stessa categoria",
        # Variants that keyword matching misses
        "Lemmi che indicano parti del corpo",
        "Parole che descrivono tipi di animali",
        "Termini per indicare categorie di oggetti",
    ],
    "POS_FILTER": [
        # Part of speech queries
        "Trova tutti i sostantivi",
        "Lemmi che sono verbi",
        "Solo gli aggettivi",
        "Parole classificate come avverbi",
        "Nomi maschili",
        "Verbi transitivi",
        "Aggettivi qualificativi",
    ],
    "MORPHO_REGEX": [
        # Pattern-based morphological queries
        "Parole che iniziano con pre-",
        "Lemmi che finiscono in -zione",
        "Termini contenenti la radice 'amor'",
        "Verbi con suffisso -are",
        "Parole con prefisso dis-",
        "Lemmi che terminano in -mente",
        "Nomi che finiscono in -tà",
    ],
    "COUNT_ENTITIES": [
        "Quanti lemmi ci sono nel database?",
        "Conta il numero di verbi",
        "Totale delle parole con questa proprietà",
        "Numero di sostantivi maschili",
        "Quante parole soddisfano questa condizione?",
    ],
    "META_GRAPH": [
        "Struttura del grafo di conoscenza",
        "Quali ontologie sono disponibili?",
        "Metadati del dataset",
        "Schema della base di conoscenza",
        "Informazioni sul grafo RDF",
    ],
    "SERVICE_INTEGRATION": [
        "Dati da CompL-it",
        "Interroga il servizio esterno",
        "Informazioni dalla risorsa federata",
        "Query al database semantico esterno",
    ],
    "COMPOSITIONAL": [
        "Trova tutti gli animali che sono mammiferi.",
        "Find all animals that are mammals.",
        "Elenca tutte le auto che sono elettriche.",
        "List all cars that are electric.",
        "Mostra i mammiferi marini.",
        "Show marine mammals.",
        "Elenca gli strumenti musicali a corda.",
        "List string musical instruments.",
        "Elenca i tipi di frutta.",
        "List the types of fruit.",
        "Quali sono i tipi di energia?",
        "What are the types of energy?",
        "Show the parts of a flower.",
        "Elenca le parti di un computer.",
        "List the parts of a computer.",
        "Che cosa hanno in comune un cane e un gatto?",
        "What do a dog and a cat have in common?",
        "Cosa hanno in comune una sedia e un tavolo?",
        "What do a chair and a table have in common?",
        "Trova gli uccelli che non sono migratori.",
        "Find birds that are not migratory.",
        "Elenca i veicoli che non sono a motore.",
        "List vehicles that are not motorized."
    ],
    "ETYMOLOGY": [
        "Qual è l'origine della parola?",
        "Da dove deriva questo termine?",
        "Etimologia di filosofia",
        "Radice storica della parola",
        "Provenienza linguistica",
        "Storia di questo vocabolo",
    ],
    "LEXICAL_RELATION": [
        # Synonymy
        "Sinonimi di felice",
        "Parole con lo stesso significato di bello",
        "Termini equivalenti a veloce",
        "Trova sinonimi per questa parola",
        "Parole che significano la stessa cosa",
        # Antonymy
        "Contrari di buono",
        "Qual è l'opposto di grande?",
        "Antonimi di felice",
        "Parola con significato opposto",
        "Il contrario di questo termine",
    ],
    "DOMAIN_REGISTER": [
        "Termini del linguaggio medico",
        "Parole del registro formale",
        "Vocabolario tecnico informatico",
        "Lessico giuridico",
        "Termini colloquiali",
        "Gergo giovanile",
        "Linguaggio settoriale",
    ],
    "SYNTACTIC_FRAME": [
        "Struttura argomentale del verbo dare",
        "Verbi che richiedono oggetto diretto",
        "Frame sintattico di comunicare",
        "Verbi ditransitivi",
        "Ruoli tematici del predicato",
        "Argomenti del verbo",
    ],
    "LEXICAL_FORM": [
        # Morphological form queries
        "Forma plurale di uomo",
        "Femminile di attore",
        "Forme flesse del verbo essere",
        "Coniugazione del verbo andare",
        "Declinazione del nome",
        "Paradigma verbale",
        "Tutte le forme di questo lemma",
        # Gender/number queries
        "Lemmi maschili singolari",
        "Parole al plurale",
        "Forme femminili plurali",
    ],
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


class PatternEmbeddingIndex:
    """
    Embedding-based index for semantic pattern matching.

    Uses sentence-transformers to encode pattern prototypes and match
    incoming questions to the most semantically similar patterns.
    """

    _instance: Optional["PatternEmbeddingIndex"] = None

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the pattern embedding index.

        Args:
            model_name: Sentence-transformer model to use
        """
        self.model_name = model_name
        self._encoder = None
        self._pattern_embeddings: dict[str, np.ndarray] = {}
        self._prototype_map: dict[int, str] = {}  # global index -> pattern name
        self._all_embeddings: Optional[np.ndarray] = None
        self._initialized = False

    @classmethod
    def get_instance(
        cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> "PatternEmbeddingIndex":
        """
        Get or create singleton instance.

        Args:
            model_name: Sentence-transformer model to use

        Returns:
            PatternEmbeddingIndex instance
        """
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    @property
    def encoder(self):
        """Lazy-load the encoder to avoid import overhead."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _initialize(self) -> None:
        """Build embeddings for all pattern prototypes."""
        if self._initialized:
            return

        all_prototypes = []
        prototype_to_pattern = []

        for pattern_name, prototypes in PATTERN_PROTOTYPES.items():
            for proto in prototypes:
                all_prototypes.append(proto)
                prototype_to_pattern.append(pattern_name)

        # Encode all prototypes at once
        embeddings = self.encoder.encode(
            all_prototypes,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self._all_embeddings = embeddings

        # Build mapping from index to pattern name
        self._prototype_map = {i: name for i, name in enumerate(prototype_to_pattern)}

        # Also store embeddings grouped by pattern (for potential future use)
        idx = 0
        for pattern_name, prototypes in PATTERN_PROTOTYPES.items():
            pattern_embeds = embeddings[idx : idx + len(prototypes)]
            self._pattern_embeddings[pattern_name] = pattern_embeds
            idx += len(prototypes)

        self._initialized = True

    def get_pattern_scores(
        self,
        question: str,
        top_k_matches: int = 5,
    ) -> dict[str, float]:
        """
        Compute semantic similarity scores between question and pattern prototypes.

        For each pattern, finds the best-matching prototype and uses that
        similarity as the pattern score.

        Args:
            question: The natural language question
            top_k_matches: Number of top matches to consider per pattern

        Returns:
            Dictionary mapping pattern names to similarity scores (0-1)
        """
        self._initialize()

        # Encode the question
        question_embedding = self.encoder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Compute similarities with all prototypes
        similarities = np.dot(self._all_embeddings, question_embedding)

        # Aggregate scores by pattern (take max similarity for each pattern)
        pattern_scores: dict[str, float] = {}
        for idx, sim in enumerate(similarities):
            pattern_name = self._prototype_map[idx]
            current_max = pattern_scores.get(pattern_name, 0.0)
            pattern_scores[pattern_name] = max(current_max, float(sim))

        return pattern_scores


def infer_patterns_semantic(
    question: str,
    threshold: float = 0.4,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[str, float]:
    """
    Infer query patterns using semantic similarity with prototypes.

    This method uses embedding-based similarity matching to identify
    patterns, which handles morphological variations and paraphrases
    better than keyword matching.

    Args:
        question: The natural language question
        threshold: Minimum similarity score to include a pattern (0-1)
        model_name: Sentence-transformer model to use

    Returns:
        Dictionary mapping pattern names to similarity scores
    """
    index = PatternEmbeddingIndex.get_instance(model_name)
    scores = index.get_pattern_scores(question)

    # Filter by threshold
    return {name: score for name, score in scores.items() if score >= threshold}


def infer_patterns_hybrid(
    question: str,
    keyword_threshold: float = 0.3,
    semantic_threshold: float = 0.4,
    keyword_weight: float = 0.4,
    semantic_weight: float = 0.6,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[str, float]:
    """
    Infer query patterns using a hybrid of keyword and semantic matching.

    Combines both approaches:
    - Keyword matching: Fast, precise for exact matches
    - Semantic matching: Handles variations and paraphrases

    The final score is a weighted combination of both methods.

    Args:
        question: The natural language question
        keyword_threshold: Minimum score for keyword-based patterns
        semantic_threshold: Minimum score for semantic-based patterns
        keyword_weight: Weight for keyword scores in final combination
        semantic_weight: Weight for semantic scores in final combination
        model_name: Sentence-transformer model to use

    Returns:
        Dictionary mapping pattern names to combined scores
    """
    # Get keyword-based scores
    keyword_scores = infer_patterns(question, threshold=0.0)  # Get all, filter later

    # Get semantic scores
    semantic_scores = infer_patterns_semantic(
        question, threshold=0.0, model_name=model_name
    )

    # Combine scores
    all_patterns = set(keyword_scores.keys()) | set(semantic_scores.keys())
    combined_scores: dict[str, float] = {}

    for pattern in all_patterns:
        kw_score = keyword_scores.get(pattern, 0.0)
        sem_score = semantic_scores.get(pattern, 0.0)

        # Weighted combination
        combined = (kw_score * keyword_weight) + (sem_score * semantic_weight)

        # Apply threshold: pattern must pass at least one individual threshold
        # OR the combined score must be significant
        passes_keyword = kw_score >= keyword_threshold
        passes_semantic = sem_score >= semantic_threshold
        passes_combined = combined >= (keyword_threshold + semantic_threshold) / 2

        if passes_keyword or passes_semantic or passes_combined:
            combined_scores[pattern] = min(1.0, combined)

    return combined_scores
