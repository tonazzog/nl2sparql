"""Resource providers for the MCP server.

Resources are read-only data that MCP clients can access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import MCPConfig


def get_ontology_catalog() -> dict[str, Any]:
    """Load the full ontology catalog.

    Returns:
        Ontology catalog with classes and properties.
    """
    catalog_path = Path(__file__).parent.parent / "data" / "ontology.json"

    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "error": "Ontology catalog not found",
            "path": str(catalog_path),
            "classes": [],
            "properties": [],
        }


def get_base_constraints() -> str:
    """Get the base system prompt and SPARQL prefixes.

    Returns:
        Text content with system prompt and prefixes.
    """
    from ..constraints.base import SYSTEM_PROMPT, SPARQL_PREFIXES

    return f"""# NL2SPARQL Base Constraints

## System Prompt

{SYSTEM_PROMPT}

## SPARQL Prefixes

```sparql
{SPARQL_PREFIXES}
```

## LiITA Knowledge Base Architecture

The LiITA (Linked Italian) knowledge base has data distributed across multiple sources:

1. **Main LiITA Graph** (`GRAPH <http://liita.it/data>`):
   - Lemmas and morphological information
   - Part of speech annotations
   - Canonical forms and written representations

2. **ELITA Graph** (`GRAPH <http://w3id.org/elita>`):
   - Emotion annotations (joy, sadness, fear, anger, etc.)
   - Emotion entries linked to lexical entries via `ontolex:canonicalForm`

3. **Dialect Translations** (no specific graph, use `vartrans:translatableAs`):
   - Sicilian translations
   - Parmigiano translations
   - Direction: Italian → Dialect (never reverse)

4. **CompL-it** (federated `SERVICE` block):
   - Senses and definitions
   - Semantic relations (hypernyms, hyponyms, meronyms)
   - Endpoint: `https://klab.ilc.cnr.it/graphdb-compl-it/`

## Important Patterns

- Emotions are on lexical ENTRIES, not directly on lemmas
- Translation uses `vartrans:translatableAs` from Italian entry to dialect entry
- Semantic relations use `SERVICE` block to CompL-it endpoint
- Variables bound inside `SERVICE` can be used outside (shared URIs)
"""


def get_server_config(config: "MCPConfig") -> dict[str, Any]:
    """Get the current server configuration.

    Args:
        config: MCPConfig instance

    Returns:
        Configuration as dictionary.
    """
    from .. import AVAILABLE_PROVIDERS

    return {
        "provider": config.provider,
        "model": config.model or AVAILABLE_PROVIDERS.get(config.provider, {}).get("default_model"),
        "endpoint": config.endpoint,
        "timeout": config.timeout,
        "available_providers": list(AVAILABLE_PROVIDERS.keys()),
        "provider_models": {
            provider: info.get("models", [])
            for provider, info in AVAILABLE_PROVIDERS.items()
        },
    }
