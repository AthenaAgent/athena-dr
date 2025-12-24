import os
from typing import Optional

import weave
from smolagents import Tool

S2_API_KEY = os.getenv("S2_API_KEY")
TIMEOUT = int(os.getenv("S2_API_TIMEOUT", 10))
S2_GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_FIELDS = "paperId,corpusId,url,title,abstract,authors,authors.name,year,venue,citationCount,openAccessPdf,externalIds,isOpenAccess"


class SemanticScholarPaperSearchTool(Tool):
    name = "semantic_scholar_paper_search"
    description = """Search for academic papers using Semantic Scholar API.
    
    This tool searches for academic papers by keywords and allows filtering by:
    - Publication year (single year like '2024' or range like '2022-2025', '2020-', '-2023')
    - Minimum citation count
    - Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')
    - Venue (e.g., 'ACL', 'EMNLP')
    
    Returns paper metadata with snippet IDs (e.g., [s2_paper_1]) for citation.
    Use these IDs to cite sources with <cite id="s2_paper_1">claim</cite> format.
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "Search query string for finding academic papers",
        },
        "year": {
            "type": "string",
            "description": "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
            "nullable": True,
        },
        "min_citation_count": {
            "type": "integer",
            "description": "Minimum number of citations required",
            "nullable": True,
        },
        "sort": {
            "type": "string",
            "description": "Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')",
            "nullable": True,
        },
        "venue": {
            "type": "string",
            "description": "Venue filter (e.g., 'ACL', 'EMNLP')",
            "nullable": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (max: 100)",
            "nullable": True,
        },
    }
    output_type = "string"

    @weave.op
    def forward(
        self,
        query: str,
        year: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        sort: Optional[str] = None,
        venue: Optional[str] = None,
        limit: int = 25,
    ) -> str:
        import requests

        # Build query parameters
        params = {
            "query": query,
            "offset": 0,
            "limit": min(limit, 100),
            "fields": S2_PAPER_SEARCH_FIELDS,
        }

        if year is not None:
            params["year"] = year
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if sort is not None:
            params["sort"] = sort
        if venue is not None:
            params["venue"] = venue

        # Make API request
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else None

        res = requests.get(
            f"{S2_GRAPH_API_URL}/paper/search",
            params=params,
            headers=headers,
            timeout=TIMEOUT,
        )

        res.raise_for_status()
        results = res.json()

        # Construct PDF links for ArXiv and ACL papers if not provided
        if "data" in results:
            for paper in results["data"]:
                if paper.get("openAccessPdf") is None:
                    if paper.get("externalIds"):
                        if "ArXiv" in paper["externalIds"]:
                            paper["openAccessPdf"] = {
                                "url": f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}"
                            }
                        elif "ACL" in paper["externalIds"]:
                            paper["openAccessPdf"] = {
                                "url": f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf"
                            }
        else:
            results["data"] = []

        # Format output with snippet IDs for citation
        formatted_papers = []
        for idx, paper in enumerate(results.get("data", []), start=1):
            snippet_id = f"s2_paper_{idx}"
            authors = ", ".join(
                [a.get("name", "") for a in paper.get("authors", [])[:5]]
            )
            if len(paper.get("authors", [])) > 5:
                authors += " et al."

            paper_info = [
                f"[{snippet_id}] {paper.get('title', 'N/A')}",
                f"Authors: {authors}",
                f"Year: {paper.get('year', 'N/A')}",
                f"Venue: {paper.get('venue', 'N/A')}",
                f"Citations: {paper.get('citationCount', 'N/A')}",
                f"URL: {paper.get('url', 'N/A')}",
            ]

            if paper.get("openAccessPdf"):
                paper_info.append(f"PDF: {paper['openAccessPdf'].get('url', 'N/A')}")

            if paper.get("abstract"):
                # Truncate abstract if too long
                abstract = paper["abstract"]
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                paper_info.append(f"Abstract: {abstract}")

            formatted_papers.append("\n".join(paper_info) + "\n")

        if not formatted_papers:
            return "No papers found matching the query."

        return "\n".join(formatted_papers)


class SemanticScholarSnippetSearchTool(Tool):
    name = "semantic_scholar_snippet_search"
    description = """Focused snippet retrieval from scientific papers using Semantic Scholar API.
    
    This tool searches for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.
    
    Use this tool when you need to:
    - Find specific quotes or passages within papers
    - Locate mentions of specific concepts or terms in academic literature
    - Get contextual snippets from papers rather than full abstracts
    
    Returns snippets with IDs (e.g., [s2_snippet_1]) for citation.
    Use these IDs to cite sources with <cite id="s2_snippet_1">claim</cite> format.
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "Search query string to find within paper content",
        },
        "year": {
            "type": "string",
            "description": "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
            "nullable": True,
        },
        "paper_ids": {
            "type": "string",
            "description": "Comma-separated list of specific paper IDs to search within (up to ~100 IDs)",
            "nullable": True,
        },
        "venue": {
            "type": "string",
            "description": "Venue filter (e.g., 'ACL', 'EMNLP')",
            "nullable": True,
        },
        "limit": {
            "type": "integer",
            "description": "Number of snippets to retrieve",
            "nullable": True,
        },
    }
    output_type = "string"

    @weave.op
    def forward(
        self,
        query: str,
        year: Optional[str] = None,
        paper_ids: Optional[str] = None,
        venue: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        import requests

        # Build query parameters
        params = {
            "query": query,
            "limit": limit,
        }

        if year is not None:
            params["year"] = year
        if venue is not None:
            params["venue"] = venue
        if paper_ids is not None:
            # Convert comma-separated string to list and join back
            paper_ids_list = [pid.strip() for pid in paper_ids.split(",")]
            params["paperIds"] = ",".join(paper_ids_list)

        # Make API request
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else None

        res = requests.get(
            f"{S2_GRAPH_API_URL}/snippet/search",
            params=params,
            headers=headers,
            timeout=TIMEOUT,
        )

        res.raise_for_status()
        results = res.json()

        # Format output with snippet IDs for citation
        formatted_snippets = []
        if "data" in results:
            for idx, snippet in enumerate(results["data"], start=1):
                snippet_id = f"s2_snippet_{idx}"
                paper = snippet.get("paper", {})
                authors = ", ".join(
                    [a.get("name", "") for a in paper.get("authors", [])[:3]]
                )
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."

                snippet_info = [
                    f"[{snippet_id}] From: {paper.get('title', 'N/A')}",
                    f"Authors: {authors}",
                    f"Year: {paper.get('year', 'N/A')}",
                    f"URL: {paper.get('url', 'N/A')}",
                    f"Snippet: {snippet.get('text', 'N/A')}",
                ]

                formatted_snippets.append("\n".join(snippet_info) + "\n")

        if not formatted_snippets:
            return "No snippets found matching the query."

        return "\n".join(formatted_snippets)
