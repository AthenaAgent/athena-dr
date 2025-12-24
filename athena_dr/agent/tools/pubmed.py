import os
from xml.etree import ElementTree

import weave
from smolagents import Tool

S2_API_KEY = os.getenv("S2_API_KEY")
TIMEOUT = int(os.getenv("S2_API_TIMEOUT", 10))
S2_GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_FIELDS = "paperId,corpusId,url,title,abstract,authors,authors.name,year,venue,citationCount,openAccessPdf,externalIds,isOpenAccess"
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def extract_all_text(tag: ElementTree.Element) -> str:
    """
    Sometimes tag.text will produce a None value if there's rich text
    inside the tag. For example,

    In this paper, https://pubmed.ncbi.nlm.nih.gov/39355906/, the returned
    title data is the following:

    <ArticleTitle><i>LRP1</i> Repression by SNAIL Results in ECM Remodeling
    in Genetic Risk for Vascular Diseases.</ArticleTitle>

    And tag.text will return None.

    This function will extract all text from the tag, including rich text.
    """
    return " ".join([_.strip() for _ in tag.itertext()])


def search_pubmed_with_keywords(keywords: str, offset: int = 0, limit: int = 10):
    import requests

    search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": keywords,
        "retmax": limit,
        "retstart": offset,
        "usehistory": "n",
        "sort": "relevance",
        "email": "your_email@example.com",
    }
    response = requests.get(search_url, params=params)
    root = ElementTree.fromstring(response.content)
    id_list = [id_elem.text for id_elem in root.findall("./IdList/Id")]
    return {
        "ids": id_list,
        "count": root.find("./Count").text,
        "offset": root.find("./RetStart").text,
        "limit": root.find("./RetMax").text,
        "next": int(root.find("./RetStart").text) + int(root.find("./RetMax").text),
    }


def fetch_pubmed_details(id_list):
    import requests

    fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "email": "your_email@example.com",
    }
    response = requests.get(fetch_url, params=params)
    papers = ElementTree.fromstring(response.content)

    paper_data_list = []
    for paper in papers.findall("./PubmedArticle"):
        article = paper.find(".//Article")
        pmid = paper.find(".//PMID").text
        title = (
            extract_all_text(article.find(".//ArticleTitle"))
            if article.find(".//ArticleTitle") is not None
            else ""
        )

        abstract = []
        if article.find(".//Abstract") is not None:
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                if abstract_text.attrib.get("Label"):
                    abstract.append(f"{abstract_text.attrib['Label']}")
                abstract.append(extract_all_text(abstract_text))
        abstract = "\n".join(abstract)

        authors = [
            {
                "name": f"{author.find('./LastName').text}, {author.find('./ForeName').text}"
            }
            for author in article.findall(".//Author")
            if author.find("./LastName") is not None
            and author.find("./ForeName") is not None
        ]
        year = article.find(".//Journal/JournalIssue/PubDate/Year")
        venue = (
            article.find(".//Journal/Title").text
            if article.find(".//Journal/Title") is not None
            else None
        )
        article_dates = article.findall(".//ArticleDate")

        publication_date = None
        if article_dates:
            publication_date = article_dates[0].find("Year").text

        paper_data = {
            "paperId": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "externalIds": {"PubMed": pmid},
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year.text if year is not None else None,
            "venue": venue,
            "publicationDate": publication_date,
        }
        paper_data_list.append(paper_data)
    return paper_data_list


def fetch_semantic_scholar_details(paper_data):
    import requests

    paper_ids = [f"PMID:{paper['externalIds']['PubMed']}" for paper in paper_data]

    try:
        res = requests.post(
            f"{S2_GRAPH_API_URL}/paper/batch",
            params={"fields": S2_PAPER_SEARCH_FIELDS},
            json={"ids": paper_ids},
            headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
            timeout=TIMEOUT,
        )
        results = res.json()

        for idx in range(len(paper_data)):
            semantic_scholar_data = results[idx]
            if semantic_scholar_data:
                for key in semantic_scholar_data.keys():
                    if key not in paper_data[idx]:
                        paper_data[idx][key] = semantic_scholar_data[key]
    except Exception as e:
        for paper in paper_data:
            paper.update({"citationCount": None})
        print(f"Error fetching data from Semantic Scholar: {e}")

    return paper_data


class PubMedSearchTool(Tool):
    name = "pubmed_search"
    description = """Search for medical and scientific papers using PubMed API.
    
    This tool searches the PubMed database for biomedical literature including research papers,
    clinical studies, and medical publications. It returns comprehensive paper metadata enriched
    with citation counts from Semantic Scholar.
    
    Use this tool for:
    - Medical and biomedical research queries
    - Clinical studies and trials
    - Health sciences literature
    - Life sciences publications
    
    Returns papers with snippet IDs (e.g., [pubmed_1]) for citation.
    Use these IDs to cite sources with <cite id="pubmed_1">claim</cite> format.
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "Search query string for finding medical/scientific papers in PubMed",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 10)",
            "nullable": True,
        },
        "offset": {
            "type": "integer",
            "description": "Starting position for pagination (default: 0)",
            "nullable": True,
        },
    }
    output_type = "string"

    @weave.op
    def forward(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> str:
        searchStat = search_pubmed_with_keywords(query, offset=offset, limit=limit)
        ids = searchStat["ids"]

        if not ids:
            return f"No papers found. Total results: {searchStat['count']}"

        paper_data = fetch_pubmed_details(ids)
        paper_data = fetch_semantic_scholar_details(paper_data)

        # Format output with snippet IDs for citation
        formatted_papers = []
        for idx, paper in enumerate(paper_data, start=1):
            snippet_id = f"pubmed_{idx}"
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
                f"PMID: {paper.get('paperId', 'N/A')}",
            ]

            if paper.get("abstract"):
                # Truncate abstract if too long
                abstract = paper["abstract"]
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                paper_info.append(f"Abstract: {abstract}")

            formatted_papers.append("\n".join(paper_info) + "\n")

        if not formatted_papers:
            return "No papers found matching the query."

        header = f"Found {searchStat['count']} total results. Showing {len(formatted_papers)} papers:\n\n"
        return header + "\n".join(formatted_papers)
