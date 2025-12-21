import os
from typing import Literal
from urllib.parse import quote

import requests
import weave
from smolagents import Tool


class TheSportsDBSearchTool(Tool):
    name = "the_sports_db_search_tool"
    description = """
    This is a tool that searches forr any sports league, team, player, event, or venue based on a query"""
    inputs = {
        "query": {
            "type": "string",
            "description": "the query to search",
        },
        "search_type": {
            "type": "string",
            "description": "Type of search to perform (league, team, player, event, venue)",
        },
    }
    output_type = "object"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = "https://www.thesportsdb.com/api/v2/json"

    @weave.op
    def forward(
        self,
        query: str,
        search_type: Literal["league", "team", "player", "event", "venue"],
    ) -> dict:
        slug = "_".join(query.strip().split()).lower()
        slug = quote(slug, safe="-_")
        url = f"{self.base_url.rstrip('/')}/search/{search_type}/{slug}"
        headers = {
            "X-API-KEY": os.getenv("SPORTSDB_API_KEY"),
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("search")
        search_results = search_results if isinstance(search_results, list) else []
        return {
            "search_results": search_results,
        }


class TheSportsDBLookupTool(Tool):
    name = "the_sports_db_lookup_tool"
    description = """
    This is a tool that looks up the following kinds of data for any sport using its unique ID:
        - league using its unique ID {idLeague}
        - team using its unique ID {idTeam}
        - team_equipment using its unique ID {idTeam}
        - player using its unique ID {idPlayer}
        - player_contracts using its unique ID {idPlayer}
        - player_results using its unique ID {idPlayer}
        - player_honours using its unique ID {idPlayer}
        - player_milestones using its unique ID {idPlayer}
        - player_teams using its unique ID {idPlayer}
        - event using its unique ID {idLeague}
        - event_lineup using its unique ID {idEvent}
        - event_results using its unique ID {idEvent}
        - event_stats using its unique ID {idEvent}
        - event_timeline using its unique ID {idEvent}
        - event_tv using its unique ID {idEvent}
        - event_highlights using its unique ID {idEvent}
        - venue using its unique ID {idVenue}
    """.strip()
    inputs = {
        "lookup_id": {
            "type": "string",
            "description": "The unique ID of the entity to lookup. This can be found found using the `the_sports_db_search_tool` tool.",
        },
        "lookup_type": {
            "type": "string",
            "description": "The type of entity to lookup.",
        },
    }
    output_type = "object"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = "https://www.thesportsdb.com/api/v2/json"

    @weave.op
    def forward(
        self,
        lookup_id: str,
        lookup_type: Literal[
            "league",
            "team",
            "team_equipment",
            "player",
            "player_contracts",
            "player_results",
            "player_honours",
            "player_milestones",
            "player_teams",
            "event",
            "event_lineup",
            "event_results",
            "event_stats",
            "event_timeline",
            "event_tv",
            "event_highlights",
            "venue",
        ],
    ) -> dict:
        slug = "_".join(lookup_id.strip().split()).lower()
        slug = quote(slug, safe="-_")
        url = f"{self.base_url.rstrip('/')}/lookup/{lookup_type}/{slug}"
        headers = {
            "X-API-KEY": os.getenv("SPORTSDB_API_KEY"),
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("lookup")
        search_results = results if isinstance(results, list) else []
        return {
            "search_results": search_results,
        }
