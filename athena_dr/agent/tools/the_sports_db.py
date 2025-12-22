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
    def thesportsdb_lookup(
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
        base_url = "https://www.thesportsdb.com/api/v2/json"
        url = f"{base_url.rstrip('/')}/lookup/{lookup_type}/{slug}"
        headers = {
            "X-API-KEY": os.getenv("SPORTSDB_API_KEY"),
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("lookup")
        return results if isinstance(results, list) else []

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
        for result_index, result in tqdm(
            enumerate(search_results),
            total=len(search_results),
            desc="Processing search results",
        ):
            for key, value in result.items():
                if key == "idPlayer":
                    player_details = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player"
                    )
                    player_contracts = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_contracts"
                    )
                    player_results = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_results"
                    )
                    player_honours = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_honours"
                    )
                    player_milestones = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_milestones"
                    )
                    player_teams = thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_teams"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": player_details,
                        "contracts": player_contracts,
                        "results": player_results,
                        "honours": player_honours,
                        "milestones": player_milestones,
                        "teams": player_teams,
                    }
                elif key == "idTeam":
                    team_details = thesportsdb_lookup(
                        lookup_id=value, lookup_type="team"
                    )
                    team_equipment = thesportsdb_lookup(
                        lookup_id=value, lookup_type="team_equipment"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": team_details,
                        "equipment": team_equipment,
                    }
                elif key == "idLeague":
                    league_details = thesportsdb_lookup(
                        lookup_id=value, lookup_type="league"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": league_details,
                    }
                elif key == "idEvent":
                    event_details = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event"
                    )
                    event_lineup = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_lineup"
                    )
                    event_results = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_results"
                    )
                    event_stats = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_stats"
                    )
                    event_timeline = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_timeline"
                    )
                    event_tv = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_tv"
                    )
                    event_highlights = thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_highlights"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": event_details,
                        "lineup": event_lineup,
                        "results": event_results,
                        "stats": event_stats,
                        "timeline": event_timeline,
                        "tv": event_tv,
                        "highlights": event_highlights,
                    }
                elif key == "idVenue":
                    venue_details = thesportsdb_lookup(
                        lookup_id=value, lookup_type="venue"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": venue_details,
                    }
        return {
            "search_results": search_results,
        }
