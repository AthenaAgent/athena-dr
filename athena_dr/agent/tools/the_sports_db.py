import os
from typing import Literal
from urllib.parse import quote

import requests
import weave
from smolagents import Tool
from tqdm.auto import tqdm


class TheSportsDBSearchTool(Tool):
    name = "the_sports_db_search_tool"
    description = """
    Use this tool when you need detailed, structured sports data about leagues, teams, players, events, or venues.
    This tool provides comprehensive sports information including player statistics, team rosters, event lineups, match results, and historical data.
    
    **When to use this tool:**
    - Queries about specific sports teams, players, leagues, events, or venues
    - Questions requiring detailed sports statistics, player contracts, honors, or career milestones
    - Requests for match/event details including lineups, results, timelines, or highlights
    - Sports-related queries where structured data is more useful than general web search results
    
    **Examples:** "Who won the 2023 NBA championship?", "Show me Lionel Messi's career statistics", "What is the lineup for Manchester United's next match?"
    
    Returns data with snippet IDs (e.g., [sportsdb_player_1]) for citation.
    Use these IDs to cite sources with <cite id="sportsdb_player_1">claim</cite> format."""
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
    output_type = "string"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = "https://www.thesportsdb.com/api/v2/json"

    @weave.op
    def thesportsdb_lookup(
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
    ) -> str:
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
                    player_details = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="player"
                    )
                    player_contracts = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_contracts"
                    )
                    player_results = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_results"
                    )
                    player_honours = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_honours"
                    )
                    player_milestones = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="player_milestones"
                    )
                    player_teams = self.thesportsdb_lookup(
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
                    team_details = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="team"
                    )
                    team_equipment = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="team_equipment"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": team_details,
                        "equipment": team_equipment,
                    }
                elif key == "idLeague":
                    league_details = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="league"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": league_details,
                    }
                elif key == "idEvent":
                    event_details = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event"
                    )
                    event_lineup = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_lineup"
                    )
                    event_results = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_results"
                    )
                    event_stats = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_stats"
                    )
                    event_timeline = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_timeline"
                    )
                    event_tv = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="event_tv"
                    )
                    event_highlights = self.thesportsdb_lookup(
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
                    venue_details = self.thesportsdb_lookup(
                        lookup_id=value, lookup_type="venue"
                    )
                    search_results[result_index][key] = {
                        "id": value,
                        "details": venue_details,
                    }

        # Format output with snippet IDs for citation
        formatted_results = []
        for idx, result in enumerate(search_results, start=1):
            snippet_id = f"sportsdb_{search_type}_{idx}"
            result_info = [f"[{snippet_id}] {search_type.capitalize()} Result"]
            result_info.append(f"Source: TheSportsDB")

            # Extract key information based on search type
            if search_type == "player":
                if result.get("strPlayer"):
                    result_info.append(f"Name: {result.get('strPlayer')}")
                if result.get("strNationality"):
                    result_info.append(f"Nationality: {result.get('strNationality')}")
                if result.get("strPosition"):
                    result_info.append(f"Position: {result.get('strPosition')}")
                if result.get("strTeam"):
                    result_info.append(f"Team: {result.get('strTeam')}")
                if result.get("dateBorn"):
                    result_info.append(f"Born: {result.get('dateBorn')}")
                if result.get("strDescriptionEN"):
                    desc = result["strDescriptionEN"]
                    if len(desc) > 500:
                        desc = desc[:500] + "..."
                    result_info.append(f"Description: {desc}")
                # Include detailed data if available
                if isinstance(result.get("idPlayer"), dict):
                    player_data = result["idPlayer"]
                    if player_data.get("honours"):
                        honours = player_data["honours"][:5]  # Limit to 5
                        honours_str = ", ".join(
                            [
                                h.get("strHonour", "")
                                for h in honours
                                if h.get("strHonour")
                            ]
                        )
                        if honours_str:
                            result_info.append(f"Honours: {honours_str}")

            elif search_type == "team":
                if result.get("strTeam"):
                    result_info.append(f"Name: {result.get('strTeam')}")
                if result.get("strLeague"):
                    result_info.append(f"League: {result.get('strLeague')}")
                if result.get("strStadium"):
                    result_info.append(f"Stadium: {result.get('strStadium')}")
                if result.get("strCountry"):
                    result_info.append(f"Country: {result.get('strCountry')}")
                if result.get("intFormedYear"):
                    result_info.append(f"Founded: {result.get('intFormedYear')}")
                if result.get("strDescriptionEN"):
                    desc = result["strDescriptionEN"]
                    if len(desc) > 500:
                        desc = desc[:500] + "..."
                    result_info.append(f"Description: {desc}")

            elif search_type == "league":
                if result.get("strLeague"):
                    result_info.append(f"Name: {result.get('strLeague')}")
                if result.get("strSport"):
                    result_info.append(f"Sport: {result.get('strSport')}")
                if result.get("strCountry"):
                    result_info.append(f"Country: {result.get('strCountry')}")
                if result.get("strDescriptionEN"):
                    desc = result["strDescriptionEN"]
                    if len(desc) > 500:
                        desc = desc[:500] + "..."
                    result_info.append(f"Description: {desc}")

            elif search_type == "event":
                if result.get("strEvent"):
                    result_info.append(f"Event: {result.get('strEvent')}")
                if result.get("dateEvent"):
                    result_info.append(f"Date: {result.get('dateEvent')}")
                if result.get("strVenue"):
                    result_info.append(f"Venue: {result.get('strVenue')}")
                if (
                    result.get("intHomeScore") is not None
                    and result.get("intAwayScore") is not None
                ):
                    result_info.append(
                        f"Score: {result.get('intHomeScore')} - {result.get('intAwayScore')}"
                    )
                if result.get("strDescriptionEN"):
                    desc = result["strDescriptionEN"]
                    if len(desc) > 500:
                        desc = desc[:500] + "..."
                    result_info.append(f"Description: {desc}")

            elif search_type == "venue":
                if result.get("strVenue"):
                    result_info.append(f"Name: {result.get('strVenue')}")
                if result.get("strLocation"):
                    result_info.append(f"Location: {result.get('strLocation')}")
                if result.get("intCapacity"):
                    result_info.append(f"Capacity: {result.get('intCapacity')}")
                if result.get("strDescriptionEN"):
                    desc = result["strDescriptionEN"]
                    if len(desc) > 500:
                        desc = desc[:500] + "..."
                    result_info.append(f"Description: {desc}")

            formatted_results.append("\n".join(result_info) + "\n")

        if not formatted_results:
            return f"No {search_type} results found for query: {query}"

        return "\n".join(formatted_results)
