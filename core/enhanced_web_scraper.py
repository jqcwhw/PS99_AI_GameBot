"""
Enhanced Web Scraper for AI Game Bot
Combines techniques from multiple scraping tools for maximum reliability
Based on your provided scrapers and real-world implementations
"""

import requests
import json
import time
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import trafilatura
from bs4 import BeautifulSoup
import random
from urllib.parse import urlparse, urljoin

class EnhancedWebScraper:
    """Enhanced web scraper with multiple fallback methods and game-specific optimizations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Rate limiting
        self.rate_limit_delay = 2000  # 2 seconds between requests
        self.last_request_time = 0
        
        # User agents rotation for reliability
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Gaming-specific URLs and patterns
        self.gaming_targets = {
            'roblox_apis': [
                'https://apis.roblox.com/universes/v1/places/{}/universe',
                'https://games.roblox.com/v1/games?universeIds={}',
                'https://users.roblox.com/v1/users/{}',
                'https://groups.roblox.com/v1/groups/{}'
            ],
            'ps99_sources': [
                'https://www.roblox.com/games/8737899170/Pet-Simulator-99',
                'https://biggames.io/pet-simulator-99',
                'https://discord.gg/biggames',
                'https://twitter.com/BigGames_Roblox',
                'https://reddit.com/r/PetSimulatorX'
            ],
            'general_sources': [
                'https://wiki.pet-simulator.com',
                'https://ps99.biggamesapi.io'
            ]
        }
        
        # Game-specific keywords for intelligent extraction
        self.game_keywords = [
            'pet simulator', 'huge', 'titanic', 'shiny', 'rainbow', 'exclusive',
            'egg', 'hatch', 'luck', 'boost', 'biggames', 'update', 'event',
            'trading', 'value', 'rare', 'legendary', 'mythical', 'chest',
            'slime tycoon', 'breakable', 'damage', 'automation', 'script',
            'roblox', 'place id', 'universe id', 'private server'
        ]
        
        self.logger.info("Enhanced web scraper initialized with gaming optimizations")
    
    def scrape_url(self, url: str, scrape_type: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Enhanced URL scraping with multiple fallback methods
        Based on your provided scraper implementations
        """
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Detect URL type and use appropriate method
            if scrape_type == "auto":
                scrape_type = self._detect_url_type(url)
            
            result = None
            
            # Try specialized scrapers first
            if scrape_type == "roblox":
                result = self._scrape_roblox_content(url)
            elif scrape_type == "ps99":
                result = self._scrape_ps99_content(url)
            elif scrape_type == "discord":
                result = self._scrape_discord_content(url)
            elif scrape_type == "reddit":
                result = self._scrape_reddit_content(url)
            elif scrape_type == "api":
                result = self._scrape_api_content(url)
            
            # Fallback to general scraping
            if not result:
                result = self._scrape_general_content(url)
            
            if result:
                result['scrape_metadata'] = {
                    'url': url,
                    'scrape_type': scrape_type,
                    'timestamp': time.time(),
                    'scraper_version': 'enhanced_v1.0'
                }
                
                self.logger.info(f"Successfully scraped {url} ({scrape_type})")
                return result
            else:
                self.logger.warning(f"Failed to scrape {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _detect_url_type(self, url: str) -> str:
        """Detect the type of URL for specialized scraping"""
        url_lower = url.lower()
        
        if 'roblox.com' in url_lower:
            return "roblox"
        elif 'biggames.io' in url_lower or 'pet-simulator' in url_lower:
            return "ps99"
        elif 'discord.gg' in url_lower:
            return "discord"
        elif 'reddit.com' in url_lower:
            return "reddit"
        elif '/api/' in url_lower or url_lower.endswith('.json'):
            return "api"
        else:
            return "general"
    
    def _scrape_roblox_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced Roblox scraping based on your Python scraper
        Extracts game IDs, universe info, creator details, etc.
        """
        try:
            # Extract ID from URL using patterns from your scraper
            roblox_id, id_type = self._extract_roblox_id(url)
            
            if not roblox_id:
                return None
            
            result = {
                'type': 'roblox_data',
                'extracted_id': roblox_id,
                'id_type': id_type,
                'source_url': url
            }
            
            # Get detailed info based on ID type
            if id_type == "game":
                game_info = self._get_roblox_game_info(roblox_id)
                if game_info:
                    result.update(game_info)
            elif id_type == "user":
                user_info = self._get_roblox_user_info(roblox_id)
                if user_info:
                    result.update(user_info)
            elif id_type == "group":
                group_info = self._get_roblox_group_info(roblox_id)
                if group_info:
                    result.update(group_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Roblox scraping error: {e}")
            return None
    
    def _extract_roblox_id(self, url: str) -> tuple:
        """Extract Roblox ID and type from URL - from your scraper logic"""
        # Game ID patterns
        game_match = re.search(r'games/(\d+)', url)
        if game_match:
            return game_match.group(1), "game"
        
        # User ID patterns
        user_match = re.search(r'users/(\d+)|profile/(\d+)', url)
        if user_match:
            return user_match.group(1) if user_match.group(1) else user_match.group(2), "user"
        
        # Group ID patterns
        group_match = re.search(r'groups/(\d+)', url)
        if group_match:
            return group_match.group(1), "group"
        
        # Asset ID patterns
        asset_match = re.search(r'catalog/(\d+)|asset/(\d+)', url)
        if asset_match:
            return asset_match.group(1) if asset_match.group(1) else asset_match.group(2), "asset"
        
        return None, None
    
    def _get_roblox_game_info(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed game info using Roblox APIs - from your scraper"""
        try:
            # Get universe ID
            universe_url = f"https://apis.roblox.com/universes/v1/places/{game_id}/universe"
            universe_response = self._make_request(universe_url)
            
            if not universe_response or "universeId" not in universe_response:
                return None
            
            universe_id = universe_response["universeId"]
            
            # Get game details
            game_url = f"https://games.roblox.com/v1/games?universeIds={universe_id}"
            game_response = self._make_request(game_url)
            
            if game_response and "data" in game_response and len(game_response["data"]) > 0:
                game_info = game_response["data"][0]
                
                return {
                    'game_name': game_info.get("name", "Unknown"),
                    'description': game_info.get("description", "")[:200],  # Truncate
                    'universe_id': universe_id,
                    'root_place_id': game_info.get("rootPlaceId"),
                    'creator_id': game_info.get("creator", {}).get("id"),
                    'creator_name': game_info.get("creator", {}).get("name"),
                    'creator_type': game_info.get("creator", {}).get("type"),
                    'players_online': game_info.get("playing", 0),
                    'total_visits': game_info.get("visits", 0),
                    'created_date': game_info.get("created", "").split("T")[0],
                    'updated_date': game_info.get("updated", "").split("T")[0],
                    'private_servers_allowed': game_info.get("createVipServersAllowed", False)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Roblox game info error: {e}")
            return None
    
    def _get_roblox_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user info from Roblox API"""
        try:
            user_url = f"https://users.roblox.com/v1/users/{user_id}"
            user_response = self._make_request(user_url)
            
            if user_response:
                return {
                    'user_id': user_id,
                    'username': user_response.get("name", "Unknown"),
                    'display_name': user_response.get("displayName", ""),
                    'description': user_response.get("description", "")[:200],
                    'created_date': user_response.get("created", "").split("T")[0],
                    'is_banned': user_response.get("isBanned", False)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Roblox user info error: {e}")
            return None
    
    def _get_roblox_group_info(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group info from Roblox API"""
        try:
            group_url = f"https://groups.roblox.com/v1/groups/{group_id}"
            group_response = self._make_request(group_url)
            
            if group_response:
                return {
                    'group_id': group_id,
                    'group_name': group_response.get("name", "Unknown"),
                    'description': group_response.get("description", "")[:200],
                    'member_count': group_response.get("memberCount", 0),
                    'is_public': group_response.get("publicEntryAllowed", False),
                    'owner_id': group_response.get("owner", {}).get("userId"),
                    'owner_name': group_response.get("owner", {}).get("username")
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Roblox group info error: {e}")
            return None
    
    def _scrape_ps99_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape PS99-specific content using techniques from your scrapers"""
        try:
            response = self._make_request(url, return_text=True)
            if not response:
                return None
            
            soup = BeautifulSoup(response, 'html.parser')
            
            result = {
                'type': 'ps99_content',
                'title': self._extract_title(soup),
                'updates': self._extract_ps99_updates(soup),
                'events': self._extract_ps99_events(soup),
                'egg_info': self._extract_egg_information(soup),
                'keywords': self._extract_keywords(response)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"PS99 scraping error: {e}")
            return None
    
    def _extract_ps99_updates(self, soup) -> List[Dict[str, str]]:
        """Extract PS99 updates from page"""
        updates = []
        
        # Look for common update patterns
        update_selectors = [
            '.news-item', '.update-item', '.announcement',
            '[class*="update"]', '[class*="news"]'
        ]
        
        for selector in update_selectors:
            elements = soup.select(selector)
            for element in elements:
                title = self._extract_text(element, ['h1', 'h2', 'h3', '.title'])
                content = self._extract_text(element, ['p', '.content', '.description'])
                date = self._extract_text(element, ['.date', '.timestamp', 'time'])
                
                if title or content:
                    updates.append({
                        'title': title,
                        'content': content[:300] if content else "",
                        'date': date
                    })
        
        return updates
    
    def _extract_ps99_events(self, soup) -> List[Dict[str, str]]:
        """Extract PS99 events from page"""
        events = []
        
        event_selectors = [
            '.event-item', '.event', '[class*="event"]'
        ]
        
        for selector in event_selectors:
            elements = soup.select(selector)
            for element in elements:
                name = self._extract_text(element, ['.event-name', '.name', 'h3'])
                description = self._extract_text(element, ['.event-description', '.description'])
                end_date = self._extract_text(element, ['.event-end', '.end-date'])
                
                if name:
                    events.append({
                        'name': name,
                        'description': description[:200] if description else "",
                        'end_date': end_date
                    })
        
        return events
    
    def _extract_egg_information(self, soup) -> List[Dict[str, str]]:
        """Extract egg-related information"""
        eggs = []
        
        # Look for egg-related content
        text_content = soup.get_text().lower()
        
        for keyword in ['angelus', 'agony', 'demon', 'yeti', 'griffin', 'tiger', 'wolf', 'monkey']:
            if keyword in text_content:
                # Find context around the keyword
                pattern = rf'.{{0,100}}{keyword}.{{0,100}}'
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                
                for match in matches:
                    eggs.append({
                        'type': keyword,
                        'context': match.strip()
                    })
        
        return eggs
    
    def _scrape_discord_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape Discord invite information"""
        try:
            response = self._make_request(url, return_text=True)
            if not response:
                return None
            
            soup = BeautifulSoup(response, 'html.parser')
            
            return {
                'type': 'discord_community',
                'server_name': self._extract_text(soup, ['h1', '.server-name']),
                'member_count': self._extract_text(soup, ['[data-testid="member-count"]', '.member-count']),
                'online_count': self._extract_text(soup, ['[data-testid="online-count"]', '.online-count']),
                'description': (lambda meta: meta.get('content', '') if meta and hasattr(meta, 'get') else str(meta) if meta else '')(soup.find('meta', {'name': 'description'}))
            }
            
        except Exception as e:
            self.logger.error(f"Discord scraping error: {e}")
            return None
    
    def _scrape_reddit_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape Reddit community data"""
        try:
            response = self._make_request(url, return_text=True)
            if not response:
                return None
            
            soup = BeautifulSoup(response, 'html.parser')
            posts = []
            
            # Extract posts
            post_selectors = ['[data-testid="post"]', '.Post']
            for selector in post_selectors:
                elements = soup.select(selector)[:10]  # Limit to 10 posts
                
                for element in elements:
                    title = self._extract_text(element, ['[data-testid="post-content"] h3', '.Post-title'])
                    author = self._extract_text(element, ['[data-testid="author-link"]', '.Post-author'])
                    score = self._extract_text(element, ['[data-testid="upvote-button"]', '.Post-score'])
                    
                    if title:
                        posts.append({
                            'title': title,
                            'author': author,
                            'score': score
                        })
            
            return {
                'type': 'reddit_community',
                'posts': posts
            }
            
        except Exception as e:
            self.logger.error(f"Reddit scraping error: {e}")
            return None
    
    def _scrape_api_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape API endpoints"""
        try:
            response = self._make_request(url)
            if response:
                return {
                    'type': 'api_data',
                    'data': response
                }
            return None
            
        except Exception as e:
            self.logger.error(f"API scraping error: {e}")
            return None
    
    def _scrape_general_content(self, url: str) -> Optional[Dict[str, Any]]:
        """General content scraping using trafilatura and BeautifulSoup"""
        try:
            # Try trafilatura first (best for clean text extraction)
            response_text = self._make_request(url, return_text=True)
            if not response_text:
                return None
            
            # Extract main content using trafilatura
            main_content = trafilatura.extract(response_text)
            
            # Also parse with BeautifulSoup for additional data
            soup = BeautifulSoup(response_text, 'html.parser')
            
            return {
                'type': 'general_content',
                'title': self._extract_title(soup),
                'main_content': main_content[:2000] if main_content else "",  # Limit content
                'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:10]],
                'links': self._extract_relevant_links(soup, url),
                'keywords': self._extract_keywords(response_text)
            }
            
        except Exception as e:
            self.logger.error(f"General scraping error: {e}")
            return None
    
    def _make_request(self, url: str, return_text: bool = False) -> Optional[Any]:
        """Make HTTP request with proper headers and error handling"""
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if return_text:
                return response.text
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
        except json.JSONDecodeError:
            if return_text:
                return response.text if 'response' in locals() and response else None
            else:
                self.logger.error(f"Invalid JSON response from {url}")
                return None
    
    def _extract_title(self, soup) -> str:
        """Extract page title"""
        title_element = soup.find('title')
        if title_element:
            return title_element.get_text().strip()
        
        h1_element = soup.find('h1')
        if h1_element:
            return h1_element.get_text().strip()
        
        return ""
    
    def _extract_text(self, element, selectors: List[str]) -> str:
        """Extract text from element using multiple selectors"""
        for selector in selectors:
            found = element.select_one(selector)
            if found:
                return found.get_text().strip()
        return ""
    
    def _extract_relevant_links(self, soup, base_url: str) -> List[Dict[str, str]]:
        """Extract relevant links with gaming keywords"""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True)[:20]:  # Limit to 20 links
            href = link['href']
            text = link.get_text().strip()
            
            # Make absolute URL
            if href.startswith('/'):
                href = urljoin(base_url, href)
            
            # Filter relevant links
            if any(keyword in text.lower() or keyword in href.lower() for keyword in self.game_keywords):
                links.append({
                    'text': text[:100],  # Limit text length
                    'href': href
                })
        
        return links
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract gaming-relevant keywords from text"""
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.game_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time() * 1000
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = (self.rate_limit_delay - time_since_last) / 1000
            time.sleep(sleep_time)
        
        self.last_request_time = time.time() * 1000
    
    def scrape_multiple_sources(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs efficiently"""
        results = []
        
        for url in urls:
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            # Small delay between requests
            time.sleep(0.5)
        
        return results
    
    def intelligent_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform intelligent search across gaming sources"""
        search_results = []
        
        # Search relevant gaming sources
        search_urls = [
            f"https://www.google.com/search?q=site:reddit.com \"{query}\" pet simulator 99",
            f"https://www.google.com/search?q=site:biggames.io \"{query}\"",
            f"https://www.google.com/search?q=\"{query}\" roblox pet simulator"
        ]
        
        for url in search_urls[:1]:  # Limit to avoid rate limits
            try:
                result = self.scrape_url(url)
                if result:
                    search_results.append(result)
            except Exception as e:
                self.logger.error(f"Search failed for {url}: {e}")
        
        return search_results