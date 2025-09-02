"""
Knowledge Management System for AI Game Bot
Handles storage, retrieval, and processing of knowledge from various sources
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urlparse
import hashlib

# Import enhanced web scraper functionality
from core.enhanced_web_scraper import EnhancedWebScraper

class KnowledgeManager:
    """Manages knowledge base for the AI game bot"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base_file = Path("data/knowledge_base.json")
        self.web_scraper = EnhancedWebScraper()
        
        # Knowledge storage
        self.knowledge_base = {
            "game_info": {},
            "strategies": [],
            "updates": [],
            "tips": [],
            "external_sources": [],
            "file_sources": [],
            "learned_patterns": {},
            "metadata": {
                "last_updated": None,
                "version": "1.0",
                "total_items": 0
            }
        }
        
        # Load existing knowledge
        self._load_knowledge_base()
        
        self.logger.info("Knowledge manager initialized")
    
    def _load_knowledge_base(self):
        """Load knowledge base from file"""
        try:
            if self.knowledge_base_file.exists():
                with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    
                # Merge with default structure
                for key, value in loaded_data.items():
                    self.knowledge_base[key] = value
                
                self.logger.info(f"Loaded knowledge base with {self.knowledge_base['metadata']['total_items']} items")
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
    
    def _save_knowledge_base(self):
        """Save knowledge base to file"""
        try:
            self.knowledge_base_file.parent.mkdir(exist_ok=True)
            
            # Update metadata
            self.knowledge_base['metadata']['last_updated'] = time.time()
            self.knowledge_base['metadata']['total_items'] = self._count_total_items()
            
            with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
                
            self.logger.debug("Knowledge base saved")
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")
    
    def _count_total_items(self) -> int:
        """Count total items in knowledge base"""
        total = 0
        for key, value in self.knowledge_base.items():
            if key == 'metadata':
                continue
            elif isinstance(value, list):
                total += len(value)
            elif isinstance(value, dict):
                total += len(value)
        return total
    
    def learn_from_url(self, url: str) -> str:
        """
        Learn knowledge from a URL
        
        Args:
            url: URL to scrape and learn from
            
        Returns:
            Result message
        """
        try:
            self.logger.info(f"Learning from URL: {url}")
            
            # Check if already processed
            url_hash = hashlib.md5(url.encode()).hexdigest()
            for source in self.knowledge_base['external_sources']:
                if source.get('url_hash') == url_hash:
                    return f"URL already processed: {url}"
            
            # Scrape content using enhanced scraper
            scrape_result = self.web_scraper.scrape_url(url)
            if not scrape_result or not scrape_result.get('success', False):
                return f"Failed to scrape content from: {url}"
            
            content = scrape_result.get('content', '')
            if not content:
                return f"No content extracted from: {url}"
            
            # Process content with enhanced data
            processed_info = self._process_enhanced_content(scrape_result, url)
            
            # Store source information
            source_info = {
                'url': url,
                'url_hash': url_hash,
                'scraped_at': time.time(),
                'title': processed_info.get('title', 'Unknown'),
                'content_length': len(content),
                'processed_items': processed_info.get('items_count', 0)
            }
            
            self.knowledge_base['external_sources'].append(source_info)
            
            # Save knowledge base
            self._save_knowledge_base()
            
            result = f"Successfully learned from {url}: {processed_info.get('items_count', 0)} items processed"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to learn from URL {url}: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def learn_from_file(self, file_path: str) -> str:
        """
        Learn knowledge from a file
        
        Args:
            file_path: Path to file to learn from
            
        Returns:
            Result message
        """
        try:
            self.logger.info(f"Learning from file: {file_path}")
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return f"File not found: {file_path}"
            
            # Check if already processed
            file_hash = self._get_file_hash(file_path_obj)
            for source in self.knowledge_base['file_sources']:
                if source.get('file_hash') == file_hash:
                    return f"File already processed: {file_path}"
            
            # Read file content
            content = self._read_file_content(file_path_obj)
            if not content:
                return f"Failed to read content from: {file_path}"
            
            # Process content
            processed_info = self._process_text_content(content, str(file_path))
            
            # Store source information
            source_info = {
                'file_path': str(file_path_obj),
                'file_hash': file_hash,
                'processed_at': time.time(),
                'file_size': file_path_obj.stat().st_size,
                'file_type': file_path_obj.suffix,
                'processed_items': processed_info.get('items_count', 0)
            }
            
            self.knowledge_base['file_sources'].append(source_info)
            
            # Save knowledge base
            self._save_knowledge_base()
            
            result = f"Successfully learned from {file_path_obj}: {processed_info.get('items_count', 0)} items processed"
            self.logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to learn from file {file_path}: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to hash file: {e}")
            return ""
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read content from file"""
        try:
            # Handle different file types
            if file_path.suffix.lower() in ['.txt', '.md', '.rst']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            elif file_path.suffix.lower() in ['.py', '.js', '.html', '.css']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Try to read as text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    return f"Binary file or unsupported format: {file_path.suffix}"
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def _process_text_content(self, content: str, source: str) -> Dict[str, Any]:
        """
        Process text content and extract knowledge
        
        Args:
            content: Text content to process
            source: Source of the content
            
        Returns:
            Processing results
        """
        try:
            processed_info = {
                'title': 'Unknown',
                'items_count': 0
            }
            
            # Simple text processing - in a real implementation, this would be more sophisticated
            lines = content.split('\n')
            processed_info['title'] = lines[0][:100] if lines else 'Unknown'
            
            # Look for game-related information
            game_keywords = ['game', 'chest', 'egg', 'breakable', 'macro', 'automation', 'bot']
            strategy_keywords = ['strategy', 'tip', 'guide', 'how to', 'best way']
            update_keywords = ['update', 'patch', 'new', 'change', 'fix']
            
            items_found = 0
            
            # Process line by line
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                if not line_lower or len(line_lower) < 10:
                    continue
                
                # Check for strategies
                if any(keyword in line_lower for keyword in strategy_keywords):
                    strategy = {
                        'content': line.strip(),
                        'source': source,
                        'line_number': i + 1,
                        'added_at': time.time()
                    }
                    self.knowledge_base['strategies'].append(strategy)
                    items_found += 1
                
                # Check for updates
                elif any(keyword in line_lower for keyword in update_keywords):
                    update = {
                        'content': line.strip(),
                        'source': source,
                        'line_number': i + 1,
                        'added_at': time.time()
                    }
                    self.knowledge_base['updates'].append(update)
                    items_found += 1
                
                # Check for general game information
                elif any(keyword in line_lower for keyword in game_keywords):
                    # Extract game info
                    info_key = f"info_{len(self.knowledge_base['game_info'])}"
                    self.knowledge_base['game_info'][info_key] = {
                        'content': line.strip(),
                        'source': source,
                        'line_number': i + 1,
                        'added_at': time.time()
                    }
                    items_found += 1
                
                # Look for tips (lines with certain patterns)
                elif any(pattern in line_lower for pattern in ['tip:', 'note:', '!', 'important']):
                    tip = {
                        'content': line.strip(),
                        'source': source,
                        'line_number': i + 1,
                        'added_at': time.time()
                    }
                    self.knowledge_base['tips'].append(tip)
                    items_found += 1
            
            processed_info['items_count'] = items_found
            return processed_info
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            return {'title': 'Error', 'items_count': 0}
    
    def get_mapped_items(self) -> Dict[str, Any]:
        """Get all mapped items from knowledge base"""
        return self.knowledge_base.get('game_info', {})
    
    def clear_mapped_items(self):
        """Clear all mapped items"""
        self.knowledge_base['game_info'] = {}
        self.knowledge_base['strategies'] = []
        self.knowledge_base['updates'] = []
        self.knowledge_base['tips'] = []
        self._save_knowledge_base()
        self.logger.info("All mapped items cleared")
    
    def _process_enhanced_content(self, scrape_result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Process enhanced scraper content with structured data
        
        Args:
            scrape_result: Enhanced scraper result with metadata
            source: Source of the content
            
        Returns:
            Processing results
        """
        try:
            processed_info = {
                'title': scrape_result.get('title', 'Unknown'),
                'items_count': 0
            }
            
            content = scrape_result.get('content', '')
            metadata = scrape_result.get('metadata', {})
            content_type = scrape_result.get('type', 'general')
            
            # Use enhanced data if available
            if metadata:
                processed_info['title'] = metadata.get('title', processed_info['title'])
                
                # Process gaming-specific metadata
                if content_type == 'roblox_data':
                    self._process_roblox_metadata(metadata, source)
                    processed_info['items_count'] += 1
                elif content_type == 'ps99_data':
                    self._process_ps99_metadata(metadata, source)
                    processed_info['items_count'] += 1
                elif content_type == 'api_data':
                    self._process_api_data(metadata, source)
                    processed_info['items_count'] += 1
            
            # Process text content as before but with enhanced keywords
            if content:
                enhanced_result = self._process_text_content(content, source)
                processed_info['items_count'] += enhanced_result.get('items_count', 0)
                
                # If we have no title from metadata, use the one from content processing
                if processed_info['title'] == 'Unknown':
                    processed_info['title'] = enhanced_result.get('title', 'Unknown')
            
            return processed_info
            
        except Exception as e:
            self.logger.error(f"Enhanced content processing failed: {e}")
            return {'title': 'Error', 'items_count': 0}
    
    def _process_roblox_metadata(self, metadata: Dict[str, Any], source: str):
        """Process Roblox-specific metadata"""
        try:
            if 'extracted_id' in metadata:
                game_info = {
                    'content': f"Roblox Game ID: {metadata['extracted_id']} ({metadata.get('id_type', 'unknown')})",
                    'source': source,
                    'metadata': metadata,
                    'added_at': time.time()
                }
                info_key = f"roblox_{metadata['extracted_id']}"
                self.knowledge_base['game_info'][info_key] = game_info
        except Exception as e:
            self.logger.error(f"Roblox metadata processing failed: {e}")
    
    def _process_ps99_metadata(self, metadata: Dict[str, Any], source: str):
        """Process Pet Simulator 99 specific metadata"""
        try:
            if 'eggs' in metadata or 'pets' in metadata:
                ps99_info = {
                    'content': f"PS99 Data: {json.dumps(metadata, indent=2)}",
                    'source': source,
                    'metadata': metadata,
                    'added_at': time.time()
                }
                info_key = f"ps99_{int(time.time())}"
                self.knowledge_base['game_info'][info_key] = ps99_info
        except Exception as e:
            self.logger.error(f"PS99 metadata processing failed: {e}")
    
    def _process_api_data(self, metadata: Dict[str, Any], source: str):
        """Process API response data"""
        try:
            if 'api_data' in metadata:
                api_info = {
                    'content': f"API Data from {source}: {json.dumps(metadata['api_data'], indent=2)}",
                    'source': source,
                    'metadata': metadata,
                    'added_at': time.time()
                }
                info_key = f"api_{int(time.time())}"
                self.knowledge_base['game_info'][info_key] = api_info
        except Exception as e:
            self.logger.error(f"API data processing failed: {e}")
    
    def get_relevant_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant knowledge items
        """
        try:
            query_lower = query.lower()
            relevant_items = []
            
            # Search strategies
            for strategy in self.knowledge_base['strategies']:
                if any(word in strategy['content'].lower() for word in query_lower.split()):
                    relevant_items.append({
                        'type': 'strategy',
                        'content': strategy['content'],
                        'source': strategy.get('source', 'unknown'),
                        'relevance_score': self._calculate_relevance(strategy['content'], query)
                    })
            
            # Search tips
            for tip in self.knowledge_base['tips']:
                if any(word in tip['content'].lower() for word in query_lower.split()):
                    relevant_items.append({
                        'type': 'tip',
                        'content': tip['content'],
                        'source': tip.get('source', 'unknown'),
                        'relevance_score': self._calculate_relevance(tip['content'], query)
                    })
            
            # Search game info
            for key, info in self.knowledge_base['game_info'].items():
                if any(word in info['content'].lower() for word in query_lower.split()):
                    relevant_items.append({
                        'type': 'game_info',
                        'content': info['content'],
                        'source': info.get('source', 'unknown'),
                        'relevance_score': self._calculate_relevance(info['content'], query)
                    })
            
            # Sort by relevance and limit results
            relevant_items.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_items[:limit]
            
        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            return []
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        try:
            content_lower = content.lower()
            query_words = query.lower().split()
            
            score = 0.0
            total_words = len(query_words)
            
            for word in query_words:
                if word in content_lower:
                    # Exact match
                    score += 1.0
                else:
                    # Partial match (simple substring)
                    for content_word in content_lower.split():
                        if word in content_word or content_word in word:
                            score += 0.5
                            break
            
            return score / total_words if total_words > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Relevance calculation failed: {e}")
            return 0.0
    
    def update_from_developer_sources(self, urls: Optional[List[str]] = None) -> str:
        """
        Update knowledge from developer blogs and official sources
        
        Args:
            urls: List of URLs to check (uses default if None)
            
        Returns:
            Update result message
        """
        try:
            # Default developer sources (these would be game-specific)
            if urls is None:
                urls = [
                    # These would be actual game developer blogs/news sites
                    "https://example-game-dev-blog.com/news",
                    "https://example-game.com/updates",
                    "https://example-game-wiki.com/latest-changes"
                ]
            
            total_updates = 0
            results = []
            
            for url in urls:
                try:
                    result = self.learn_from_url(url)
                    results.append(f"{url}: {result}")
                    if "successfully learned" in result.lower():
                        total_updates += 1
                except Exception as e:
                    results.append(f"{url}: Error - {e}")
            
            summary = f"Updated knowledge from {total_updates}/{len(urls)} sources"
            self.logger.info(summary)
            
            return f"{summary}\n" + "\n".join(results)
            
        except Exception as e:
            error_msg = f"Failed to update from developer sources: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base"""
        try:
            summary = {
                'total_items': self.knowledge_base['metadata']['total_items'],
                'last_updated': self.knowledge_base['metadata']['last_updated'],
                'categories': {
                    'strategies': len(self.knowledge_base['strategies']),
                    'tips': len(self.knowledge_base['tips']),
                    'game_info': len(self.knowledge_base['game_info']),
                    'updates': len(self.knowledge_base['updates'])
                },
                'sources': {
                    'external_urls': len(self.knowledge_base['external_sources']),
                    'files': len(self.knowledge_base['file_sources'])
                },
                'recent_items': []
            }
            
            # Get recent items (last 5)
            all_items = []
            
            for strategy in self.knowledge_base['strategies'][-5:]:
                all_items.append({
                    'type': 'strategy',
                    'content': strategy['content'][:100] + '...',
                    'added_at': strategy.get('added_at', 0)
                })
            
            for tip in self.knowledge_base['tips'][-5:]:
                all_items.append({
                    'type': 'tip',
                    'content': tip['content'][:100] + '...',
                    'added_at': tip.get('added_at', 0)
                })
            
            # Sort by added_at and take recent ones
            all_items.sort(key=lambda x: x['added_at'], reverse=True)
            summary['recent_items'] = all_items[:5]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate knowledge summary: {e}")
            return {'error': str(e)}
    
    def get_knowledge_count(self) -> int:
        """Get total count of knowledge items"""
        return self.knowledge_base['metadata']['total_items']
    
    def search_knowledge(self, query: str) -> str:
        """
        Search knowledge base and return formatted results
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        relevant_items = self.get_relevant_knowledge(query, limit=5)
        
        if not relevant_items:
            return f"No knowledge found for query: {query}"
        
        result = f"Found {len(relevant_items)} relevant items for '{query}':\n\n"
        
        for i, item in enumerate(relevant_items, 1):
            result += f"{i}. [{item['type']}] {item['content'][:200]}{'...' if len(item['content']) > 200 else ''}\n"
            result += f"   Source: {item['source']}\n"
            result += f"   Relevance: {item['relevance_score']:.2f}\n\n"
        
        return result
    
    def clear_knowledge_base(self):
        """Clear all knowledge base data (use with caution)"""
        self.knowledge_base = {
            "game_info": {},
            "strategies": [],
            "updates": [],
            "tips": [],
            "external_sources": [],
            "file_sources": [],
            "learned_patterns": {},
            "metadata": {
                "last_updated": None,
                "version": "1.0",
                "total_items": 0
            }
        }
        
        self._save_knowledge_base()
        self.logger.info("Knowledge base cleared")
