"""
Repository Analyzer for AI Game Bot
Clones and analyzes repositories to extract game automation techniques
"""

import os
import subprocess
import logging
import psycopg2
from psycopg2.extensions import connection as Connection
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import tempfile
import shutil

class RepositoryAnalyzer:
    """Analyzes repositories for game automation techniques and patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_connection: Optional[Connection] = None
        self.temp_dir = Path("temp_repos")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Connect to database
        self._connect_to_database()
        
        # Patterns to look for in code
        self.automation_patterns = {
            'screen_capture': ['screenshot', 'screen', 'capture', 'pyautogui', 'mss', 'pillow'],
            'computer_vision': ['cv2', 'opencv', 'template', 'match', 'contour', 'detection'],
            'input_automation': ['click', 'key', 'mouse', 'keyboard', 'input', 'pyautogui'],
            'ai_learning': ['tensorflow', 'pytorch', 'keras', 'sklearn', 'reinforcement', 'neural'],
            'game_detection': ['window', 'process', 'hwnd', 'findwindow', 'game'],
            'macro_system': ['macro', 'record', 'playback', 'sequence', 'action'],
            'pattern_recognition': ['template', 'feature', 'match', 'recognize', 'classify']
        }
        
        self.logger.info("Repository analyzer initialized")
    
    def _connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_connection = psycopg2.connect(
                host=os.getenv('PGHOST'),
                port=os.getenv('PGPORT'),
                database=os.getenv('PGDATABASE'),
                user=os.getenv('PGUSER'),
                password=os.getenv('PGPASSWORD')
            )
            self.logger.info("Connected to database")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def analyze_repositories(self, repo_urls: List[str]) -> Dict[str, Any]:
        """Analyze multiple repositories"""
        results = {
            'total_repos': len(repo_urls),
            'successful_clones': 0,
            'failed_clones': 0,
            'total_patterns_found': 0,
            'processing_errors': []
        }
        
        for i, url in enumerate(repo_urls, 1):
            self.logger.info(f"Processing repository {i}/{len(repo_urls)}: {url}")
            
            try:
                # Insert repository record
                repo_id = self._insert_repository_record(url)
                
                # Clone repository
                repo_path = self._clone_repository(url, repo_id)
                
                if repo_path:
                    # Analyze repository
                    analysis_results = self._analyze_repository(repo_path, repo_id)
                    results['total_patterns_found'] += analysis_results.get('patterns_found', 0)
                    results['successful_clones'] += 1
                    
                    # Clean up cloned repo to save space
                    shutil.rmtree(repo_path, ignore_errors=True)
                else:
                    results['failed_clones'] += 1
                    
            except Exception as e:
                error_msg = f"Error processing {url}: {e}"
                self.logger.error(error_msg)
                results['processing_errors'].append(error_msg)
                results['failed_clones'] += 1
        
        return results
    
    def _insert_repository_record(self, url: str) -> int:
        """Insert repository record and return ID"""
        if not self.db_connection:
            raise Exception("Database connection not available")
        
        cursor = self.db_connection.cursor()
        
        # Extract repository name from URL
        repo_name = self._extract_repo_name(url)
        
        cursor.execute("""
            INSERT INTO repositories (name, url, clone_status) 
            VALUES (%s, %s, 'cloning') 
            RETURNING id
        """, (repo_name, url))
        
        result = cursor.fetchone()
        if not result:
            raise Exception("Failed to insert repository record")
        repo_id = result[0]
        self.db_connection.commit()
        cursor.close()
        
        return repo_id
    
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from URL"""
        if 'github.com' in url:
            parts = url.rstrip('/').split('/')
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
        elif 'replit.com' in url:
            parts = url.rstrip('/').split('/')
            if len(parts) >= 2:
                return f"replit/{parts[-1]}"
        
        return urlparse(url).path.strip('/')
    
    def _clone_repository(self, url: str, repo_id: int) -> Optional[Path]:
        """Clone repository to temporary location"""
        try:
            # Create unique directory for this repo
            repo_name = self._extract_repo_name(url).replace('/', '_')
            repo_path = self.temp_dir / f"{repo_name}_{repo_id}"
            
            # Skip SSH URLs that require authentication
            if url.startswith('git@'):
                self.logger.warning(f"Skipping SSH URL (requires auth): {url}")
                self._update_repository_status(repo_id, 'failed', 'SSH URL requires authentication')
                return None
            
            # Clone repository
            result = subprocess.run([
                'git', 'clone', '--depth', '1', url, str(repo_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully cloned {url}")
                self._update_repository_status(repo_id, 'cloned')
                return repo_path
            else:
                error_msg = f"Git clone failed: {result.stderr}"
                self.logger.error(error_msg)
                self._update_repository_status(repo_id, 'failed', error_msg)
                return None
                
        except subprocess.TimeoutExpired:
            error_msg = "Clone timeout"
            self.logger.error(f"Clone timeout for {url}")
            self._update_repository_status(repo_id, 'failed', error_msg)
            return None
        except Exception as e:
            error_msg = f"Clone error: {e}"
            self.logger.error(error_msg)
            self._update_repository_status(repo_id, 'failed', error_msg)
            return None
    
    def _update_repository_status(self, repo_id: int, status: str, description: Optional[str] = None):
        """Update repository status in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        
        if description:
            cursor.execute("""
                UPDATE repositories 
                SET clone_status = %s, description = %s, clone_date = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (status, description, repo_id))
        else:
            cursor.execute("""
                UPDATE repositories 
                SET clone_status = %s, clone_date = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (status, repo_id))
        
        self.db_connection.commit()
        cursor.close()
    
    def _analyze_repository(self, repo_path: Path, repo_id: int) -> Dict[str, Any]:
        """Analyze repository contents for automation patterns"""
        analysis_results = {
            'patterns_found': 0,
            'files_analyzed': 0,
            'total_lines': 0
        }
        
        try:
            # Get repository statistics
            file_count, total_lines, primary_language = self._get_repo_stats(repo_path)
            
            # Update repository info
            self._update_repo_info(repo_id, file_count, total_lines, primary_language)
            
            # Analyze Python files for automation patterns
            python_files = list(repo_path.rglob("*.py"))
            analysis_results['files_analyzed'] = len(python_files)
            
            for py_file in python_files[:50]:  # Limit to prevent memory issues
                patterns = self._analyze_python_file(py_file, repo_id)
                analysis_results['patterns_found'] += len(patterns)
            
            # Look for specific automation frameworks
            self._detect_automation_frameworks(repo_path, repo_id)
            
            # Extract configuration patterns
            self._extract_config_patterns(repo_path, repo_id)
            
            # Find learning resources
            self._extract_learning_resources(repo_path, repo_id)
            
            self._update_repository_status(repo_id, 'analyzed')
            
        except Exception as e:
            self.logger.error(f"Analysis error for repo {repo_id}: {e}")
            self._update_repository_status(repo_id, 'analysis_failed', str(e))
        
        return analysis_results
    
    def _get_repo_stats(self, repo_path: Path) -> tuple:
        """Get basic repository statistics"""
        file_count = 0
        total_lines = 0
        language_counts = {}
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_count += 1
                
                # Count lines for text files
                try:
                    if file_path.suffix in ['.py', '.js', '.java', '.cpp', '.c', '.h']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.readlines())
                            total_lines += lines
                        
                        # Track language distribution
                        ext = file_path.suffix
                        language_counts[ext] = language_counts.get(ext, 0) + 1
                except Exception:
                    continue
        
        # Determine primary language
        primary_language = 'unknown'
        if language_counts:
            primary_language = max(language_counts.keys(), key=lambda x: language_counts[x])
        
        return file_count, total_lines, primary_language
    
    def _update_repo_info(self, repo_id: int, file_count: int, total_lines: int, primary_language: str):
        """Update repository information in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        cursor.execute("""
            UPDATE repositories 
            SET file_count = %s, total_lines = %s, primary_language = %s 
            WHERE id = %s
        """, (file_count, total_lines, primary_language, repo_id))
        self.db_connection.commit()
        cursor.close()
    
    def _analyze_python_file(self, file_path: Path, repo_id: int) -> List[Dict]:
        """Analyze Python file for automation patterns"""
        patterns_found = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for automation patterns
            for pattern_type, keywords in self.automation_patterns.items():
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        pattern_info = {
                            'type': pattern_type,
                            'keyword': keyword,
                            'file_path': str(file_path),
                            'content_snippet': self._extract_code_snippet(content, keyword)
                        }
                        patterns_found.append(pattern_info)
                        self._store_code_pattern(repo_id, pattern_info)
                        break  # Only store one pattern per type per file
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {file_path}: {e}")
        
        return patterns_found
    
    def _extract_code_snippet(self, content: str, keyword: str) -> str:
        """Extract code snippet around a keyword"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Extract 3 lines before and after
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                snippet = '\n'.join(lines[start:end])
                return snippet[:500]  # Limit snippet size
        return ""
    
    def _store_code_pattern(self, repo_id: int, pattern_info: Dict):
        """Store code pattern in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO code_patterns 
            (repository_id, pattern_type, pattern_name, description, code_snippet, file_path, relevance_score) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            repo_id,
            pattern_info['type'],
            pattern_info['keyword'],
            f"Found {pattern_info['keyword']} pattern in {pattern_info['type']}",
            pattern_info['content_snippet'],
            pattern_info['file_path'],
            0.7  # Default relevance score
        ))
        self.db_connection.commit()
        cursor.close()
    
    def _detect_automation_frameworks(self, repo_path: Path, repo_id: int):
        """Detect specific automation frameworks and techniques"""
        frameworks = {
            'SerpentAI': ['serpent', 'serpentai'],
            'PyAutoGUI': ['pyautogui'],
            'OpenCV': ['cv2', 'opencv'],
            'TensorFlow': ['tensorflow', 'tf.'],
            'PyTorch': ['torch', 'pytorch'],
            'Selenium': ['selenium', 'webdriver']
        }
        
        # Check requirements files
        req_files = list(repo_path.glob("*requirements*.txt")) + list(repo_path.glob("setup.py"))
        
        for req_file in req_files:
            try:
                with open(req_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    
                for framework, keywords in frameworks.items():
                    if any(keyword in content for keyword in keywords):
                        self._store_automation_technique(repo_id, framework, content)
            except Exception:
                continue
    
    def _store_automation_technique(self, repo_id: int, framework: str, context: str):
        """Store automation technique in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO automation_techniques 
            (repository_id, technique_name, category, description, implementation_approach, applicability_rating) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            repo_id,
            framework,
            'framework',
            f"{framework} automation framework detected",
            context[:1000],
            8  # High applicability for known frameworks
        ))
        self.db_connection.commit()
        cursor.close()
    
    def _extract_config_patterns(self, repo_path: Path, repo_id: int):
        """Extract configuration patterns from repository"""
        config_files = (
            list(repo_path.glob("*.json")) + 
            list(repo_path.glob("*.yaml")) + 
            list(repo_path.glob("*.yml")) + 
            list(repo_path.glob("config/*"))
        )
        
        for config_file in config_files[:10]:  # Limit number of config files
            try:
                if config_file.suffix in ['.json', '.yaml', '.yml']:
                    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self._store_config_pattern(repo_id, config_file.name, content)
            except Exception:
                continue
    
    def _store_config_pattern(self, repo_id: int, config_name: str, content: str):
        """Store configuration pattern in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        
        # Try to parse as JSON for better storage
        try:
            config_data = json.loads(content)
        except:
            config_data = {"raw_content": content[:1000]}
        
        cursor.execute("""
            INSERT INTO config_patterns 
            (repository_id, config_type, config_name, config_data, description) 
            VALUES (%s, %s, %s, %s, %s)
        """, (
            repo_id,
            'configuration',
            config_name,
            json.dumps(config_data),
            f"Configuration from {config_name}"
        ))
        self.db_connection.commit()
        cursor.close()
    
    def _extract_learning_resources(self, repo_path: Path, repo_id: int):
        """Extract learning resources like README, docs, tutorials"""
        doc_files = (
            list(repo_path.glob("README*")) + 
            list(repo_path.glob("*.md")) + 
            list(repo_path.glob("docs/*")) +
            list(repo_path.glob("tutorial*"))
        )
        
        for doc_file in doc_files[:5]:  # Limit number of doc files
            try:
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 100:  # Only store meaningful content
                        self._store_learning_resource(repo_id, doc_file.name, content)
            except Exception:
                continue
    
    def _store_learning_resource(self, repo_id: int, title: str, content: str):
        """Store learning resource in database"""
        if not self.db_connection:
            return
        
        cursor = self.db_connection.cursor()
        
        # Extract key insights
        key_insights = self._extract_key_insights(content)
        
        cursor.execute("""
            INSERT INTO learning_sources 
            (repository_id, source_type, title, content, key_insights, relevance_to_project) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            repo_id,
            'documentation',
            title,
            content[:5000],  # Limit content size
            key_insights,
            0.5  # Default relevance
        ))
        self.db_connection.commit()
        cursor.close()
    
    def _extract_key_insights(self, content: str) -> str:
        """Extract key insights from documentation"""
        # Simple extraction of important lines
        lines = content.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['usage:', 'example:', 'how to', 'install', 'setup']):
                insights.append(line)
                if len(insights) >= 5:
                    break
        
        return '\n'.join(insights)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of repository analysis"""
        if not self.db_connection:
            return {'error': 'Database connection not available'}
        
        cursor = self.db_connection.cursor()
        
        # Get repository counts by status
        cursor.execute("""
            SELECT clone_status, COUNT(*) 
            FROM repositories 
            GROUP BY clone_status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Get pattern counts by type
        cursor.execute("""
            SELECT pattern_type, COUNT(*) 
            FROM code_patterns 
            GROUP BY pattern_type
        """)
        pattern_counts = dict(cursor.fetchall())
        
        # Get technique counts
        cursor.execute("""
            SELECT COUNT(*) FROM automation_techniques
        """)
        result = cursor.fetchone()
        technique_count = result[0] if result else 0
        
        cursor.close()
        
        return {
            'repository_status': status_counts,
            'pattern_distribution': pattern_counts,
            'total_techniques': technique_count,
            'database_status': 'connected'
        }
    
    def __del__(self):
        """Clean up database connection"""
        if self.db_connection:
            self.db_connection.close()