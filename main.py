#!/usr/bin/env python3
"""
TMP (Test My Python) - Auto Evaluation and Marking System
A comprehensive system for evaluating Python student projects from GitHub repositories.

Usage:
    python tmp.py --config config.yaml --submissions submissions.csv --output results.csv

Author: TMP Team
Version: 1.0.0
"""

import os
import sys
import csv
import json
import yaml
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from urllib.parse import urlparse
import tempfile
import zipfile
import io

# Third-party imports (install with: pip install openai anthropic gitpython)
try:
    import git
    from git import Repo
except ImportError:
    print("Please install GitPython: pip install GitPython")
    sys.exit(1)

try:
    import openai
except ImportError:
    openai = None
    print("Warning: OpenAI not installed. Install with: pip install openai")

try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: Anthropic not installed. Install with: pip install anthropic")


# Configuration and Data Classes
@dataclass
class ProjectSubmission:
    """Data class for student project submission"""
    student_name: str
    student_id: str
    email: str
    project_category: str
    github_url: str
    submission_date: str
    additional_notes: str = ""

    def __post_init__(self):
        """Validate and normalize data after initialization"""
        self.project_category = self.project_category.lower().strip()
        if self.project_category not in ['data_analysis', 'crud', 'script_tool']:
            # Try to map common variations
            category_mapping = {
                'data analysis': 'data_analysis',
                'data-analysis': 'data_analysis',
                'eda': 'data_analysis',
                'crud project': 'crud',
                'api': 'crud',
                'web scraper': 'script_tool',
                'scraper': 'script_tool',
                'script': 'script_tool',
                'tool': 'script_tool'
            }
            self.project_category = category_mapping.get(self.project_category, 'unknown')


@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    student_id: str
    student_name: str
    project_category: str
    github_url: str
    
    # Technical evaluation
    code_quality_score: float = 0.0
    functionality_score: float = 0.0
    requirements_fulfillment: float = 0.0
    code_structure_score: float = 0.0
    
    # LLM-based evaluation
    llm_generated_likelihood: float = 0.0
    assessment_goals_fulfillment: float = 0.0
    creativity_score: float = 0.0
    documentation_score: float = 0.0
    
    # Overall scores
    total_score: float = 0.0
    grade: str = "F"
    
    # Detailed feedback
    detailed_feedback: str = ""
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        
        # Calculate total score and grade
        self.calculate_total_score()
    
    def calculate_total_score(self):
        """Calculate total score and assign grade"""
        weights = {
            'code_quality': 0.25,
            'functionality': 0.25,
            'requirements': 0.20,
            'structure': 0.15,
            'goals_fulfillment': 0.15
        }
        
        self.total_score = (
            self.code_quality_score * weights['code_quality'] +
            self.functionality_score * weights['functionality'] +
            self.requirements_fulfillment * weights['requirements'] +
            self.code_structure_score * weights['structure'] +
            self.assessment_goals_fulfillment * weights['goals_fulfillment']
        )
        
        # Penalize for likely LLM generation
        if self.llm_generated_likelihood > 0.8:
            self.total_score *= 0.3  # Heavy penalty
            self.warnings.append("High likelihood of LLM-generated code detected")
        elif self.llm_generated_likelihood > 0.6:
            self.total_score *= 0.7  # Moderate penalty
            self.warnings.append("Possible LLM-generated code detected")
        
        # Assign grade
        if self.total_score >= 90:
            self.grade = "A"
        elif self.total_score >= 80:
            self.grade = "B"
        elif self.total_score >= 70:
            self.grade = "C"
        elif self.total_score >= 60:
            self.grade = "D"
        else:
            self.grade = "F"


class TMPEvaluator:
    """Main evaluation system class"""
    
    def __init__(self, config_path: str):
        """Initialize the evaluator with configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_llm_clients()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tmp_eval_"))
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'evaluation_criteria': {
                'data_analysis': {
                    'required_files': ['*.py', '*.ipynb'],
                    'required_libraries': ['pandas', 'numpy', 'matplotlib', 'seaborn'],
                    'evaluation_points': [
                        'Data loading and cleaning',
                        'Exploratory data analysis',
                        'Statistical analysis',
                        'Data visualization',
                        'Insights and conclusions'
                    ]
                },
                'crud': {
                    'required_files': ['*.py'],
                    'required_libraries': ['fastapi', 'flask', 'sqlalchemy'],
                    'evaluation_points': [
                        'API endpoints implementation',
                        'Database operations (CRUD)',
                        'Error handling',
                        'API documentation',
                        'Testing'
                    ]
                },
                'script_tool': {
                    'required_files': ['*.py'],
                    'required_libraries': ['requests', 'beautifulsoup4', 'scrapy'],
                    'evaluation_points': [
                        'Web scraping implementation',
                        'Data extraction accuracy',
                        'Error handling and robustness',
                        'Output format (CSV/JSON)',
                        'Code organization'
                    ]
                }
            },
            'llm': {
                'provider': 'openai',  # or 'anthropic'
                'model': 'gpt-4',
                'api_key_env': 'OPENAI_API_KEY',  # or 'ANTHROPIC_API_KEY'
                'max_tokens': 2000,
                'temperature': 0.1
            },
            'scoring': {
                'llm_penalty_threshold': 0.6,
                'max_repo_size_mb': 100,
                'timeout_seconds': 300
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tmp_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_llm_clients(self):
        """Setup LLM clients based on configuration"""
        self.llm_client = None
        
        if self.config['llm']['provider'] == 'openai' and openai:
            api_key = os.getenv(self.config['llm']['api_key_env'])
            if api_key:
                openai.api_key = api_key
                self.llm_client = 'openai'
        elif self.config['llm']['provider'] == 'anthropic' and anthropic:
            api_key = os.getenv(self.config['llm']['api_key_env'])
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.llm_client = 'anthropic'
        
        if not self.llm_client:
            self.logger.warning("No LLM client configured. LLM evaluation will be skipped.")
    
    def load_submissions(self, csv_path: str) -> List[ProjectSubmission]:
        """Load student submissions from CSV file"""
        submissions = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map CSV columns to ProjectSubmission fields
                    submission = ProjectSubmission(
                        student_name=row.get('student_name', ''),
                        student_id=row.get('student_id', ''),
                        email=row.get('email', ''),
                        project_category=row.get('project_category', ''),
                        github_url=row.get('github_url', ''),
                        submission_date=row.get('submission_date', ''),
                        additional_notes=row.get('additional_notes', '')
                    )
                    submissions.append(submission)
        except Exception as e:
            self.logger.error(f"Error loading submissions: {e}")
            return []
        
        self.logger.info(f"Loaded {len(submissions)} submissions")
        return submissions
    
    def download_repository(self, github_url: str, student_id: str) -> Optional[Path]:
        """Download GitHub repository"""
        try:
            # Parse GitHub URL
            parsed_url = urlparse(github_url)
            if 'github.com' not in parsed_url.netloc:
                self.logger.error(f"Invalid GitHub URL: {github_url}")
                return None
            
            # Create directory for this student
            student_dir = self.temp_dir / f"student_{student_id}"
            student_dir.mkdir(exist_ok=True)
            
            # Try to clone the repository
            try:
                repo = Repo.clone_from(github_url, student_dir / "repo")
                self.logger.info(f"Successfully cloned repository for student {student_id}")
                return student_dir / "repo"
            except Exception as clone_error:
                self.logger.warning(f"Git clone failed for {student_id}, trying ZIP download: {clone_error}")
                
                # Try downloading as ZIP if clone fails
                zip_url = github_url.replace('github.com', 'github.com').rstrip('/') + '/archive/main.zip'
                response = requests.get(zip_url, timeout=30)
                
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        z.extractall(student_dir)
                    
                    # Find the extracted directory
                    extracted_dirs = [d for d in student_dir.iterdir() if d.is_dir()]
                    if extracted_dirs:
                        return extracted_dirs[0]
                
                # Try master branch if main doesn't exist
                zip_url = github_url.replace('github.com', 'github.com').rstrip('/') + '/archive/master.zip'
                response = requests.get(zip_url, timeout=30)
                
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        z.extractall(student_dir)
                    
                    extracted_dirs = [d for d in student_dir.iterdir() if d.is_dir()]
                    if extracted_dirs:
                        return extracted_dirs[0]
                
                self.logger.error(f"Failed to download repository for student {student_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading repository for student {student_id}: {e}")
            return None
    
    def analyze_code_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze the code structure and quality"""
        analysis = {
            'total_files': 0,
            'python_files': 0,
            'notebook_files': 0,
            'total_lines': 0,
            'avg_function_length': 0,
            'has_readme': False,
            'has_requirements': False,
            'libraries_used': set(),
            'file_types': {},
            'large_files': []
        }
        
        try:
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    analysis['total_files'] += 1
                    
                    # Check file types
                    suffix = file_path.suffix.lower()
                    analysis['file_types'][suffix] = analysis['file_types'].get(suffix, 0) + 1
                    
                    # Check specific file types
                    if suffix == '.py':
                        analysis['python_files'] += 1
                        self.analyze_python_file(file_path, analysis)
                    elif suffix == '.ipynb':
                        analysis['notebook_files'] += 1
                    elif file_path.name.lower() in ['readme.md', 'readme.txt', 'readme']:
                        analysis['has_readme'] = True
                    elif file_path.name.lower() in ['requirements.txt', 'requirements.pip']:
                        analysis['has_requirements'] = True
                        self.parse_requirements(file_path, analysis)
                    
                    # Check for large files
                    if file_path.stat().st_size > 1024 * 1024:  # 1MB
                        analysis['large_files'].append(str(file_path.relative_to(repo_path)))
            
        except Exception as e:
            self.logger.error(f"Error analyzing code structure: {e}")
        
        analysis['libraries_used'] = list(analysis['libraries_used'])
        return analysis
    
    def analyze_python_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze individual Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                analysis['total_lines'] += len(lines)
                
                # Look for import statements
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract library names
                        if line.startswith('import '):
                            lib = line.split()[1].split('.')[0]
                        else:  # from ... import ...
                            lib = line.split()[1].split('.')[0]
                        analysis['libraries_used'].add(lib)
        
        except Exception as e:
            self.logger.debug(f"Error analyzing Python file {file_path}: {e}")
    
    def parse_requirements(self, file_path: Path, analysis: Dict[str, Any]):
        """Parse requirements.txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, etc.)
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                        analysis['libraries_used'].add(package)
        except Exception as e:
            self.logger.debug(f"Error parsing requirements file {file_path}: {e}")
    
    def evaluate_with_llm(self, repo_path: Path, submission: ProjectSubmission, 
                         code_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate project using LLM"""
        if not self.llm_client:
            return {
                'llm_generated_likelihood': 0.0,
                'assessment_goals_fulfillment': 0.0,
                'creativity_score': 0.0,
                'documentation_score': 0.0,
                'detailed_feedback': "LLM evaluation not available"
            }
        
        # Collect code samples
        code_samples = self.collect_code_samples(repo_path)
        
        # Prepare prompt
        prompt = self.create_evaluation_prompt(submission, code_analysis, code_samples)
        
        try:
            if self.llm_client == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.config['llm']['model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config['llm']['max_tokens'],
                    temperature=self.config['llm']['temperature']
                )
                result_text = response.choices[0].message.content
            
            elif self.llm_client == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=self.config['llm']['model'],
                    max_tokens=self.config['llm']['max_tokens'],
                    temperature=self.config['llm']['temperature'],
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            
            # Parse LLM response
            return self.parse_llm_response(result_text)
        
        except Exception as e:
            self.logger.error(f"Error in LLM evaluation: {e}")
            return {
                'llm_generated_likelihood': 0.0,
                'assessment_goals_fulfillment': 0.0,
                'creativity_score': 0.0,
                'documentation_score': 0.0,
                'detailed_feedback': f"LLM evaluation failed: {str(e)}"
            }
    
    def collect_code_samples(self, repo_path: Path, max_samples: int = 5) -> List[str]:
        """Collect representative code samples from the repository"""
        code_samples = []
        
        # Prioritize main files
        priority_files = ['main.py', 'app.py', 'run.py', '__init__.py']
        
        python_files = list(repo_path.rglob('*.py'))
        
        # Sort by priority and size
        def file_priority(file_path):
            name = file_path.name
            if name in priority_files:
                return 0
            elif name.startswith('test_'):
                return 2
            else:
                return 1
        
        python_files = sorted(python_files, key=lambda f: (file_priority(f), -f.stat().st_size))
        
        for file_path in python_files[:max_samples]:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 100:  # Skip very small files
                        code_samples.append(f"## {file_path.name}\n```python\n{content[:2000]}\n```")
            except Exception as e:
                self.logger.debug(f"Error reading {file_path}: {e}")
        
        return code_samples
    
    def create_evaluation_prompt(self, submission: ProjectSubmission, 
                               code_analysis: Dict[str, Any], 
                               code_samples: List[str]) -> str:
        """Create evaluation prompt for LLM"""
        
        criteria = self.config['evaluation_criteria'].get(submission.project_category, {})
        
        prompt = f"""
You are an expert Python instructor evaluating a student project. Please analyze the following project and provide scores (0-100) for each criterion.

**Project Information:**
- Student: {submission.student_name}
- Category: {submission.project_category}
- GitHub URL: {submission.github_url}

**Code Analysis:**
- Total Python files: {code_analysis['python_files']}
- Total lines of code: {code_analysis['total_lines']}
- Libraries used: {', '.join(code_analysis['libraries_used'])}
- Has README: {code_analysis['has_readme']}
- Has requirements.txt: {code_analysis['has_requirements']}

**Evaluation Criteria for {submission.project_category}:**
{chr(10).join([f"- {point}" for point in criteria.get('evaluation_points', [])])}

**Code Samples:**
{chr(10).join(code_samples)}

**Please evaluate and provide scores (0-100) for:**
1. LLM Generated Likelihood (0=definitely human-written, 100=definitely AI-generated)
2. Assessment Goals Fulfillment (how well does it meet the project requirements)
3. Creativity Score (originality and creative problem-solving)
4. Documentation Score (code comments, README, documentation quality)

**Also provide detailed feedback (2-3 sentences) explaining your evaluation.**

**Response Format (JSON):**
{{
    "llm_generated_likelihood": 0-100,
    "assessment_goals_fulfillment": 0-100,
    "creativity_score": 0-100,
    "documentation_score": 0-100,
    "detailed_feedback": "Your detailed feedback here"
}}
"""
        return prompt
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract scores"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Convert scores to 0-1 range
                return {
                    'llm_generated_likelihood': result.get('llm_generated_likelihood', 0) / 100,
                    'assessment_goals_fulfillment': result.get('assessment_goals_fulfillment', 0) / 100,
                    'creativity_score': result.get('creativity_score', 0) / 100,
                    'documentation_score': result.get('documentation_score', 0) / 100,
                    'detailed_feedback': result.get('detailed_feedback', 'No feedback provided')
                }
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
        
        # Fallback: try to extract scores with regex
        scores = {}
        patterns = {
            'llm_generated_likelihood': r'llm.*likelihood.*?(\d+)',
            'assessment_goals_fulfillment': r'assessment.*goals.*?(\d+)',
            'creativity_score': r'creativity.*?(\d+)',
            'documentation_score': r'documentation.*?(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            scores[key] = (int(match.group(1)) / 100) if match else 0.0
        
        scores['detailed_feedback'] = response_text[:500]  # First 500 chars as feedback
        return scores
    
    def calculate_technical_scores(self, code_analysis: Dict[str, Any], 
                                 submission: ProjectSubmission) -> Dict[str, float]:
        """Calculate technical scores based on code analysis"""
        scores = {
            'code_quality_score': 0.0,
            'functionality_score': 0.0,
            'requirements_fulfillment': 0.0,
            'code_structure_score': 0.0
        }
        
        # Code quality score
        quality_factors = 0
        if code_analysis['has_readme']:
            quality_factors += 20
        if code_analysis['has_requirements']:
            quality_factors += 20
        if code_analysis['python_files'] > 0:
            quality_factors += 30
        if code_analysis['total_lines'] > 50:  # Minimum code amount
            quality_factors += 30
        
        scores['code_quality_score'] = quality_factors
        
        # Requirements fulfillment
        criteria = self.config['evaluation_criteria'].get(submission.project_category, {})
        required_libs = set(criteria.get('required_libraries', []))
        used_libs = set(code_analysis['libraries_used'])
        
        if required_libs:
            fulfillment = len(required_libs.intersection(used_libs)) / len(required_libs) * 100
        else:
            fulfillment = 50  # Default if no specific requirements
        
        scores['requirements_fulfillment'] = fulfillment
        
        # Structure score
        structure_score = 0
        if code_analysis['python_files'] > 1:  # Multiple files indicate good structure
            structure_score += 30
        if any(f.endswith('.py') for f in code_analysis['file_types'].keys()):
            structure_score += 40
        if not code_analysis['large_files']:  # No overly large files
            structure_score += 30
        
        scores['code_structure_score'] = structure_score
        
        # Functionality score (basic heuristic)
        functionality = 50  # Base score
        if code_analysis['total_lines'] > 100:
            functionality += 20
        if len(code_analysis['libraries_used']) > 3:
            functionality += 20
        if code_analysis['notebook_files'] > 0 and submission.project_category == 'data_analysis':
            functionality += 10
        
        scores['functionality_score'] = min(100, functionality)
        
        return scores
    
    def evaluate_submission(self, submission: ProjectSubmission) -> EvaluationResult:
        """Evaluate a single submission"""
        self.logger.info(f"Evaluating submission for {submission.student_name} ({submission.student_id})")
        
        # Initialize result
        result = EvaluationResult(
            student_id=submission.student_id,
            student_name=submission.student_name,
            project_category=submission.project_category,
            github_url=submission.github_url
        )
        
        try:
            # Download repository
            repo_path = self.download_repository(submission.github_url, submission.student_id)
            if not repo_path:
                result.errors.append("Failed to download repository")
                return result
            
            # Analyze code structure
            code_analysis = self.analyze_code_structure(repo_path)
            
            # Calculate technical scores
            technical_scores = self.calculate_technical_scores(code_analysis, submission)
            result.code_quality_score = technical_scores['code_quality_score']
            result.functionality_score = technical_scores['functionality_score']
            result.requirements_fulfillment = technical_scores['requirements_fulfillment']
            result.code_structure_score = technical_scores['code_structure_score']
            
            # LLM evaluation
            llm_scores = self.evaluate_with_llm(repo_path, submission, code_analysis)
            result.llm_generated_likelihood = llm_scores['llm_generated_likelihood']
            result.assessment_goals_fulfillment = llm_scores['assessment_goals_fulfillment'] * 100
            result.creativity_score = llm_scores['creativity_score'] * 100
            result.documentation_score = llm_scores['documentation_score'] * 100
            result.detailed_feedback = llm_scores['detailed_feedback']
            
            # Recalculate total score and grade
            result.calculate_total_score()
            
        except Exception as e:
            self.logger.error(f"Error evaluating submission {submission.student_id}: {e}")
            result.errors.append(f"Evaluation error: {str(e)}")
        
        return result
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to CSV"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'student_id', 'student_name', 'project_category', 'github_url',
                    'code_quality_score', 'functionality_score', 'requirements_fulfillment',
                    'code_structure_score', 'llm_generated_likelihood', 'assessment_goals_fulfillment',
                    'creativity_score', 'documentation_score', 'total_score', 'grade',
                    'detailed_feedback', 'warnings', 'errors'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = asdict(result)
                    row['warnings'] = '; '.join(result.warnings)
                    row['errors'] = '; '.join(result.errors)
                    writer.writerow(row)
            
            self.logger.info(f"Results saved to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {e}")
    
    def run_evaluation(self, submissions_csv: str, output_csv: str) -> List[EvaluationResult]:
        """Run the complete evaluation process"""
        self.logger.info("Starting TMP evaluation process")
        
        # Load submissions
        submissions = self.load_submissions(submissions_csv)
        if not submissions:
            self.logger.error("No submissions loaded")
            return []
        
        # Evaluate each submission
        results = []
        for i, submission in enumerate(submissions, 1):
            self.logger.info(f"Processing {i}/{len(submissions)}: {submission.student_name}")
            result = self.evaluate_submission(submission)
            results.append(result)
        
        # Save results
        self.save_results(results, output_csv)
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def generate_summary(self, results: List[EvaluationResult]):
        """Generate evaluation summary"""
        total_students = len(results)
        successful_evaluations = len([r for r in results if not r.errors])
        
        avg_score = sum(r.total_score for r in results) / total_students if total_students > 0 else 0
        
        grade_distribution = {}
        for result in results:
            grade_distribution[result.grade] = grade_distribution.get(result.grade, 0) + 1
        
        llm_detected = len([r for r in results if r.llm_generated_likelihood > 0.6])
        
        self.logger.info("=== EVALUATION SUMMARY ===")
        self.logger.info(f"Total students: {total_students}")
        self.logger.info(f"Successful evaluations: {successful_evaluations}")
        self.logger.info(f"Average score: {avg_score:.2f}")
        self.logger.info(f"Grade distribution: {grade_distribution}")
        self.logger.info(f"Possible LLM-generated projects: {llm_detected}")


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        'evaluation_criteria': {
            'data_analysis': {
                'required_files': ['*.py', '*.ipynb'],
                'required_libraries': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'],
                'evaluation_points': [
                    'Data loading and preprocessing',
                    'Exploratory data analysis with visualizations',
                    'Statistical analysis and hypothesis testing',
                    'Clear insights and conclusions',
                    'Code organization and documentation'
                ],
                'weight_factors': {
                    'technical_implementation': 0.4,
                    'analysis_quality': 0.3,
                    'visualization': 0.2,
                    'documentation': 0.1
                }
            },
            'crud': {
                'required_files': ['*.py'],
                'required_libraries': ['fastapi', 'flask', 'sqlalchemy', 'sqlite3', 'requests'],
                'evaluation_points': [
                    'Complete CRUD operations implementation',
                    'Proper API design and endpoints',
                    'Database integration and management',
                    'Error handling and validation',
                    'API documentation and testing'
                ],
                'weight_factors': {
                    'functionality': 0.4,
                    'api_design': 0.3,
                    'error_handling': 0.2,
                    'documentation': 0.1
                }
            },
            'script_tool': {
                'required_files': ['*.py'],
                'required_libraries': ['requests', 'beautifulsoup4', 'scrapy', 'selenium', 'pandas'],
                'evaluation_points': [
                    'Robust web scraping implementation',
                    'Data extraction accuracy and completeness',
                    'Error handling and retry mechanisms',
                    'Output in required format (CSV/JSON)',
                    'Code modularity and reusability'
                ],
                'weight_factors': {
                    'scraping_quality': 0.4,
                    'data_accuracy': 0.3,
                    'error_handling': 0.2,
                    'code_structure': 0.1
                }
            }
        },
        'llm': {
            'provider': 'openai',  # Options: 'openai', 'anthropic'
            'model': 'gpt-4',
            'api_key_env': 'OPENAI_API_KEY',
            'max_tokens': 2000,
            'temperature': 0.1,
            'fallback_model': 'gpt-3.5-turbo'
        },
        'scoring': {
            'llm_penalty_threshold': 0.6,
            'heavy_penalty_threshold': 0.8,
            'max_repo_size_mb': 100,
            'timeout_seconds': 300,
            'min_code_lines': 50,
            'max_file_size_mb': 10
        },
        'github': {
            'timeout_seconds': 30,
            'max_retries': 3,
            'preferred_branches': ['main', 'master', 'develop']
        }
    }
    
    with open('tmp_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("Sample configuration created: tmp_config.yaml")


def create_sample_submissions():
    """Create a sample submissions CSV file"""
    sample_data = [
        {
            'student_name': 'John Doe',
            'student_id': 'CS2023001',
            'email': 'john.doe@university.edu',
            'project_category': 'data_analysis',
            'github_url': 'https://github.com/johndoe/python-data-analysis',
            'submission_date': '2024-01-15',
            'additional_notes': 'Used Titanic dataset'
        },
        {
            'student_name': 'Jane Smith',
            'student_id': 'CS2023002',
            'email': 'jane.smith@university.edu',
            'project_category': 'crud',
            'github_url': 'https://github.com/janesmith/fastapi-todo-app',
            'submission_date': '2024-01-14',
            'additional_notes': 'FastAPI implementation with SQLite'
        },
        {
            'student_name': 'Bob Johnson',
            'student_id': 'CS2023003',
            'email': 'bob.johnson@university.edu',
            'project_category': 'script_tool',
            'github_url': 'https://github.com/bobjohnson/news-scraper',
            'submission_date': '2024-01-13',
            'additional_notes': 'Scrapes multiple news websites'
        }
    ]
    
    with open('sample_submissions.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['student_name', 'student_id', 'email', 'project_category', 
                     'github_url', 'submission_date', 'additional_notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)
    
    print("Sample submissions created: sample_submissions.csv")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TMP - Test My Python: Auto Evaluation System')
    parser.add_argument('--config', '-c', required=True, help='Configuration YAML file')
    parser.add_argument('--submissions', '-s', required=True, help='Submissions CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for results')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--create-sample-submissions', action='store_true',
                       help='Create sample submissions CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create sample files if requested
    if args.create_sample_config:
        create_sample_config()
        return
    
    if args.create_sample_submissions:
        create_sample_submissions()
        return
    
    # Check if required arguments are provided
    if not all([args.config, args.submissions, args.output]):
        parser.print_help()
        return
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    try:
        evaluator = TMPEvaluator(args.config)
        results = evaluator.run_evaluation(args.submissions, args.output)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {args.output}")
        print(f"Evaluated {len(results)} submissions")
        
        # Show quick summary
        if results:
            avg_score = sum(r.total_score for r in results) / len(results)
            print(f"Average score: {avg_score:.2f}")
            
            # Grade distribution
            grades = {}
            for result in results:
                grades[result.grade] = grades.get(result.grade, 0) + 1
            print(f"Grade distribution: {grades}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            evaluator.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()