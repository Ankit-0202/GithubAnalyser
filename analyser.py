"""
Title: GitHub Repository Analyser

Description:
This program pulls data using the GitHub API and visualises this data to deliver actionable insights, such as:

- Repository Metrics: Statistics including stars, forks, open issues and creation dates.
- Language Distribution: Breakdown of programming languages used across all repositories.
- Commit Activity Trends: Visualisation of commit frequency over time to identify development patterns.
- Top Contributors: Identification of key contributors across repositories.
- Issue Analysis: Insights into open and closed issues to assess project activity and maintenance.
- Visualisation Dashboards: Charts and graphs that depict the analysed data for easy interpretation.

Intended Use Case:
This analyser is ideal for developers, project managers and data analysts seeking to gain a holistic understanding of a GitHub user's repository landscape. Whether assessing personal projects or evaluating potential collaborators, this tool provides the necessary insights to make informed decisions.

Author: [Ankit Kapoor]
Date: 2024-04-27
"""

import os
import sys
import asyncio
import aiohttp
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from dotenv import load_dotenv
from datetime import datetime
import argparse
import jinja2
import aiofiles
import json

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("github_insights_analyser.log")
    ]
)
logger = logging.getLogger(__name__)

class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass

class GitHubRepoInsightsAnalyser:
    def __init__(self, username: str, token: Optional[str] = None, cache: bool = True):
        """
        Initialises the GitHubRepoInsightsAnalyser with a GitHub username and optional personal access token.
        
        Parameters:
            username (str): GitHub username to analyse.
            token (str, optional): Personal access token for authenticated API requests.
            cache (bool): Enable caching of API responses to reduce redundant calls.
        """
        self.username = username
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.cache = cache
        self.base_url = "https://api.github.com"
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.token:
            self.headers['Authorisation'] = f'token {self.token}'
        self.session = None
        self.repos = []
        self.repo_details = []
        logger.info(f"Initialised GitHubRepoInsightsAnalyser for user: {self.username}")
    
    async def fetch_json(self, url: str, session: aiohttp.ClientSession, params: Dict = {}) -> Dict:
        """
        Fetches JSON data from a given URL using aiohttp session.
        
        Parameters:
            url (str): The API endpoint to fetch data from.
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            params (Dict): Query parameters for the API request.
        
        Returns:
            Dict: The JSON response from the API.
        """
        async with session.get(url, params=params, headers=self.headers) as response:
            if response.status == 403:
                reset_time = response.headers.get('X-RateLimit-Reset')
                if reset_time:
                    reset_datetime = datetime.fromtimestamp(int(reset_time))
                    message = f"Rate limit exceeded. Reset at {reset_datetime}."
                else:
                    message = "Access forbidden: Possibly bad credentials or rate limit exceeded."
                logger.error(message)
                raise GitHubAPIError(message)
            elif response.status == 404:
                message = f"Resource not found: {url}"
                logger.error(message)
                raise GitHubAPIError(message)
            elif response.status >= 400:
                message = f"HTTP Error {response.status} for URL: {url}"
                logger.error(message)
                raise GitHubAPIError(message)
            return await response.json()
    
    async def fetch_repositories(self):
        """
        Asynchronously fetches all public repositories of the specified GitHub user.
        """
        logger.info("Starting to fetch repositories from GitHub API.")
        async with aiohttp.ClientSession() as session:
            self.session = session
            repos = []
            page = 1
            per_page = 100
            while True:
                url = f"{self.base_url}/users/{self.username}/repos"
                params = {'per_page': per_page, 'page': page, 'type': 'public'}
                try:
                    data = await self.fetch_json(url, session, params)
                    if not data:
                        break
                    repos.extend(data)
                    logger.info(f"Fetched page {page} with {len(data)} repositories.")
                    page += 1
                except GitHubAPIError as e:
                    logger.error(f"Error fetching repositories: {e}")
                    break
            self.repos = repos
            logger.info(f"Total repositories fetched: {len(self.repos)}")
    
    async def fetch_commit_activity(self, repo_name: str) -> Optional[List[Dict]]:
        """
        Asynchronously fetches commit activity for a specific repository.
        
        Parameters:
            repo_name (str): Name of the repository.
        
        Returns:
            Optional[List[Dict]]: Commit activity data or None if unavailable.
        """
        url = f"{self.base_url}/repos/{self.username}/{repo_name}/stats/commit_activity"
        try:
            data = await self.fetch_json(url, self.session)
            return data
        except GitHubAPIError as e:
            logger.warning(f"Commit activity for '{repo_name}' could not be fetched: {e}")
            return None
    
    async def fetch_contributors(self, repo_name: str) -> List[Dict]:
        """
        Asynchronously fetches contributors for a specific repository.
        
        Parameters:
            repo_name (str): Name of the repository.
        
        Returns:
            List[Dict]: List of contributors.
        """
        contributors = []
        page = 1
        per_page = 100
        while True:
            url = f"{self.base_url}/repos/{self.username}/{repo_name}/contributors"
            params = {'per_page': per_page, 'page': page}
            try:
                data = await self.fetch_json(url, self.session, params)
                if not data:
                    break
                contributors.extend(data)
                logger.info(f"Fetched page {page} with {len(data)} contributors for repository '{repo_name}'.")
                page += 1
            except GitHubAPIError as e:
                logger.warning(f"Contributors for '{repo_name}' could not be fetched: {e}")
                break
        return contributors
    
    async def gather_repo_details(self):
        """
        Asynchronously gathers detailed information for each repository.
        """
        logger.info("Starting to gather detailed repository information.")
        tasks = []
        for repo in self.repos:
            tasks.append(self.process_repository(repo))
        self.repo_details = await asyncio.gather(*tasks)
        logger.info("Completed gathering repository details.")
    
    async def process_repository(self, repo: Dict) -> Dict:
        """
        Processes individual repository data by fetching commit activity and contributors.
        
        Parameters:
            repo (Dict): Repository data.
        
        Returns:
            Dict: Processed repository metrics.
        """
        repo_info = {
            'Name': repo['name'],
            'Stars': repo['stargazers_count'],
            'Forks': repo['forks_count'],
            'Open Issues': repo['open_issues_count'],
            'Language': repo['language'] or 'Not Specified',
            'Created At': repo['created_at'],
            'Updated At': repo['updated_at'],
            'Pushed At': repo['pushed_at'],
            'Commits Last Week': 0,
            'Top Contributor': 'N/A',
            'Total Contributors': 0
        }
        commit_activity = await self.fetch_commit_activity(repo['name'])
        if commit_activity:
            weekly_commits = sum(week.get('total', 0) for week in commit_activity)
            repo_info['Commits Last Week'] = weekly_commits
        contributors = await self.fetch_contributors(repo['name'])
        if contributors:
            top_contributor = max(contributors, key=lambda c: c.get('contributions', 0))
            repo_info['Top Contributor'] = top_contributor.get('login', 'N/A')
            repo_info['Total Contributors'] = len(contributors)
        return repo_info
    
    def analyse_repositories(self) -> pd.DataFrame:
        """
        Analyses repository data and returns a DataFrame with key metrics.
        
        Returns:
            pd.DataFrame: DataFrame containing analysed repository metrics.
        """
        logger.info("Starting analysis of repositories.")
        df = pd.DataFrame(self.repo_details)
        logger.info("Repository analysis complete.")
        return df
    
    def summarise_data(self, df: pd.DataFrame) -> Dict:
        """
        Generates summary statistics from the repository DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing repository metrics.
        
        Returns:
            Dict: Summary of key insights.
        """
        logger.info("Generating summary statistics.")
        summary = {
            'Total Repositories': len(df),
            'Most Starred Repository': df.loc[df['Stars'].idxmax()][['Name', 'Stars']].to_dict(),
            'Most Forked Repository': df.loc[df['Forks'].idxmax()][['Name', 'Forks']].to_dict(),
            'Language Distribution': df['Language'].value_counts().to_dict(),
            'Average Stars': round(df['Stars'].mean(), 2),
            'Average Forks': round(df['Forks'].mean(), 2),
            'Total Commits Last Week': df['Commits Last Week'].sum(),
            'Top Overall Contributor': df['Top Contributor'].mode()[0] if not df['Top Contributor'].mode().empty else 'N/A',
            'Total Contributors Across Repos': df['Total Contributors'].sum(),
            'Average Contributors per Repo': round(df['Total Contributors'].mean(), 2)
        }
        logger.info("Summary statistics generated.")
        return summary
    
    def visualise_language_distribution(self, language_distribution: Dict):
        """
        Creates a pie chart for language distribution across repositories.
        
        Parameters:
            language_distribution (Dict): Dictionary with language counts.
        """
        logger.info("Creating language distribution pie chart.")
        languages = list(language_distribution.keys())
        counts = list(language_distribution.values())
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title('Language Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('language_distribution.png')
        plt.close()
        logger.info("Language distribution pie chart saved as 'language_distribution.png'.")
    
    def visualise_commit_activity(self, df: pd.DataFrame):
        """
        Creates a bar chart for commits in the last week across repositories.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing repository metrics.
        """
        logger.info("Creating commit activity bar chart.")
        plt.figure(figsize=(12, 6))
        df_sorted = df.sort_values(by='Commits Last Week', ascending=False)
        plt.bar(df_sorted['Name'], df_sorted['Commits Last Week'], color='skyblue')
        plt.xlabel('Repository')
        plt.ylabel('Commits Last Week')
        plt.title('Commits in the Last Week per Repository')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('commit_activity.png')
        plt.close()
        logger.info("Commit activity bar chart saved as 'commit_activity.png'.")
    
    def visualise_commit_trends(self, df: pd.DataFrame):
        """
        Creates a line chart showing commit trends over time for all repositories.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing repository metrics.
        """
        logger.info("Creating commit trends line chart.")
        # Aggregate commit activity data
        commit_trends = {}
        for repo in self.repo_details:
            commits = repo.get('Commits Last Week', 0)
            repo_name = repo['Name']
            commit_trends[repo_name] = commits
        plt.figure(figsize=(12, 6))
        repos = list(commit_trends.keys())
        commits = list(commit_trends.values())
        plt.plot(repos, commits, marker='o', linestyle='-', color='green')
        plt.xlabel('Repository')
        plt.ylabel('Commits Last Week')
        plt.title('Commit Trends Across Repositories')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('commit_trends.png')
        plt.close()
        logger.info("Commit trends line chart saved as 'commit_trends.png'.")
    
    def generate_html_report(self, summary: Dict, df: pd.DataFrame):
        """
        Generates an HTML report summarising the analysis with embedded visualisations.
        
        Parameters:
            summary (Dict): Summary statistics.
            df (pd.DataFrame): DataFrame containing repository metrics.
        """
        logger.info("Generating HTML report.")
        env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GitHub Repository Insights Report for {{ username }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #333; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 40px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>GitHub Repository Insights Report</h1>
            <h2>User: {{ username }}</h2>
            <h3>Summary Statistics</h3>
            <table>
                <tbody>
                    {% for key, value in summary.items() %}
                        <tr>
                            <th>{{ key }}</th>
                            <td>
                                {% if value is mapping %}
                                    <ul>
                                        {% for subkey, subvalue in value.items() %}
                                            <li>{{ subkey }}: {{ subvalue }}</li>
                                        {% endfor %}
                                    </ul>
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </td>
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
            <h3>Language Distribution</h3>
            <img src="language_distribution.png" alt="Language Distribution Pie Chart">
            <h3>Commit Activity</h3>
            <img src="commit_activity.png" alt="Commit Activity Bar Chart">
            <h3>Commit Trends</h3>
            <img src="commit_trends.png" alt="Commit Trends Line Chart">
            <h3>Repository Details</h3>
            {{ table | safe }}
        </body>
        </html>
        """
        template = env.from_string(template_str)
        table_html = df.to_html(index=False, classes='table table-striped')
        html_content = template.render(username=self.username, summary=summary, table=table_html)
        with open('github_insights_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("HTML report generated and saved as 'github_insights_report.html'.")
    
    def save_dataframe_as_csv(self, df: pd.DataFrame):
        """
        Saves the repository DataFrame to a CSV file.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing repository metrics.
        """
        df.to_csv('repository_details.csv', index=False)
        logger.info("Repository details saved as 'repository_details.csv'.")
    
    async def run_analysis(self):
        """
        Executes the full analysis workflow: fetching data, analysing, summarising and visualising.
        """
        try:
            await self.fetch_repositories()
            if not self.repos:
                logger.warning(f"No repositories found for user '{self.username}'. Exiting analysis.")
                return
            await self.gather_repo_details()
            df = self.analyse_repositories()
            summary = self.summarise_data(df)
            
            # Generate visualisations
            self.visualise_language_distribution(summary['Language Distribution'])
            self.visualise_commit_activity(df)
            self.visualise_commit_trends(df)
            
            # Save data
            self.save_dataframe_as_csv(df)
            
            # Generate HTML report
            self.generate_html_report(summary, df)
            
            logger.info("Analysis complete. All outputs have been generated.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during analysis: {e}")
    
def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Comprehensive GitHub Repository Insights Analyser")
    parser.add_argument('username', type=str, help='GitHub username to analyse')
    parser.add_argument('--token', type=str, help='GitHub personal access token (optional)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of API responses')
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyser = GitHubRepoInsightsAnalyser(username=args.username, token=args.token, cache=not args.no_cache)
    asyncio.run(analyser.run_analysis())

if __name__ == "__main__":
    main()
