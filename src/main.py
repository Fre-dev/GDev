from composio import Composio
from openai import OpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from composio_openai import OpenAIProvider
from fastapi import FastAPI, Request, HTTPException
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="GitHub Issue Analyzer", description="Analyze GitHub issues and suggest solutions", version="1.0.0")

# Initialize clients
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
composio = Composio(provider=OpenAIProvider(), api_key=os.getenv("COMPOSIO_API_KEY"))
user_id = "nks8839@nyu.edu"

# Pydantic models
class IssueAnalysis(BaseModel):
    issue_id: int
    title: str
    body: str
    analysis: str
    suggested_solution: str
    priority: str
    complexity: str

class RepositoryData(BaseModel):
    name: str
    description: str
    open_issues_count: int
    issues: List[Dict[str, Any]]

# Global variables to store connection state
connection_initialized = False
github_tools = None

async def initialize_github_connection():
    """Initialize GitHub connection if not already done"""
    global connection_initialized, github_tools
    
    if not connection_initialized:
        try:
            connection_request = composio.toolkits.authorize(user_id=user_id, toolkit="github")
            print(f"🔗 Visit the URL to authorize:\n👉 {connection_request.redirect_url}")
            
            # Get tools
            github_tools = composio.tools.get(user_id=user_id, toolkits=["GITHUB_LIST_REPOSITORY_ISSUES", "GITHUB_GET_A_REPOSITORY"])
            
            # Wait for connection to be active
            connection_request.wait_for_connection()
            connection_initialized = True
            print("✅ GitHub connection initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize GitHub connection: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize GitHub connection")

async def get_repository_data(repo_name: str) -> Dict[str, Any]:
    """Get repository details and issues"""
    await initialize_github_connection()
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked about GitHub repositories, you MUST use the available tools to fetch real data. Always use GITHUB_GET_A_REPOSITORY to get repository details and GITHUB_LIST_REPOSITORY_ISSUES to get issues."
                },
                {
                    "role": "user",
                    "content": f"Use the GitHub tools to get the full details and list of open issues for the public repo {repo_name}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        return result
        
    except Exception as e:
        print(f"❌ Error fetching repository data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch repository data: {str(e)}")

async def analyze_issue_with_llm(issue: Dict[str, Any], repo_context: Dict[str, Any]) -> IssueAnalysis:
    """Analyze an issue using LLM and suggest solutions"""
    try:
        # Extract issue details
        issue_title = issue.get('title', 'No title')
        issue_body = issue.get('body', 'No description')
        issue_id = issue.get('number', 0)
        
        # Create analysis prompt
        analysis_prompt = f"""
        Analyze this GitHub issue and provide suggestions:
        
        Repository: {repo_context.get('name', 'Unknown')}
        Repository Description: {repo_context.get('description', 'No description')}
        
        Issue #{issue_id}: {issue_title}
        Description: {issue_body}
        
        Please provide:
        1. A detailed analysis of the issue
        2. Suggested solution approach
        3. Priority level (High/Medium/Low)
        4. Complexity level (Simple/Medium/Complex)
        
        Format your response as a structured analysis.
        """
        
        # Make LLM call for analysis
        analysis_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior software engineer analyzing GitHub issues. Provide detailed technical analysis and actionable solutions."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        analysis_text = analysis_response.choices[0].message.content
        
        # Parse the analysis to extract components
        lines = analysis_text.split('\n')
        analysis_parts = {
            'analysis': '',
            'solution': '',
            'priority': 'Medium',
            'complexity': 'Medium'
        }
        
        current_section = 'analysis'
        for line in lines:
            line = line.strip()
            if 'suggested solution' in line.lower() or 'solution' in line.lower():
                current_section = 'solution'
            elif 'priority' in line.lower():
                if 'high' in line.lower():
                    analysis_parts['priority'] = 'High'
                elif 'low' in line.lower():
                    analysis_parts['priority'] = 'Low'
                else:
                    analysis_parts['priority'] = 'Medium'
            elif 'complexity' in line.lower():
                if 'complex' in line.lower():
                    analysis_parts['complexity'] = 'Complex'
                elif 'simple' in line.lower():
                    analysis_parts['complexity'] = 'Simple'
                else:
                    analysis_parts['complexity'] = 'Medium'
            elif line and not line.startswith('##'):
                analysis_parts[current_section] += line + ' '
        
        return IssueAnalysis(
            issue_id=issue_id,
            title=issue_title,
            body=issue_body or "No description provided",
            analysis=analysis_parts['analysis'].strip() or analysis_text,
            suggested_solution=analysis_parts['solution'].strip() or "Solution analysis included in main analysis",
            priority=analysis_parts['priority'],
            complexity=analysis_parts['complexity']
        )
        
    except Exception as e:
        print(f"❌ Error analyzing issue: {e}")
        # Return basic analysis if LLM call fails
        return IssueAnalysis(
            issue_id=issue.get('number', 0),
            title=issue.get('title', 'No title'),
            body=issue.get('body', 'No description'),
            analysis=f"Error analyzing issue: {str(e)}",
            suggested_solution="Manual analysis required",
            priority="Medium",
            complexity="Medium"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GitHub Issue Analyzer API", "status": "running"}

@app.get("/repository/{owner}/{repo}/issues", response_model=List[IssueAnalysis])
async def analyze_repository_issues(owner: str, repo: str):
    """Get and analyze all issues for a repository"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository data and issues
        repo_data = await get_repository_data(repo_name)
        
        # Extract repository info and issues
        repository_info = None
        issues_data = []
        
        if isinstance(repo_data, list):
            for item in repo_data:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'name' in data:  # This is repository data
                        repository_info = data
                    elif 'details' in data:  # This is issues data
                        issues_data = data['details']
        
        if not repository_info:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Analyze each issue
        analyzed_issues = []
        for issue in issues_data:
            if issue.get('state') == 'open':  # Only analyze open issues
                analysis = await analyze_issue_with_llm(issue, repository_info)
                analyzed_issues.append(analysis)
        
        return analyzed_issues
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing repository issues: {str(e)}")

@app.get("/repository/{owner}/{repo}/info")
async def get_repository_info(owner: str, repo: str):
    """Get basic repository information"""
    repo_name = f"{owner}/{repo}"
    
    try:
        repo_data = await get_repository_data(repo_name)
        
        # Extract repository info
        for item in repo_data:
            if item.get('successful') and item.get('data') and 'name' in item['data']:
                data = item['data']
                return {
                    "name": data.get('name'),
                    "full_name": data.get('full_name'),
                    "description": data.get('description'),
                    "language": data.get('language'),
                    "open_issues_count": data.get('open_issues_count'),
                    "stargazers_count": data.get('stargazers_count'),
                    "created_at": data.get('created_at'),
                    "updated_at": data.get('updated_at')
                }
        
        raise HTTPException(status_code=404, detail="Repository not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repository info: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "github_connected": connection_initialized}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



