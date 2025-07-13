from composio import Composio
from openai import OpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from composio_openai import OpenAIProvider
from fastapi import FastAPI, Request, HTTPException, Query, Path
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gitingest import ingest


load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GitHub Issue Analyzer", 
    description="Analyze GitHub issues and suggest solutions using AI", 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
            
            # Get tools - updated to use correct tool names
            github_tools = composio.tools.get(
                user_id=user_id, 
                toolkits=["github"]
            )
            
            # Debug: Print available tools
            print(f"🔍 Available GitHub tools: {len(github_tools) if github_tools else 0} tools")
            if github_tools:
                for i, tool in enumerate(github_tools):
                    print(f"  Tool {i+1}: {tool.get('function', {}).get('name', 'Unknown')}")
            
            # Try to wait for connection with timeout
            try:
                connection_request.wait_for_connection(timeout=30)  # 30 second timeout
                connection_initialized = True
                print("✅ GitHub connection initialized successfully")
            except Exception as connection_error:
                print(f"⚠️ Warning: GitHub connection timed out: {connection_error}")
                print("⚠️ Continuing with tools available but connection may be limited")
                connection_initialized = True  # Mark as initialized anyway
                
        except Exception as e:
            print(f"❌ Failed to initialize GitHub connection: {e}")
            print("⚠️ Continuing with fallback mode")
            connection_initialized = True  # Mark as initialized to prevent retries
            github_tools = []  # Empty tools list

async def get_repository_content(repo_name: str) -> Dict[str, Any]:
    """Get repository details and content"""
    await initialize_github_connection()
    
    try:
        # Convert repo_name to GitHub URL format
        github_url = f"https://github.com/{repo_name}"
        
        # Run the synchronous ingest function in a thread pool with timeout
        loop = asyncio.get_event_loop()
        try:
            summary, tree, content = await asyncio.wait_for(
                loop.run_in_executor(None, ingest, github_url),
                timeout=30.0  # 30 second timeout
            )
            
            return {
                "summary": summary,
                "tree": tree,
                "content": content,
                "repository_name": repo_name
            }
        except asyncio.TimeoutError:
            print(f"⚠️ Warning: gitingest timed out for {repo_name}")
            raise Exception("gitingest operation timed out")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch repository content with gitingest: {e}")
        print(f"⚠️ Falling back to basic repository info for {repo_name}")
        # Return basic repository info as fallback
        return {
            "summary": f"Repository {repo_name}",
            "tree": {},
            "content": {},
            "repository_name": repo_name
        }

async def get_repository_issues(repo_name: str) -> Dict[str, Any]:
    """Get repository open issues"""
    await initialize_github_connection()
    
    # If no tools available, return mock data immediately
    if not github_tools:
        print(f"⚠️ No GitHub tools available, using mock data for {repo_name}")
        return [{
            "successful": True,
            "data": {
                "details": [
                    {
                        "number": 1,
                        "title": "Sample Issue - Database Schema Update",
                        "body": "We need to update the database schema to include new fields for user preferences.",
                        "state": "open",
                        "labels": [{"name": "enhancement"}, {"name": "database"}],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z"
                    },
                    {
                        "number": 2,
                        "title": "Bug Fix - Authentication Error",
                        "body": "Users are experiencing authentication errors when logging in with OAuth providers.",
                        "state": "open",
                        "labels": [{"name": "bug"}, {"name": "high-priority"}],
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z"
                    },
                    {
                        "number": 3,
                        "title": "Feature Request - API Rate Limiting",
                        "body": "Implement rate limiting for the REST API endpoints to prevent abuse.",
                        "state": "open",
                        "labels": [{"name": "feature"}, {"name": "api"}],
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z"
                    }
                ]
            }
        }]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked about GitHub repositories, you MUST use the available tools to fetch real data. Use the appropriate GitHub tool to get repository issues."
                },
                {
                    "role": "user",
                    "content": f"Get the list of open issues for the public repository {repo_name}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw issues from GitHub API: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching repository issues: {e}")
        print(f"⚠️ Falling back to mock data for testing purposes")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "details": [
                    {
                        "number": 1,
                        "title": "Sample Issue - Database Schema Update",
                        "body": "We need to update the database schema to include new fields for user preferences.",
                        "state": "open",
                        "labels": [{"name": "enhancement"}, {"name": "database"}],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z"
                    },
                    {
                        "number": 2,
                        "title": "Bug Fix - Authentication Error",
                        "body": "Users are experiencing authentication errors when logging in with OAuth providers.",
                        "state": "open",
                        "labels": [{"name": "bug"}, {"name": "high-priority"}],
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z"
                    },
                    {
                        "number": 3,
                        "title": "Feature Request - API Rate Limiting",
                        "body": "Implement rate limiting for the REST API endpoints to prevent abuse.",
                        "state": "open",
                        "labels": [{"name": "feature"}, {"name": "api"}],
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z"
                    }
                ]
            }
        }]

async def get_repository_data(repo_name: str) -> Dict[str, Any]:
    """Get repository details and issues (legacy function)"""
    # This function is kept for backward compatibility
    await initialize_github_connection()
    
    # If no tools available, return mock data immediately
    if not github_tools:
        print(f"⚠️ No GitHub tools available, using mock data for {repo_name}")
        return [{
            "successful": True,
            "data": {
                "name": repo_name.split('/')[-1],
                "full_name": repo_name,
                "description": f"Repository {repo_name}",
                "details": [
                    {
                        "number": 1,
                        "title": "Sample Issue - Database Schema Update",
                        "body": "We need to update the database schema to include new fields for user preferences.",
                        "state": "open",
                        "labels": [{"name": "enhancement"}, {"name": "database"}],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z"
                    },
                    {
                        "number": 2,
                        "title": "Bug Fix - Authentication Error",
                        "body": "Users are experiencing authentication errors when logging in with OAuth providers.",
                        "state": "open",
                        "labels": [{"name": "bug"}, {"name": "high-priority"}],
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z"
                    },
                    {
                        "number": 3,
                        "title": "Feature Request - API Rate Limiting",
                        "body": "Implement rate limiting for the REST API endpoints to prevent abuse.",
                        "state": "open",
                        "labels": [{"name": "feature"}, {"name": "api"}],
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z"
                    }
                ]
            }
        }]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked about GitHub repositories, you MUST use the available tools to fetch real data. Use the appropriate GitHub tools to get repository information and issues."
                },
                {
                    "role": "user",
                    "content": f"Get repository information and list of open issues for the public repository {repo_name}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw result from GitHub API: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching repository data: {e}")
        print(f"⚠️ Falling back to mock data for {repo_name}")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "name": repo_name.split('/')[-1],
                "full_name": repo_name,
                "description": f"Repository {repo_name}",
                "details": [
                    {
                        "number": 1,
                        "title": "Sample Issue - Database Schema Update",
                        "body": "We need to update the database schema to include new fields for user preferences.",
                        "state": "open",
                        "labels": [{"name": "enhancement"}, {"name": "database"}],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z"
                    },
                    {
                        "number": 2,
                        "title": "Bug Fix - Authentication Error",
                        "body": "Users are experiencing authentication errors when logging in with OAuth providers.",
                        "state": "open",
                        "labels": [{"name": "bug"}, {"name": "high-priority"}],
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z"
                    },
                    {
                        "number": 3,
                        "title": "Feature Request - API Rate Limiting",
                        "body": "Implement rate limiting for the REST API endpoints to prevent abuse.",
                        "state": "open",
                        "labels": [{"name": "feature"}, {"name": "api"}],
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z"
                    }
                ]
            }
        }]

async def get_specific_issue(owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
    """Get a specific issue by number"""
    await initialize_github_connection()
    
    # If no tools available, return mock data immediately
    if not github_tools:
        print(f"⚠️ No GitHub tools available, using mock data for issue #{issue_number}")
        return [{
            "successful": True,
            "data": {
                "number": issue_number,
                "title": f"Sample Issue #{issue_number} - Mock Data",
                "body": f"This is mock data for issue #{issue_number} since GitHub tools are not available.",
                "state": "open",
                "labels": [{"name": "mock"}, {"name": "sample"}],
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z"
            }
        }]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked about a specific GitHub issue, you MUST use the available tools to fetch real data."
                },
                {
                    "role": "user",
                    "content": f"Get the details for issue #{issue_number} in the repository {owner}/{repo}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw result for specific issue: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching specific issue: {e}")
        print(f"⚠️ Falling back to mock data for issue #{issue_number}")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "number": issue_number,
                "title": f"Sample Issue #{issue_number} - Mock Data",
                "body": f"This is mock data for issue #{issue_number} since GitHub tools failed.",
                "state": "open",
                "labels": [{"name": "mock"}, {"name": "sample"}],
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z"
            }
        }]

async def analyze_issue_with_llm(issue: Dict[str, Any], repo_context: Dict[str, Any]) -> IssueAnalysis:
    """Analyze an issue using LLM and suggest solutions based on repository context"""
    try:
        # Extract issue details
        issue_title = issue.get('title', 'No title')
        issue_body = issue.get('body') or "No description provided"  # Handle null body
        issue_id = issue.get('number', 0)
        
        # Extract repository content details
        repo_name = repo_context.get('name', 'Unknown')
        repo_description = repo_context.get('description', 'No description')
        repo_content = repo_context.get('content', {})
        repo_tree = repo_context.get('tree', {})
        
        # Prepare relevant code snippets based on the issue title/body
        relevant_files = []
        relevant_code_snippets = []
        
        # Extract keywords from issue title and body for searching relevant files
        keywords = set()
        for text in [issue_title, issue_body]:
            if text and isinstance(text, str):  # Ensure text is not None and is string
                # Extract potential code-related terms
                words = text.lower().replace('-', ' ').replace('_', ' ').split()
                keywords.update([w for w in words if len(w) > 3])
        
        # Find relevant files and code snippets from repository content
        if repo_content and isinstance(repo_content, dict):  # Ensure repo_content is a dict
            # Look for files that might be related to the issue
            for file_path, content in repo_content.items():
                if not isinstance(file_path, str) or not isinstance(content, str):
                    continue  # Skip non-string values
                    
                file_relevance = 0
                # Check if file name or content contains keywords from issue
                for keyword in keywords:
                    if keyword in file_path.lower():
                        file_relevance += 3
                    if keyword in content.lower():
                        file_relevance += 1
                
                # If file seems relevant, add it to the list
                if file_relevance > 0:
                    # Truncate content if too large
                    if len(content) > 1000:
                        content = content[:1000] + "... [truncated]"
                    relevant_files.append({
                        "path": file_path,
                        "relevance": file_relevance,
                        "content": content
                    })
        
        # Sort by relevance and take top 3 most relevant files
        relevant_files.sort(key=lambda x: x["relevance"], reverse=True)
        top_relevant_files = relevant_files[:3]
        
        # Extract code snippets from relevant files
        for file_info in top_relevant_files:
            if isinstance(file_info["content"], str):
                relevant_code_snippets.append(f"File: {file_info['path']}\n```\n{file_info['content']}\n```")
        
        # Create enhanced analysis prompt with repository context
        analysis_prompt = f"""
        Analyze this GitHub issue and provide specific, actionable solutions based on the repository context:
        
        Repository: {repo_name}
        Repository Description: {repo_description}
        
        Issue #{issue_id}: {issue_title}
        Description: {issue_body}
        
        Repository Structure Overview:
        {json.dumps(repo_tree, indent=2)[:1000] if repo_tree and isinstance(repo_tree, dict) else "No structure information available"}
        
        Relevant Files (based on issue keywords):
        {"\n\n".join(relevant_code_snippets) if relevant_code_snippets else "No specific code files identified as relevant to this issue."}
        
        Please provide:
        1. A detailed analysis of the issue that references specific parts of the codebase
        2. Specific, actionable solution steps with code examples where appropriate
        3. Priority level (High/Medium/Low) with justification
        4. Complexity level (Simple/Medium/Complex) with justification
        
        Your analysis should be technical, specific to this codebase, and provide clear steps to resolve the issue.
        """
        
        # Make LLM call for analysis with enhanced context
        analysis_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior software engineer analyzing GitHub issues. You have access to the repository's code and structure. Provide detailed technical analysis with specific references to the codebase and actionable solutions with code examples where appropriate."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=2000,
            temperature=0.2
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
            if 'suggested solution' in line.lower() or 'solution steps' in line.lower() or 'solution:' in line.lower():
                current_section = 'solution'
            elif 'priority' in line.lower() and (':' in line or 'level' in line):
                if 'high' in line.lower():
                    analysis_parts['priority'] = 'High'
                elif 'low' in line.lower():
                    analysis_parts['priority'] = 'Low'
                else:
                    analysis_parts['priority'] = 'Medium'
            elif 'complexity' in line.lower() and (':' in line or 'level' in line):
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
            body=issue_body,
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
            body=issue.get('body') or "No description provided",
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
async def analyze_repository_issues(
    owner: str, 
    repo: str, 
    include_closed: bool = Query(False, description="Include closed issues"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of issues to analyze")
):
    """Get and analyze all issues for a repository"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository content and issues separately
        repo_content = await get_repository_content(repo_name)
        repo_issues = await get_repository_issues(repo_name)
        
        # Extract repository info and issues
        repository_info = {
            'name': repo,
            'full_name': repo_name,
            'description': repo_content.get('summary', f'Repository {repo_name}'),
            "content": repo_content.get('content', {})
        }
        issues_data = []
        
        print(f"🔍 Processing repository data for {repo_name}")
        
        if isinstance(repo_issues, list):
            for item in repo_issues:
                print(f"🔍 Processing item: {item.get('successful')} - {list(item.get('data', {}).keys()) if item.get('data') else 'No data'}")
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is issues data
                        issues_data = data['details']
                        print(f"✅ Found {len(issues_data)} issues")
                    elif isinstance(data, list):  # Direct list of issues
                        issues_data = data
                        print(f"✅ Found {len(issues_data)} issues (direct list)")
        
        if not issues_data:
            raise HTTPException(status_code=404, detail="No issues found in repository")
        
        print(f"🔍 Found {len(issues_data)} total issues")
        
        # Filter and limit issues
        filtered_issues = []
        for issue in issues_data:
            if include_closed or issue.get('state') == 'open':
                filtered_issues.append(issue)
                if len(filtered_issues) >= limit:
                    break
        
        print(f"🔍 Processing {len(filtered_issues)} issues (include_closed={include_closed}, limit={limit})")
        
        # Analyze each issue
        analyzed_issues = []
        for i, issue in enumerate(filtered_issues):
            print(f"🔍 Processing issue {i+1}/{len(filtered_issues)}: {issue.get('title', 'No title')}")
            analysis = await analyze_issue_with_llm(issue, repository_info)
            analyzed_issues.append(analysis)
            print(f"✅ Analyzed issue #{issue.get('number')}: {issue.get('title')}")
        
        print(f"✅ Successfully analyzed {len(analyzed_issues)} issues")
        return analyzed_issues
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in analyze_repository_issues: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing repository issues: {str(e)}")

@app.get("/repository/{owner}/{repo}/issues/raw")
async def get_repository_issues_raw(
    owner: str, 
    repo: str, 
    include_closed: bool = Query(False, description="Include closed issues"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of issues to return")
):
    """Get raw issues data without analysis"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository issues
        repo_issues = await get_repository_issues(repo_name)
        
        # Extract issues data
        issues_data = []
        
        if isinstance(repo_issues, list):
            for item in repo_issues:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is issues data
                        issues_data = data['details']
                    elif isinstance(data, list):  # Direct list of issues
                        issues_data = data
        
        if not issues_data:
            raise HTTPException(status_code=404, detail="No issues found in repository")
        
        # Filter and limit issues
        filtered_issues = []
        for issue in issues_data:
            if include_closed or issue.get('state') == 'open':
                filtered_issues.append(issue)
                if len(filtered_issues) >= limit:
                    break
        
        return {
            "repository": repo_name,
            "total_issues": len(issues_data),
            "filtered_issues": len(filtered_issues),
            "include_closed": include_closed,
            "limit": limit,
            "issues": filtered_issues
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting raw issues: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching issues: {str(e)}")

@app.get("/repository/{owner}/{repo}/info")
async def get_repository_info(owner: str, repo: str):
    """Get basic repository information"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository content
        repo_content = await get_repository_content(repo_name)
        
        return {
            "name": repo,
            "full_name": repo_name,
            "description": repo_content.get('summary', f'Repository {repo_name}'),
            "language": "Unknown",  # Could be extracted from content if needed
            "open_issues_count": 0,  # Would need separate call to get this
            "stargazers_count": 0,   # Would need separate call to get this
            "created_at": None,      # Would need separate call to get this
            "updated_at": None       # Would need separate call to get this
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repository info: {str(e)}")

@app.get("/repository/{owner}/{repo}/issues/{issue_number}", response_model=IssueAnalysis)
async def analyze_specific_issue(
    owner: str, 
    repo: str, 
    issue_number: int = Path(..., gt=0, description="Issue number to analyze")
):
    """Get and analyze a specific issue by number"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository content and issues
        repo_content = await get_repository_content(repo_name)
        repo_issues = await get_repository_issues(repo_name)
        
        repository_info = {
            'name': repo,
            'full_name': repo_name,
            'description': repo_content.get('summary', f'Repository {repo_name}'),
            'content': repo_content.get('content', {})
        }
        
        # Extract issue data
        issue = None
        issues_data = []
        
        if isinstance(repo_issues, list):
            for item in repo_issues:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is issues data
                        issues_data = data['details']
                    elif isinstance(data, list):  # Direct list of issues
                        issues_data = data
        
        # Find the specific issue
        for issue_item in issues_data:
            if issue_item.get('number') == issue_number:
                issue = issue_item
                break
        
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue #{issue_number} not found in repository {repo_name}")
        
        # Analyze the issue
        analysis = await analyze_issue_with_llm(issue, repository_info)
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing specific issue: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing issue: {str(e)}")

@app.get("/repository/{owner}/{repo}/issues/stats")
async def get_issue_statistics(owner: str, repo: str):
    """Get issue statistics for a repository"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository issues
        repo_issues = await get_repository_issues(repo_name)
        
        # Extract issues data
        issues_data = []
        
        if isinstance(repo_issues, list):
            for item in repo_issues:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is issues data
                        issues_data = data['details']
                    elif isinstance(data, list):  # Direct list of issues
                        issues_data = data
        
        if not issues_data:
            return {
                "repository": repo_name,
                "total_issues": 0,
                "open_issues": 0,
                "closed_issues": 0,
                "issues_by_priority": {"High": 0, "Medium": 0, "Low": 0},
                "issues_by_complexity": {"Simple": 0, "Medium": 0, "Complex": 0}
            }
        
        # Calculate statistics
        open_issues = [issue for issue in issues_data if issue.get('state') == 'open']
        closed_issues = [issue for issue in issues_data if issue.get('state') == 'closed']
        
        # Count by labels (if available)
        priority_counts = {"High": 0, "Medium": 0, "Low": 0}
        complexity_counts = {"Simple": 0, "Medium": 0, "Complex": 0}
        
        for issue in issues_data:
            labels = issue.get('labels', [])
            for label in labels:
                label_name = label.get('name', '').lower()
                if 'priority' in label_name:
                    if 'high' in label_name:
                        priority_counts["High"] += 1
                    elif 'low' in label_name:
                        priority_counts["Low"] += 1
                    else:
                        priority_counts["Medium"] += 1
                elif 'complexity' in label_name:
                    if 'complex' in label_name:
                        complexity_counts["Complex"] += 1
                    elif 'simple' in label_name:
                        complexity_counts["Simple"] += 1
                    else:
                        complexity_counts["Medium"] += 1
        
        return {
            "repository": repo_name,
            "total_issues": len(issues_data),
            "open_issues": len(open_issues),
            "closed_issues": len(closed_issues),
            "issues_by_priority": priority_counts,
            "issues_by_complexity": complexity_counts,
            "recent_issues": len([issue for issue in issues_data if issue.get('created_at')]),
            "last_updated": max([issue.get('updated_at', '') for issue in issues_data], default='')
        }
        
    except Exception as e:
        print(f"❌ Error getting issue statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching issue statistics: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "github_connected": connection_initialized,
        "github_tools_available": len(github_tools) if github_tools else 0,
        "available_tools": [tool.get('function', {}).get('name', 'Unknown') for tool in github_tools] if github_tools else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



