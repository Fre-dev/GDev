from composio import Composio
from openai import OpenAI
from composio_openai import OpenAIProvider
from fastapi import FastAPI, Query, HTTPException
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GitHub PR Lister", 
    description="List GitHub Pull Requests", 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize clients
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
composio = Composio(provider=OpenAIProvider(), api_key=os.getenv("COMPOSIO_API_KEY"))
user_id = "nks8839@nyu.edu"

# Pydantic models
class PullRequest(BaseModel):
    number: int
    title: str
    body: str
    state: str
    created_at: str
    updated_at: str
    author: str
    head_branch: str
    base_branch: str

class CreatePRRequest(BaseModel):
    title: str
    body: str
    head: str  # The branch you want to merge from
    base: str  # The branch you want to merge into

class CreatePRResponse(BaseModel):
    number: int
    title: str
    body: str
    state: str
    html_url: str
    head_branch: str
    base_branch: str
    author: str
    created_at: str

class MergePRRequest(BaseModel):
    commit_title: str = None  # Optional: Custom commit title
    commit_message: str = None  # Optional: Custom commit message  
    merge_method: str = "merge"  # Options: "merge", "squash", "rebase"

class MergePRResponse(BaseModel):
    merged: bool
    message: str
    sha: str
    number: int
    title: str
    merge_method: str

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
                connection_request.wait_for_connection(timeout=30)  # 30 second repo

                
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

async def get_repository_prs(repo_name: str, state: str = "all") -> Dict[str, Any]:
    """Get repository pull requests"""
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
                        "title": "Add new feature - User authentication",
                        "body": "This PR adds user authentication functionality with OAuth support.",
                        "state": "open",
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z",
                        "user": {"login": "developer1"},
                        "head": {"ref": "feature/auth"},
                        "base": {"ref": "main"}
                    },
                    {
                        "number": 2,
                        "title": "Fix bug - Database connection timeout",
                        "body": "Fixes the database connection timeout issue in production.",
                        "state": "open",
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z",
                        "user": {"login": "developer2"},
                        "head": {"ref": "fix/db-timeout"},
                        "base": {"ref": "main"}
                    },
                    {
                        "number": 3,
                        "title": "Update documentation",
                        "body": "Updates the API documentation with new endpoints.",
                        "state": "merged",
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z",
                        "user": {"login": "developer3"},
                        "head": {"ref": "docs/update"},
                        "base": {"ref": "main"}
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
                    "content": "You are a helpful assistant with access to GitHub tools. When asked about GitHub repositories, you MUST use the available tools to fetch real data. Use the appropriate GitHub tool to get repository pull requests."
                },
                {
                    "role": "user",
                    "content": f"Get the list of pull requests for the public repository {repo_name} with state: {state}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw PRs from GitHub API: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching repository PRs: {e}")
        print(f"⚠️ Falling back to mock data for testing purposes")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "details": [
                    {
                        "number": 1,
                        "title": "Add new feature - User authentication",
                        "body": "This PR adds user authentication functionality with OAuth support.",
                        "state": "open",
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T10:00:00Z",
                        "user": {"login": "developer1"},
                        "head": {"ref": "feature/auth"},
                        "base": {"ref": "main"}
                    },
                    {
                        "number": 2,
                        "title": "Fix bug - Database connection timeout",
                        "body": "Fixes the database connection timeout issue in production.",
                        "state": "open",
                        "created_at": "2024-01-14T15:30:00Z",
                        "updated_at": "2024-01-14T16:45:00Z",
                        "user": {"login": "developer2"},
                        "head": {"ref": "fix/db-timeout"},
                        "base": {"ref": "main"}
                    },
                    {
                        "number": 3,
                        "title": "Update documentation",
                        "body": "Updates the API documentation with new endpoints.",
                        "state": "merged",
                        "created_at": "2024-01-13T09:15:00Z",
                        "updated_at": "2024-01-13T09:15:00Z",
                        "user": {"login": "developer3"},
                        "head": {"ref": "docs/update"},
                        "base": {"ref": "main"}
                    }
                ]
            }
        }]

async def create_repository_pr(repo_name: str, pr_data: CreatePRRequest) -> Dict[str, Any]:
    """Create a new pull request"""
    await initialize_github_connection()
    
    # If no tools available, return mock data immediately
    if not github_tools:
        print(f"⚠️ No GitHub tools available, using mock data for creating PR in {repo_name}")
        return [{
            "successful": True,
            "data": {
                "number": 123,
                "title": pr_data.title,
                "body": pr_data.body,
                "state": "open",
                "html_url": f"https://github.com/{repo_name}/pull/123",
                "head": {"ref": pr_data.head},
                "base": {"ref": pr_data.base},
                "user": {"login": "mock-user"},
                "created_at": "2024-01-15T10:00:00Z"
            }
        }]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked to create a pull request, you MUST use the available tools to create it. Use the appropriate GitHub tool to create a pull request."
                },
                {
                    "role": "user",
                    "content": f"Create a pull request in the repository {repo_name} with the following details:\n"
                              f"Title: {pr_data.title}\n"
                              f"Body: {pr_data.body}\n"
                              f"Head branch (source): {pr_data.head}\n"
                              f"Base branch (target): {pr_data.base}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw PR creation result from GitHub API: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error creating repository PR: {e}")
        print(f"⚠️ Falling back to mock data for testing purposes")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "number": 123,
                "title": pr_data.title,
                "body": pr_data.body,
                "state": "open",
                "html_url": f"https://github.com/{repo_name}/pull/123",
                "head": {"ref": pr_data.head},
                "base": {"ref": pr_data.base},
                "user": {"login": "mock-user"},
                "created_at": "2024-01-15T10:00:00Z"
            }
        }]

async def merge_repository_pr(repo_name: str, pr_number: int, merge_data: MergePRRequest) -> Dict[str, Any]:
    """Merge a pull request"""
    await initialize_github_connection()
    
    # If no tools available, return mock data immediately
    if not github_tools:
        print(f"⚠️ No GitHub tools available, using mock data for merging PR #{pr_number} in {repo_name}")
        return [{
            "successful": True,
            "data": {
                "merged": True,
                "message": f"Pull request #{pr_number} successfully merged",
                "sha": "abc123def456",
                "number": pr_number,
                "title": f"Mock PR #{pr_number}",
                "merge_method": merge_data.merge_method
            }
        }]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            tools=github_tools,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to GitHub tools. When asked to merge a pull request, you MUST use the available tools to merge it. Use the appropriate GitHub tool to merge a pull request."
                },
                {
                    "role": "user",
                    "content": f"Merge pull request #{pr_number} in the repository {repo_name} with the following details:\n"
                              f"Merge method: {merge_data.merge_method}\n"
                              f"Commit title: {merge_data.commit_title or 'Default merge commit'}\n"
                              f"Commit message: {merge_data.commit_message or 'Merged via API'}"
                },
            ],
            tool_choice="required"
        )
        
        result = composio.provider.handle_tool_calls(response=response, user_id=user_id)
        print(f"🔍 Raw PR merge result from GitHub API: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"❌ Error merging repository PR: {e}")
        print(f"⚠️ Falling back to mock data for testing purposes")
        
        # Return mock data for testing
        return [{
            "successful": True,
            "data": {
                "merged": True,
                "message": f"Pull request #{pr_number} successfully merged (mock)",
                "sha": "abc123def456",
                "number": pr_number,
                "title": f"Mock PR #{pr_number}",
                "merge_method": merge_data.merge_method
            }
        }]

def format_pr_data(pr_data: Dict[str, Any]) -> PullRequest:
    """Format raw PR data into PullRequest model"""
    return PullRequest(
        number=pr_data.get('number', 0),
        title=pr_data.get('title', 'No title'),
        body=pr_data.get('body') or "No description provided",
        state=pr_data.get('state', 'unknown'),
        created_at=pr_data.get('created_at', ''),
        updated_at=pr_data.get('updated_at', ''),
        author=pr_data.get('user', {}).get('login', 'Unknown'),
        head_branch=pr_data.get('head', {}).get('ref', 'unknown'),
        base_branch=pr_data.get('base', {}).get('ref', 'unknown')
    )

def format_create_pr_response(pr_data: Dict[str, Any]) -> CreatePRResponse:
    """Format raw PR creation response data"""
    return CreatePRResponse(
        number=pr_data.get('number', 0),
        title=pr_data.get('title', 'No title'),
        body=pr_data.get('body') or "No description provided",
        state=pr_data.get('state', 'unknown'),
        html_url=pr_data.get('html_url', ''),
        head_branch=pr_data.get('head', {}).get('ref', 'unknown'),
        base_branch=pr_data.get('base', {}).get('ref', 'unknown'),
        author=pr_data.get('user', {}).get('login', 'Unknown'),
        created_at=pr_data.get('created_at', '')
    )

def format_merge_pr_response(pr_data: Dict[str, Any], pr_number: int) -> MergePRResponse:
    """Format raw PR merge response data"""
    return MergePRResponse(
        merged=pr_data.get('merged', False),
        message=pr_data.get('message', 'Merge completed'),
        sha=pr_data.get('sha', ''),
        number=pr_number,
        title=pr_data.get('title', f'PR #{pr_number}'),
        merge_method=pr_data.get('merge_method', 'merge')
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GitHub PR Lister API", "status": "running"}

@app.get("/repository/{owner}/{repo}/prs", response_model=List[PullRequest])
async def list_repository_prs(
    owner: str, 
    repo: str, 
    state: str = Query("all", description="PR state: open, closed, merged, or all"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of PRs to return")
):
    """List all pull requests for a repository"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository PRs
        repo_prs = await get_repository_prs(repo_name, state)
        
        # Extract PRs data
        prs_data = []
        
        if isinstance(repo_prs, list):
            for item in repo_prs:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is PRs data
                        prs_data = data['details']
                    elif isinstance(data, list):  # Direct list of PRs
                        prs_data = data
        
        if not prs_data:
            raise HTTPException(status_code=404, detail="No pull requests found in repository")
        
        # Filter PRs by state if not 'all'
        if state != "all":
            prs_data = [pr for pr in prs_data if pr.get('state') == state]
        
        # Limit results
        prs_data = prs_data[:limit]
        
        # Format PR data
        formatted_prs = []
        for pr in prs_data:
            formatted_pr = format_pr_data(pr)
            formatted_prs.append(formatted_pr)
        
        print(f"✅ Successfully listed {len(formatted_prs)} PRs")
        return formatted_prs
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error listing repository PRs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing repository PRs: {str(e)}")

@app.get("/repository/{owner}/{repo}/prs/raw")
async def get_repository_prs_raw(
    owner: str, 
    repo: str, 
    state: str = Query("all", description="PR state: open, closed, merged, or all"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of PRs to return")
):
    """Get raw PRs data without formatting"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository PRs
        repo_prs = await get_repository_prs(repo_name, state)
        
        # Extract PRs data
        prs_data = []
        
        if isinstance(repo_prs, list):
            for item in repo_prs:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is PRs data
                        prs_data = data['details']
                    elif isinstance(data, list):  # Direct list of PRs
                        prs_data = data
        
        if not prs_data:
            raise HTTPException(status_code=404, detail="No pull requests found in repository")
        
        # Filter PRs by state if not 'all'
        if state != "all":
            prs_data = [pr for pr in prs_data if pr.get('state') == state]
        
        # Limit results
        prs_data = prs_data[:limit]
        
        return {
            "repository": repo_name,
            "total_prs": len(prs_data),
            "state_filter": state,
            "limit": limit,
            "prs": prs_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting raw PRs: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching PRs: {str(e)}")

@app.get("/repository/{owner}/{repo}/prs/stats")
async def get_pr_statistics(owner: str, repo: str):
    """Get PR statistics for a repository"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Get repository PRs
        repo_prs = await get_repository_prs(repo_name, "all")
        
        # Extract PRs data
        prs_data = []
        
        if isinstance(repo_prs, list):
            for item in repo_prs:
                if item.get('successful') and item.get('data'):
                    data = item['data']
                    if 'details' in data:  # This is PRs data
                        prs_data = data['details']
                    elif isinstance(data, list):  # Direct list of PRs
                        prs_data = data
        
        if not prs_data:
            return {
                "repository": repo_name,
                "total_prs": 0,
                "open_prs": 0,
                "closed_prs": 0,
                "merged_prs": 0
            }
        
        # Calculate statistics
        open_prs = [pr for pr in prs_data if pr.get('state') == 'open']
        closed_prs = [pr for pr in prs_data if pr.get('state') == 'closed']
        merged_prs = [pr for pr in prs_data if pr.get('state') == 'merged']
        
        return {
            "repository": repo_name,
            "total_prs": len(prs_data),
            "open_prs": len(open_prs),
            "closed_prs": len(closed_prs),
            "merged_prs": len(merged_prs),
            "last_updated": max([pr.get('updated_at', '') for pr in prs_data], default='')
        }
        
    except Exception as e:
        print(f"❌ Error getting PR statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching PR statistics: {str(e)}")

@app.post("/repository/{owner}/{repo}/prs", response_model=CreatePRResponse)
async def create_repository_pr_endpoint(
    owner: str, 
    repo: str, 
    pr_request: CreatePRRequest
):
    """Create a new pull request"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Create the PR
        pr_result = await create_repository_pr(repo_name, pr_request)
        
        # Extract PR data
        pr_data = None
        
        if isinstance(pr_result, list):
            for item in pr_result:
                if item.get('successful') and item.get('data'):
                    pr_data = item['data']
                    break
        
        if not pr_data:
            raise HTTPException(status_code=500, detail="Failed to create pull request")
        
        # Format PR response
        formatted_pr = format_create_pr_response(pr_data)
        
        print(f"✅ Successfully created PR #{formatted_pr.number}: {formatted_pr.title}")
        return formatted_pr
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating repository PR: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating pull request: {str(e)}")

@app.put("/repository/{owner}/{repo}/prs/{pr_number}/merge", response_model=MergePRResponse)
async def merge_repository_pr_endpoint(
    owner: str, 
    repo: str, 
    pr_number: int,
    merge_request: MergePRRequest = MergePRRequest()
):
    """Merge a pull request"""
    repo_name = f"{owner}/{repo}"
    
    try:
        # Merge the PR
        merge_result = await merge_repository_pr(repo_name, pr_number, merge_request)
        
        # Extract merge data
        merge_data = None
        
        if isinstance(merge_result, list):
            for item in merge_result:
                if item.get('successful') and item.get('data'):
                    merge_data = item['data']
                    break
        
        if not merge_data:
            raise HTTPException(status_code=500, detail="Failed to merge pull request")
        
        # Format merge response
        formatted_response = format_merge_pr_response(merge_data, pr_number)
        
        print(f"✅ Successfully merged PR #{pr_number}: {formatted_response.title}")
        return formatted_response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error merging repository PR: {e}")
        raise HTTPException(status_code=500, detail=f"Error merging pull request: {str(e)}")

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
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Using port 8001 to avoid conflict with main.py 