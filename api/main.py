from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
from datetime import datetime

from database.db_manager import db_manager
from database.auth_manager import AuthManager
from database.session_manager import SessionManager

app = FastAPI(
    title="AI Data Analysis Platform API",
    description="REST API for AI-powered data analysis with ML, NLP, and time series capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "analyst"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    is_public: bool = False

class DatasetUpload(BaseModel):
    name: str
    description: Optional[str] = ""
    project_id: Optional[int] = None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token and return user info"""
    token = credentials.credentials
    payload = db_manager.verify_jwt_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return payload

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Data Analysis Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.post("/auth/register", response_model=Dict)
async def register(user: UserCreate):
    """Register a new user"""
    existing_user = db_manager.create_user(
        username=user.username,
        email=user.email,
        password=user.password,
        role=user.role
    )
    
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    
    return {
        "message": "User registered successfully",
        "user_id": existing_user.id,
        "username": existing_user.username
    }

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login and get JWT token"""
    authenticated_user = db_manager.authenticate_user(user.username, user.password)
    
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    token = db_manager.generate_jwt_token(authenticated_user.id)
    
    return TokenResponse(
        access_token=token,
        user_id=authenticated_user.id,
        username=authenticated_user.username,
        role=authenticated_user.role
    )

@app.get("/auth/me")
async def get_current_user(payload: Dict = Depends(verify_token)):
    """Get current user info"""
    session = db_manager.get_session()
    try:
        from database.db_manager import User
        user = session.query(User).filter(User.id == payload['user_id']).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    finally:
        session.close()

@app.post("/projects")
async def create_project(project: ProjectCreate, payload: Dict = Depends(verify_token)):
    """Create a new project"""
    session = db_manager.get_session()
    try:
        from database.db_manager import Project
        
        new_project = Project(
            name=project.name,
            description=project.description,
            owner_id=payload['user_id'],
            is_public=project.is_public
        )
        
        session.add(new_project)
        session.commit()
        session.refresh(new_project)
        
        return {
            "project_id": new_project.id,
            "name": new_project.name,
            "created_at": new_project.created_at.isoformat()
        }
    finally:
        session.close()

@app.get("/projects")
async def list_projects(payload: Dict = Depends(verify_token)):
    """List all projects for current user"""
    session = db_manager.get_session()
    try:
        from database.db_manager import Project
        
        projects = session.query(Project).filter(
            Project.owner_id == payload['user_id']
        ).all()
        
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat()
            }
            for p in projects
        ]
    finally:
        session.close()

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = None,
    description: str = "",
    project_id: int = None,
    payload: Dict = Depends(verify_token)
):
    """Upload a dataset file"""
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        dataset_name = name or file.filename
        
        session = db_manager.get_session()
        from database.db_manager import Dataset
        
        csv_data = df.to_csv(index=False)
        encrypted_data = db_manager.encrypt_data(csv_data)
        
        dataset = Dataset(
            name=dataset_name,
            description=description,
            file_type=file.filename.split('.')[-1],
            file_size=len(contents),
            rows=len(df),
            columns=len(df.columns),
            data_encrypted=encrypted_data,
            data_info={
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            owner_id=payload['user_id'],
            project_id=project_id
        )
        
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        
        session.close()
        
        return {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "rows": dataset.rows,
            "columns": dataset.columns,
            "file_type": dataset.file_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def list_datasets(payload: Dict = Depends(verify_token)):
    """List all datasets for current user"""
    session = db_manager.get_session()
    try:
        from database.db_manager import Dataset
        
        datasets = session.query(Dataset).filter(
            Dataset.owner_id == payload['user_id']
        ).all()
        
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "rows": d.rows,
                "columns": d.columns,
                "file_type": d.file_type,
                "created_at": d.created_at.isoformat()
            }
            for d in datasets
        ]
    finally:
        session.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

@app.get("/stats")
async def get_stats(payload: Dict = Depends(verify_token)):
    """Get user statistics"""
    session = db_manager.get_session()
    try:
        from database.db_manager import Project, Dataset, Model, AnalysisSession
        
        projects_count = session.query(Project).filter(
            Project.owner_id == payload['user_id']
        ).count()
        
        datasets_count = session.query(Dataset).filter(
            Dataset.owner_id == payload['user_id']
        ).count()
        
        models_count = session.query(Model).filter(
            Model.owner_id == payload['user_id']
        ).count()
        
        sessions_count = session.query(AnalysisSession).filter(
            AnalysisSession.user_id == payload['user_id']
        ).count()
        
        return {
            "projects": projects_count,
            "datasets": datasets_count,
            "models": models_count,
            "analysis_sessions": sessions_count
        }
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
