import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, LargeBinary, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import NullPool
import bcrypt
import jwt
from cryptography.fernet import Fernet

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='analyst')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("AnalysisSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=False)
    shared_with = Column(JSON, default=list)
    
    owner = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="project", cascade="all, delete-orphan")
    sessions = relationship("AnalysisSession", back_populates="project", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="project", cascade="all, delete-orphan")
    versions = relationship("ProjectVersion", back_populates="project", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=True)
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    data_encrypted = Column(LargeBinary, nullable=True)
    data_info = Column(JSON, default=dict)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="datasets")
    project = relationship("Project", back_populates="datasets")

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    model_type = Column(String(100), nullable=False)
    algorithm = Column(String(100), nullable=False)
    model_binary = Column(LargeBinary, nullable=True)
    model_path = Column(String(500), nullable=True)
    metrics = Column(JSON, default=dict)
    hyperparameters = Column(JSON, default=dict)
    feature_importance = Column(JSON, default=dict)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="models")
    project = relationship("Project", back_populates="models")

class AnalysisSession(Base):
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    session_state = Column(JSON, default=dict)
    visualizations = Column(JSON, default=list)
    results = Column(JSON, default=dict)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")
    project = relationship("Project", back_populates="sessions")

class Comment(Base):
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    parent_id = Column(Integer, ForeignKey('comments.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="comments")
    replies = relationship("Comment", backref='parent', remote_side=[id])

class ProjectVersion(Base):
    __tablename__ = 'project_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    version_number = Column(Integer, nullable=False)
    changes = Column(JSON, default=dict)
    snapshot = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    project = relationship("Project", back_populates="versions")

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(Integer, nullable=True)
    details = Column(JSON, default=dict)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User", back_populates="audit_logs")

class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(
            self.database_url,
            poolclass=NullPool,
            echo=False
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        key_file = '.encryption_key'
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def create_tables(self):
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        Base.metadata.drop_all(self.engine)
    
    def get_session(self):
        return self.SessionLocal()
    
    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def encrypt_data(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode('utf-8'))
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        return self.cipher.decrypt(encrypted_data).decode('utf-8')
    
    def create_user(self, username: str, email: str, password: str, role: str = 'analyst') -> Optional[User]:
        session = self.get_session()
        try:
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return None
            
            user = User(
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                role=role
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user and self.verify_password(password, user.password_hash):
                user.last_login = datetime.utcnow()
                session.commit()
                session.refresh(user)
                return user
            return None
        finally:
            session.close()
    
    def generate_jwt_token(self, user_id: int, secret_key: str = None) -> str:
        if not secret_key:
            secret_key = os.getenv('JWT_SECRET_KEY', 'default-secret-key-change-in-production')
        
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow().timestamp() + (24 * 60 * 60),
            'iat': datetime.utcnow().timestamp()
        }
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str, secret_key: str = None) -> Optional[Dict]:
        if not secret_key:
            secret_key = os.getenv('JWT_SECRET_KEY', 'default-secret-key-change-in-production')
        
        try:
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def log_audit(self, user_id: int, action: str, resource_type: str, 
                   resource_id: Optional[int] = None, details: Optional[Dict] = None,
                   ip_address: Optional[str] = None):
        session = self.get_session()
        try:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details or {},
                ip_address=ip_address
            )
            session.add(audit_log)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

try:
    db_manager = DatabaseManager()
    db_manager.create_tables()
except Exception as e:
    print(f"Warning: Database initialization failed: {str(e)}")
    print("Database features will not be available")
    db_manager = None
