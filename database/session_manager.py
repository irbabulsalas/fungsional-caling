import streamlit as st
import pandas as pd
import pickle
import io
from typing import Optional, Dict, List, Any
from datetime import datetime
from database.db_manager import db_manager, Project, Dataset, Model, AnalysisSession
from database.auth_manager import AuthManager

class SessionManager:
    @staticmethod
    def save_project(name: str, description: str = "", is_public: bool = False) -> Optional[int]:
        """Save current analysis as a project"""
        if not AuthManager.is_authenticated():
            st.error("❌ Please login to save projects")
            return None
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            project = Project(
                name=name,
                description=description,
                owner_id=user_id,
                is_public=is_public
            )
            session.add(project)
            session.commit()
            session.refresh(project)
            
            db_manager.log_audit(
                user_id=user_id,
                action='create_project',
                resource_type='project',
                resource_id=project.id,
                details={'name': name}
            )
            
            return project.id
        except Exception as e:
            session.rollback()
            st.error(f"❌ Error saving project: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def save_dataset(project_id: Optional[int], name: str, df: pd.DataFrame, 
                     file_type: str = 'csv', description: str = "") -> Optional[int]:
        """Save dataset to database"""
        if not AuthManager.is_authenticated():
            return None
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            encrypted_data = db_manager.encrypt_data(csv_data)
            
            dataset = Dataset(
                name=name,
                description=description,
                file_type=file_type,
                file_size=len(csv_data),
                rows=len(df),
                columns=len(df.columns),
                data_encrypted=encrypted_data,
                data_info={
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                owner_id=user_id,
                project_id=project_id
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            
            db_manager.log_audit(
                user_id=user_id,
                action='save_dataset',
                resource_type='dataset',
                resource_id=dataset.id,
                details={'name': name, 'rows': len(df), 'columns': len(df.columns)}
            )
            
            return dataset.id
        except Exception as e:
            session.rollback()
            st.error(f"❌ Error saving dataset: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def load_dataset(dataset_id: int) -> Optional[pd.DataFrame]:
        """Load dataset from database"""
        if not AuthManager.is_authenticated():
            return None
        
        session = db_manager.get_session()
        try:
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                st.error("❌ Dataset not found")
                return None
            
            if dataset.owner_id != AuthManager.get_user_id():
                st.error("❌ You don't have permission to access this dataset")
                return None
            
            decrypted_data = db_manager.decrypt_data(dataset.data_encrypted)
            df = pd.read_csv(io.StringIO(decrypted_data))
            
            db_manager.log_audit(
                user_id=AuthManager.get_user_id(),
                action='load_dataset',
                resource_type='dataset',
                resource_id=dataset_id
            )
            
            return df
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def save_model(project_id: Optional[int], name: str, model_obj: Any, 
                   model_type: str, algorithm: str, metrics: Dict = None,
                   hyperparameters: Dict = None, feature_importance: Dict = None) -> Optional[int]:
        """Save trained model to database"""
        if not AuthManager.is_authenticated():
            return None
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            model_binary = pickle.dumps(model_obj)
            
            model = Model(
                name=name,
                model_type=model_type,
                algorithm=algorithm,
                model_binary=model_binary,
                metrics=metrics or {},
                hyperparameters=hyperparameters or {},
                feature_importance=feature_importance or {},
                owner_id=user_id,
                project_id=project_id
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            
            db_manager.log_audit(
                user_id=user_id,
                action='save_model',
                resource_type='model',
                resource_id=model.id,
                details={'name': name, 'algorithm': algorithm}
            )
            
            return model.id
        except Exception as e:
            session.rollback()
            st.error(f"❌ Error saving model: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def load_model(model_id: int) -> Optional[Any]:
        """Load trained model from database"""
        if not AuthManager.is_authenticated():
            return None
        
        session = db_manager.get_session()
        try:
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                st.error("❌ Model not found")
                return None
            
            if model.owner_id != AuthManager.get_user_id():
                st.error("❌ You don't have permission to access this model")
                return None
            
            model_obj = pickle.loads(model.model_binary)
            
            db_manager.log_audit(
                user_id=AuthManager.get_user_id(),
                action='load_model',
                resource_type='model',
                resource_id=model_id
            )
            
            return model_obj
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def save_analysis_session(project_id: Optional[int], name: str, 
                               session_state: Dict, visualizations: List = None,
                               results: Dict = None) -> Optional[int]:
        """Save complete analysis session"""
        if not AuthManager.is_authenticated():
            return None
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            analysis_session = AnalysisSession(
                name=name,
                session_state=session_state,
                visualizations=visualizations or [],
                results=results or {},
                user_id=user_id,
                project_id=project_id
            )
            session.add(analysis_session)
            session.commit()
            session.refresh(analysis_session)
            
            db_manager.log_audit(
                user_id=user_id,
                action='save_analysis_session',
                resource_type='analysis_session',
                resource_id=analysis_session.id,
                details={'name': name}
            )
            
            return analysis_session.id
        except Exception as e:
            session.rollback()
            st.error(f"❌ Error saving analysis session: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def load_analysis_session(session_id: int) -> Optional[Dict]:
        """Load complete analysis session"""
        if not AuthManager.is_authenticated():
            return None
        
        session = db_manager.get_session()
        try:
            analysis_session = session.query(AnalysisSession).filter(
                AnalysisSession.id == session_id
            ).first()
            
            if not analysis_session:
                st.error("❌ Analysis session not found")
                return None
            
            if analysis_session.user_id != AuthManager.get_user_id():
                st.error("❌ You don't have permission to access this session")
                return None
            
            db_manager.log_audit(
                user_id=AuthManager.get_user_id(),
                action='load_analysis_session',
                resource_type='analysis_session',
                resource_id=session_id
            )
            
            return {
                'name': analysis_session.name,
                'session_state': analysis_session.session_state,
                'visualizations': analysis_session.visualizations,
                'results': analysis_session.results,
                'created_at': analysis_session.created_at,
                'updated_at': analysis_session.updated_at
            }
        except Exception as e:
            st.error(f"❌ Error loading analysis session: {str(e)}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def list_user_projects() -> List[Dict]:
        """List all projects owned by current user"""
        if not AuthManager.is_authenticated():
            return []
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            projects = session.query(Project).filter(Project.owner_id == user_id).all()
            return [
                {
                    'id': p.id,
                    'name': p.name,
                    'description': p.description,
                    'created_at': p.created_at,
                    'updated_at': p.updated_at
                }
                for p in projects
            ]
        finally:
            session.close()
    
    @staticmethod
    def list_user_datasets() -> List[Dict]:
        """List all datasets owned by current user"""
        if not AuthManager.is_authenticated():
            return []
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            datasets = session.query(Dataset).filter(Dataset.owner_id == user_id).all()
            return [
                {
                    'id': d.id,
                    'name': d.name,
                    'description': d.description,
                    'rows': d.rows,
                    'columns': d.columns,
                    'file_type': d.file_type,
                    'created_at': d.created_at
                }
                for d in datasets
            ]
        finally:
            session.close()
    
    @staticmethod
    def list_user_models() -> List[Dict]:
        """List all models owned by current user"""
        if not AuthManager.is_authenticated():
            return []
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            models = session.query(Model).filter(Model.owner_id == user_id).all()
            return [
                {
                    'id': m.id,
                    'name': m.name,
                    'model_type': m.model_type,
                    'algorithm': m.algorithm,
                    'metrics': m.metrics,
                    'created_at': m.created_at
                }
                for m in models
            ]
        finally:
            session.close()
    
    @staticmethod
    def list_user_sessions() -> List[Dict]:
        """List all analysis sessions owned by current user"""
        if not AuthManager.is_authenticated():
            return []
        
        user_id = AuthManager.get_user_id()
        session = db_manager.get_session()
        
        try:
            sessions = session.query(AnalysisSession).filter(
                AnalysisSession.user_id == user_id
            ).all()
            return [
                {
                    'id': s.id,
                    'name': s.name,
                    'created_at': s.created_at,
                    'updated_at': s.updated_at
                }
                for s in sessions
            ]
        finally:
            session.close()

session_manager = SessionManager()
