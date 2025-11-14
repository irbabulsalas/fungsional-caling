import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import db_manager

def init_database():
    print("🗄️  Initializing database...")
    
    try:
        db_manager.create_tables()
        print("✅ Database tables created successfully!")
        
        session = db_manager.get_session()
        from database.db_manager import User
        
        admin_exists = session.query(User).filter(User.username == 'admin').first()
        
        if not admin_exists:
            admin_user = db_manager.create_user(
                username='admin',
                email='admin@aidata.com',
                password='admin123',
                role='admin'
            )
            print(f"✅ Admin user created: username='admin', password='admin123'")
        else:
            print("ℹ️  Admin user already exists")
        
        session.close()
        print("✅ Database initialization complete!")
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        raise e

if __name__ == "__main__":
    init_database()
