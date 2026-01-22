#!/usr/bin/env python3
"""
Migration script to convert user_info.json to .env format
"""
import json
import os

def migrate_config():
    """Migrate user_info.json to .env format"""
    try:
        with open('user_info.json', 'r') as f:
            config = json.load(f)
        
        env_content = f"""# Generated from user_info.json
# Django Settings
SECRET_KEY=django-insecure-CHANGE-THIS-IMMEDIATELY
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DB_NAME={config.get('db', '')}
DB_USER={config.get('user', '')}
DB_PASSWORD={config.get('psw', '')}
DB_HOST={config.get('ip', 'localhost')}
DB_PORT=3306

# CORS Settings
CORS_ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
        """.strip()
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Migration successful! Don't forget to:")
        print("1. Generate a new SECRET_KEY")
        print("   Run: python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'")
        print("2. Set DEBUG=False for production")
        print("3. Update ALLOWED_HOSTS with your domain")
        print("4. Review and update CORS_ALLOWED_ORIGINS")
        
    except FileNotFoundError:
        print("❌ user_info.json not found")
        print("Creating .env.example as template...")
        print("Please copy .env.example to .env and fill in your values")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    migrate_config()
