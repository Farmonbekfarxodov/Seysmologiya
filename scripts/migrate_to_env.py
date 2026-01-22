#!/usr/bin/env python3
"""
Migration script to convert user_info.json to .env format
"""
import json
import os
import secrets
import string

def generate_secret_key(length=50):
    """Generate a random secret key for Django"""
    chars = string.ascii_letters + string.digits + '!@#$%^&*(-_=+)'
    return ''.join(secrets.choice(chars) for _ in range(length))

def migrate_config():
    """Migrate user_info.json to .env format"""
    try:
        with open('user_info.json', 'r') as f:
            config = json.load(f)
        
        # Generate a proper random SECRET_KEY
        new_secret_key = generate_secret_key()
        
        env_content = f"""# Generated from user_info.json
# Django Settings
SECRET_KEY={new_secret_key}
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
        
        print("✅ Migration successful!")
        print("⚠️  IMPORTANT: A new SECRET_KEY has been generated automatically.")
        print("   For production:")
        print("   1. Set DEBUG=False")
        print("   2. Update ALLOWED_HOSTS with your domain")
        print("   3. Review and update CORS_ALLOWED_ORIGINS")
        print("   4. Use strong database passwords")
        
    except FileNotFoundError:
        print("❌ user_info.json not found")
        print("Creating .env.example as template...")
        print("Please copy .env.example to .env and fill in your values")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    migrate_config()
