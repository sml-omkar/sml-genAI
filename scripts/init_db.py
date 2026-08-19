"""
Database Initialization Script
Creates tables and seeds the default admin user.
Run once: python scripts/init_db.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.config import get_settings
from app.database import engine, Base, AsyncSessionLocal
from app.models.user import User, RoleType, Department
from app.models.folder import Folder
from app.models.document import Document
from app.auth.jwt import hash_password


settings = get_settings()

# Default admin credentials
DEFAULT_ADMIN_EMAIL = "admin@company.com"
DEFAULT_ADMIN_PASSWORD = "admin123"
DEFAULT_ADMIN_NAME = "Super Admin"


async def init_database():
    """Create all tables and seed default admin user."""
    print("=" * 60)
    print("AI-Bot Database Initialization")
    print("=" * 60)

    # Step 1: Create tables
    print("\n[1/3] Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  Tables created successfully.")

    # Step 2: Check if admin exists
    print("\n[2/3] Checking for default admin user...")
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(User).where(User.email == DEFAULT_ADMIN_EMAIL)
        )
        existing_admin = result.scalar_one_or_none()

        if existing_admin:
            print(f"  Admin user already exists: {DEFAULT_ADMIN_EMAIL}")
        else:
            # Step 3: Create default admin
            print("\n[3/3] Creating default admin user...")
            admin = User(
                email=DEFAULT_ADMIN_EMAIL,
                hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
                full_name=DEFAULT_ADMIN_NAME,
                department=Department.HR,
                role=RoleType.SUPER_ADMIN,
                is_active=True,
            )
            db.add(admin)
            await db.commit()
            print(f"  Admin created: {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD}")

    print("\n" + "=" * 60)
    print("Initialization complete!")
    print(f"\nAdmin Login:")
    print(f"  Email:    {DEFAULT_ADMIN_EMAIL}")
    print(f"  Password: {DEFAULT_ADMIN_PASSWORD}")
    print(f"\nAdmin Portal: http://localhost:{settings.APP_PORT}/admin")
    print(f"API Docs:     http://localhost:{settings.APP_PORT}/docs")
    print("=" * 60)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_database())
