"""
Department Routes
Create, list, update, and delete departments.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User
from app.models.department import Department
from app.auth.dependencies import require_admin

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


router = APIRouter(prefix="/api/departments", tags=["Departments"])


class DepartmentCreate(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None


class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class DepartmentResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj):
        return cls(
            id=str(obj.id),
            name=obj.name,
            slug=obj.slug,
            description=obj.description,
            is_active=obj.is_active,
            created_at=obj.created_at,
        )


@router.get("/", response_model=list[DepartmentResponse])
async def list_departments(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all departments."""
    result = await db.execute(select(Department).order_by(Department.name))
    departments = result.scalars().all()
    return [DepartmentResponse.from_orm(d) for d in departments]


@router.get("/all", response_model=list[DepartmentResponse])
async def list_all_departments(
    db: AsyncSession = Depends(get_db),
):
    """List all active departments (public endpoint for dropdowns)."""
    result = await db.execute(
        select(Department)
        .where(Department.is_active == True)
        .order_by(Department.name)
    )
    departments = result.scalars().all()
    return [DepartmentResponse.from_orm(d) for d in departments]


@router.post("/", response_model=DepartmentResponse, status_code=201)
async def create_department(
    request: DepartmentCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new department."""
    existing = await db.execute(select(Department).where(Department.slug == request.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Department with this slug already exists.")

    existing_name = await db.execute(select(Department).where(Department.name == request.name))
    if existing_name.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Department with this name already exists.")

    dept = Department(
        name=request.name,
        slug=request.slug.lower().strip(),
        description=request.description,
    )
    db.add(dept)
    await db.flush()
    await db.refresh(dept)
    return DepartmentResponse.from_orm(dept)


@router.put("/{dept_id}", response_model=DepartmentResponse)
async def update_department(
    dept_id: str,
    request: DepartmentUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update a department."""
    result = await db.execute(select(Department).where(Department.id == dept_id))
    dept = result.scalar_one_or_none()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")

    if request.name is not None:
        dept.name = request.name
    if request.description is not None:
        dept.description = request.description
    if request.is_active is not None:
        dept.is_active = request.is_active

    await db.flush()
    await db.refresh(dept)
    return DepartmentResponse.from_orm(dept)


@router.delete("/{dept_id}")
async def delete_department(
    dept_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a department."""
    result = await db.execute(select(Department).where(Department.id == dept_id))
    dept = result.scalar_one_or_none()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")

    await db.delete(dept)
    return {"message": f"Department '{dept.name}' deleted."}
