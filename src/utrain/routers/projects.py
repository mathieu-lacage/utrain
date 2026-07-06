import json
import time
import uuid

import fastapi
import pydantic
import sqlalchemy
import sqlalchemy.engine

from .. import ctx, models

router = fastapi.APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreateRequest(pydantic.BaseModel):
    name: str
    preset_name: str
    config: dict[str, object] = {}


class ProjectResponse(pydantic.BaseModel):
    id: str
    name: str
    preset_name: str
    config: dict[str, object]
    created_at: float


def _mapping_to_response(row: sqlalchemy.engine.RowMapping) -> ProjectResponse:
    return ProjectResponse(
        id=str(row["id"]),
        name=str(row["name"]),
        preset_name=str(row["preset_name"]),
        config=json.loads(str(row["config"])),
        created_at=float(row["created_at"]),
    )


_PROJECT_COLS = [
    models.projects.c.id,
    models.projects.c.name,
    models.projects.c.preset_name,
    models.projects.c.config,
    models.projects.c.created_at,
]


@router.get("")
def list_projects() -> list[ProjectResponse]:
    rows = (
        ctx.db.get()
        .execute(sqlalchemy.select(*_PROJECT_COLS).order_by(models.projects.c.created_at.desc()))
        .mappings()
        .fetchall()
    )
    return [_mapping_to_response(r) for r in rows]


@router.post("", status_code=201)
def create_project(body: ProjectCreateRequest) -> ProjectResponse:
    project_id = str(uuid.uuid4())
    now = time.time()
    ctx.db.get().execute(
        sqlalchemy.insert(models.projects).values(
            id=project_id,
            name=body.name,
            preset_name=body.preset_name,
            config=json.dumps(body.config),
            created_at=now,
        )
    )
    return ProjectResponse(
        id=project_id,
        name=body.name,
        preset_name=body.preset_name,
        config=body.config,
        created_at=now,
    )


@router.get("/{project_id}")
def get_project(project_id: str) -> ProjectResponse:
    row = (
        ctx.db.get()
        .execute(sqlalchemy.select(*_PROJECT_COLS).where(models.projects.c.id == project_id))
        .mappings()
        .fetchone()
    )
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")
    return _mapping_to_response(row)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    existing = (
        ctx.db.get()
        .execute(sqlalchemy.select(models.projects.c.id).where(models.projects.c.id == project_id))
        .scalar_one_or_none()
    )
    if existing is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")
    ctx.db.get().execute(
        sqlalchemy.delete(models.projects).where(models.projects.c.id == project_id)
    )
