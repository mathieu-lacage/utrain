import json
import time
import uuid

import fastapi
import pydantic
import sqlalchemy

from .. import ctx, models

router = fastapi.APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectIn(pydantic.BaseModel):
    name: str
    preset_name: str
    config: dict[str, object] = {}


class ProjectOut(pydantic.BaseModel):
    id: str
    name: str
    preset_name: str
    config: dict[str, object]
    created_at: float


def _row_to_out(
    row: tuple[str, str, str, str, float],
) -> ProjectOut:
    return ProjectOut(
        id=row[0],
        name=row[1],
        preset_name=row[2],
        config=json.loads(row[3]),
        created_at=row[4],
    )


_PROJECT_COLS = [
    models.projects.c.id,
    models.projects.c.name,
    models.projects.c.preset_name,
    models.projects.c.config,
    models.projects.c.created_at,
]


@router.get("")
def list_projects() -> list[ProjectOut]:
    rows = (
        ctx.db.get()
        .execute(sqlalchemy.select(*_PROJECT_COLS).order_by(models.projects.c.created_at.desc()))
        .fetchall()
    )
    return [_row_to_out(r) for r in rows]  # type: ignore[arg-type]


@router.post("", status_code=201)
def create_project(body: ProjectIn) -> ProjectOut:
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
    return ProjectOut(
        id=project_id,
        name=body.name,
        preset_name=body.preset_name,
        config=body.config,
        created_at=now,
    )


@router.get("/{project_id}")
def get_project(project_id: str) -> ProjectOut:
    row = (
        ctx.db.get()
        .execute(sqlalchemy.select(*_PROJECT_COLS).where(models.projects.c.id == project_id))
        .fetchone()
    )
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")
    return _row_to_out(row)  # type: ignore[arg-type]


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    existing = ctx.db.get().execute(
        sqlalchemy.select(models.projects.c.id).where(models.projects.c.id == project_id)
    ).scalar_one_or_none()
    if existing is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")
    ctx.db.get().execute(
        sqlalchemy.delete(models.projects).where(models.projects.c.id == project_id)
    )
