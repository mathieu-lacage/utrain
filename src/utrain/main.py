import collections.abc
import contextlib
import pathlib

import fastapi
import sqlalchemy
import sqlalchemy.orm
import starlette.middleware.base
import starlette.requests
import starlette.responses
import starlette.types
import uvicorn

from . import config, ctx, models, routers


def _create_engine(settings: config.Settings) -> sqlalchemy.Engine:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )
    models.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine: sqlalchemy.Engine) -> None:
    inspector = sqlalchemy.inspect(engine)
    run_columns = {c["name"] for c in inspector.get_columns("runs")}
    if "pid" not in run_columns:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN pid INTEGER")


class SettingsMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    def __init__(self, app: starlette.types.ASGIApp, settings: config.Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self,
        request: starlette.requests.Request,
        call_next: starlette.middleware.base.RequestResponseEndpoint,
    ) -> starlette.responses.Response:
        with ctx.settings(self._settings):
            return await call_next(request)


class DbMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    def __init__(self, app: starlette.types.ASGIApp, engine: sqlalchemy.Engine) -> None:
        super().__init__(app)
        self._engine = engine

    async def dispatch(
        self,
        request: starlette.requests.Request,
        call_next: starlette.middleware.base.RequestResponseEndpoint,
    ) -> starlette.responses.Response:
        with sqlalchemy.orm.Session(self._engine) as session:
            with ctx.db(session):
                response = await call_next(request)
                session.commit()
                return response


def create_app(settings: config.Settings) -> fastapi.FastAPI:
    engine = _create_engine(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_app: fastapi.FastAPI) -> collections.abc.AsyncGenerator[None, None]:
        _app.state.engine = engine
        routers.runs.reconcile_running_runs(engine)
        yield

    app = fastapi.FastAPI(title="utrain", lifespan=lifespan)
    app.add_middleware(DbMiddleware, engine=engine)
    app.add_middleware(SettingsMiddleware, settings=settings)

    app.include_router(routers.presets.router)
    app.include_router(routers.projects.router)
    app.include_router(routers.runs.router)

    frontend_dist = pathlib.Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        _frontend_dist = frontend_dist.resolve()

        async def _spa_fallback(full_path: str) -> starlette.responses.Response:
            candidate = (_frontend_dist / full_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(_frontend_dist):
                return starlette.responses.FileResponse(candidate)
            return starlette.responses.FileResponse(_frontend_dist / "index.html")

        app.add_api_route("/{full_path:path}", _spa_fallback, include_in_schema=False)

    return app


def cli() -> None:
    settings = config.Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
