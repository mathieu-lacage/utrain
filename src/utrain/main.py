import collections.abc
import contextlib
import pathlib

import fastapi
import fastapi.middleware.base
import fastapi.staticfiles
import sqlalchemy
import sqlalchemy.orm
import uvicorn

from . import config, ctx, models, routers


def _create_engine(settings: config.Settings) -> sqlalchemy.Engine:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )
    models.metadata.create_all(engine)
    return engine


class SettingsMiddleware(fastapi.middleware.base.BaseHTTPMiddleware):
    def __init__(self, app: fastapi.FastAPI, settings: config.Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self,
        request: fastapi.Request,
        call_next: fastapi.middleware.base.RequestResponseEndpoint,
    ) -> fastapi.Response:
        with ctx.settings(self._settings):
            return await call_next(request)


class DbMiddleware(fastapi.middleware.base.BaseHTTPMiddleware):
    def __init__(self, app: fastapi.FastAPI, engine: sqlalchemy.Engine) -> None:
        super().__init__(app)
        self._engine = engine

    async def dispatch(
        self,
        request: fastapi.Request,
        call_next: fastapi.middleware.base.RequestResponseEndpoint,
    ) -> fastapi.Response:
        with sqlalchemy.orm.Session(self._engine) as session:
            with ctx.db(session):
                response = await call_next(request)
                session.commit()
                return response


def create_app(settings: config.Settings) -> fastapi.FastAPI:
    engine = _create_engine(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_app: fastapi.FastAPI) -> collections.abc.AsyncIterator[None]:
        _app.state.engine = engine
        yield

    app = fastapi.FastAPI(title="utrain", lifespan=lifespan)
    app.add_middleware(DbMiddleware, engine=engine)
    app.add_middleware(SettingsMiddleware, settings=settings)

    app.include_router(routers.presets.router)
    app.include_router(routers.projects.router)
    app.include_router(routers.runs.router)

    frontend_dist = pathlib.Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/", fastapi.staticfiles.StaticFiles(directory=frontend_dist, html=True), name="static"
        )

    return app


def cli() -> None:
    settings = config.Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
