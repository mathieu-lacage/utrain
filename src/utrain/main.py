import collections.abc
import contextlib
import pathlib

import fastapi
import fastapi.staticfiles
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
    return engine


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
        static = fastapi.staticfiles.StaticFiles(directory=frontend_dist, html=True)

        # StaticFiles asserts scope["type"] == "http" and crashes on WebSocket connections.
        # Vue DevTools browser extension probes ws:// on the page host, so we need to
        # reject those cleanly instead of letting them bubble up as an unhandled exception.
        async def static_http_only(
            scope: starlette.types.Scope,
            receive: starlette.types.Receive,
            send: starlette.types.Send,
        ) -> None:
            if scope["type"] == "websocket":
                await receive()  # consume the connect event
                await send({"type": "websocket.close", "code": 1008})
                return
            await static(scope, receive, send)

        app.mount("/", static_http_only, name="static")

    return app


def cli() -> None:
    settings = config.Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
