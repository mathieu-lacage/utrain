import fastapi

from .. import container, ctx

router = fastapi.APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("")
def list_presets() -> list[str]:
    return list(ctx.settings.get().presets.keys())


@router.get("/{name}/describe")
def get_describe(name: str) -> container.schema.DescribeOutput:
    settings = ctx.settings.get()
    image = settings.presets.get(name)
    if image is None:
        raise fastapi.HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return container.enroot.describe(image)


@router.post("/{name}/check-compat")
def post_check_compat(name: str) -> container.schema.CompatResult:
    settings = ctx.settings.get()
    image = settings.presets.get(name)
    if image is None:
        raise fastapi.HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return container.enroot.check_compat(image)
