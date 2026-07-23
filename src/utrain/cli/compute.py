import json
import os
import pathlib
import subprocess


class CpuInfo:
    name: str
    cores: int
    mem_total_gb: float
    mem_available_gb: float

    def __init__(
        self,
        name: str,
        cores: int,
        mem_total_gb: float,
        mem_available_gb: float,
    ) -> None:
        self.name = name
        self.cores = cores
        self.mem_total_gb = mem_total_gb
        self.mem_available_gb = mem_available_gb


class GpuInfo:
    name: str
    power_draw: float
    power_limit: float
    util: int
    mem_used_mb: float
    mem_total_mb: float

    def __init__(
        self,
        name: str,
        power_draw: float,
        power_limit: float,
        util: int,
        mem_used_mb: float,
        mem_total_mb: float,
    ) -> None:
        self.name = name
        self.power_draw = power_draw
        self.power_limit = power_limit
        self.util = util
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb


class ComputeInfo:
    cpu: CpuInfo
    gpus: list[GpuInfo]

    def __init__(self, cpu: CpuInfo, gpus: list[GpuInfo]) -> None:
        self.cpu = cpu
        self.gpus = gpus


def _read_cpu_name() -> str:
    try:
        text = pathlib.Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Unknown CPU"


def _read_meminfo() -> tuple[float, float]:
    try:
        text = pathlib.Path("/proc/meminfo").read_text()
        total_kb = 0
        available_kb = 0
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
        return total_kb / (1024 * 1024), available_kb / (1024 * 1024)
    except OSError:
        return 0.0, 0.0


def _read_gpus() -> list[GpuInfo]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,power.draw,power.limit,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus: list[GpuInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            def _float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return 0.0

            def _int(s: str) -> int:
                try:
                    return int(s)
                except ValueError:
                    return 0

            gpus.append(
                GpuInfo(
                    name=parts[0],
                    power_draw=_float(parts[1]),
                    power_limit=_float(parts[2]),
                    mem_used_mb=_float(parts[3]),
                    mem_total_mb=_float(parts[4]),
                    util=_int(parts[5]),
                )
            )
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def collect_compute() -> ComputeInfo:
    fixture = os.environ.get("UTRAIN_COMPUTE_FIXTURE")
    if fixture:
        data: dict[str, object] = json.loads(pathlib.Path(fixture).read_text())
        cpu_data = data["cpu"]
        assert isinstance(cpu_data, dict)
        cpu = CpuInfo(
            name=str(cpu_data["name"]),
            cores=int(cpu_data["cores"]),  # type: ignore[arg-type]
            mem_total_gb=float(cpu_data["mem_total_gb"]),  # type: ignore[arg-type]
            mem_available_gb=float(cpu_data["mem_available_gb"]),  # type: ignore[arg-type]
        )
        gpus: list[GpuInfo] = []
        for g in data.get("gpus", []):  # type: ignore[union-attr]
            assert isinstance(g, dict)
            gpus.append(
                GpuInfo(
                    name=str(g["name"]),
                    power_draw=float(g["power_draw"]),  # type: ignore[arg-type]
                    power_limit=float(g["power_limit"]),  # type: ignore[arg-type]
                    util=int(g["util"]),  # type: ignore[arg-type]
                    mem_used_mb=float(g["mem_used_mb"]),  # type: ignore[arg-type]
                    mem_total_mb=float(g["mem_total_mb"]),  # type: ignore[arg-type]
                )
            )
        return ComputeInfo(cpu=cpu, gpus=gpus)

    mem_total, mem_available = _read_meminfo()
    cpu = CpuInfo(
        name=_read_cpu_name(),
        cores=os.cpu_count() or 1,
        mem_total_gb=mem_total,
        mem_available_gb=mem_available,
    )
    return ComputeInfo(cpu=cpu, gpus=_read_gpus())
