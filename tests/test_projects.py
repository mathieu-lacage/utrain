import unittest.mock

import fastapi.testclient
import pytest

import utrain.config
import utrain.container.enroot
import utrain.main


@pytest.fixture()
def settings(tmp_path: pytest.TempPathFactory) -> utrain.config.Settings:
    return utrain.config.Settings(data_dir=tmp_path)


@pytest.fixture()
def client(settings: utrain.config.Settings) -> fastapi.testclient.TestClient:
    app = utrain.main.create_app(settings)
    with (
        fastapi.testclient.TestClient(app) as c,
        unittest.mock.patch.object(
            utrain.container.enroot,
            "list_presets",
            return_value={"fake": "localhost/fake:utrain"},
        ),
    ):
        yield c


def test_list_presets(client: fastapi.testclient.TestClient) -> None:
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    assert resp.json() == ["fake"]


def test_create_and_get_project(client: fastapi.testclient.TestClient) -> None:
    resp = client.post("/api/projects", json={"name": "test", "preset_name": "fake", "config": {}})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test"
    project_id = data["id"]

    resp2 = client.get(f"/api/projects/{project_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == project_id


def test_delete_project(client: fastapi.testclient.TestClient) -> None:
    resp = client.post("/api/projects", json={"name": "del", "preset_name": "fake", "config": {}})
    pid = resp.json()["id"]
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_list_projects_empty(client: fastapi.testclient.TestClient) -> None:
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []
