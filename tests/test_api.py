"""HTTP 接口测试：标准库服务器与 Api.dispatch 一致性。"""

import json
import urllib.request

from rag_eval.api import Api
from rag_eval.dashboard import serve_in_thread
from rag_eval.dataset import load_cases
from rag_eval.runner import RunOptions, run_evaluation
from tests.helpers import temp_config, tempdir

_CACHE: dict = {}


def _api() -> Api:
    if "api" not in _CACHE:
        tmp = tempdir()
        cfg = temp_config(tmp)
        cases = load_cases("data/testsets/core.jsonl")[:2]
        run = run_evaluation(cfg, cases, RunOptions(name="api-run", dataset="core", persist=True))
        _CACHE.update({"cfg": cfg, "run": run, "api": Api(cfg), "tmp": tmp})
    return _CACHE["api"]


def _json(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_health_endpoint():
    payload = _json(_api().dispatch("GET", "/api/health"))
    assert payload["status"] == "ok"
    assert payload["version"]
    assert "environment" in payload


def test_index_and_static_assets():
    api = _api()
    page = api.dispatch("GET", "/")
    assert page.status == 200
    assert b"<!DOCTYPE html>" in page.body
    css = api.dispatch("GET", "/static/styles.css")
    assert css.status == 200 and b":root" in css.body
    js = api.dispatch("GET", "/static/app.js")
    assert js.status == 200 and b"renderOverview" in js.body


def test_static_path_traversal_blocked():
    api = _api()
    assert api.dispatch("GET", "/static/../__init__.py").status in (400, 404)


def test_runs_and_run_detail():
    api = _api()
    run_id = _CACHE["run"].record.run_id
    runs = _json(api.dispatch("GET", "/api/runs"))
    assert any(item["run_id"] == run_id for item in runs["runs"])

    detail = _json(api.dispatch("GET", f"/api/runs/{run_id}"))
    assert detail["run"]["run_id"] == run_id
    assert detail["metrics"]
    assert detail["scorecard"]

    cases = _json(api.dispatch("GET", f"/api/runs/{run_id}/cases"))
    assert cases["count"] >= 1
    case_id = cases["cases"][0]["case_id"]
    one = _json(api.dispatch("GET", f"/api/runs/{run_id}/cases/{case_id}"))
    assert one["case_id"] == case_id


def test_missing_run_returns_404():
    api = _api()
    response = api.dispatch("GET", "/api/runs/does-not-exist")
    assert response.status == 404
    assert "error" in _json(response)


def test_compare_endpoint_requires_two_ids():
    api = _api()
    assert api.dispatch("GET", "/api/compare").status == 400
    response = api.dispatch("GET", "/api/compare?left=a&right=b")
    assert response.status == 404


def test_metrics_and_datasets_endpoints():
    api = _api()
    metrics = _json(api.dispatch("GET", "/api/metrics"))
    assert len(metrics["metrics"]) >= 30
    assert any(item["name"] == "faithfulness" for item in metrics["metrics"])
    datasets = _json(api.dispatch("GET", "/api/datasets"))
    assert datasets["datasets"]
    assert datasets["datasets"][0]["count"] > 0


def test_report_download():
    api = _api()
    run_id = _CACHE["run"].record.run_id
    response = api.dispatch("GET", f"/api/report?run_id={run_id}&format=markdown")
    assert response.status == 200
    assert b"# " in response.body
    assert "attachment" in response.headers.get("Content-Disposition", "")


def test_post_eval_validates_json():
    api = _api()
    response = api.dispatch("POST", "/api/eval", {}, b"not-json")
    assert response.status == 500 or response.status == 400


def test_socket_server_serves_api():
    _api()
    cfg = _CACHE["cfg"]
    httpd, thread, url = serve_in_thread(cfg, port=0)
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        with urllib.request.urlopen(f"{url}/", timeout=10) as response:
            assert b"<!DOCTYPE html>" in response.read()
    finally:
        httpd.shutdown()
        httpd.server_close()
