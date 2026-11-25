import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from garmin_personal_api import HttpError

from garmin_connect_analyses import garmin

DOWNLOAD = "/download-service/files/activity/123"


def fit_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("123_ACTIVITY.fit", b"fake fit payload")
    return buf.getvalue()


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Local stand-in for Garmin. routes[path] is a list of (status, headers, body); last entry repeats."""
    routes, hits = {}, []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            queue = routes[self.path]
            status, headers, body = queue.pop(0) if len(queue) > 1 else queue[0]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("GARMIN_PERSONAL_API_BASE_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("GARMIN_CONNECT_API_KEY", "test-key")
    monkeypatch.setattr(garmin, "DATA", tmp_path)
    monkeypatch.setattr(garmin, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(
        garmin.PersonalApiAuthenticator,
        "obtain_token",
        lambda self, key: SimpleNamespace(access_token="test-token"),
    )
    yield SimpleNamespace(routes=routes, hits=hits)
    httpd.shutdown()


def test_retries_rate_limit_then_succeeds(server):
    server.routes[DOWNLOAD] = [(429, {"Retry-After": "0"}, b""), (200, {}, fit_zip())]
    with garmin.GarminDownloader() as g:
        assert g.download_activity(123).exists()
    assert len(server.hits) == 2


def test_gives_up_after_retry_budget(server):
    server.routes[DOWNLOAD] = [(429, {}, b"slow down")]
    with garmin.GarminDownloader() as g, pytest.raises(HttpError):
        g.download_activity(123)
    assert len(server.hits) == 6  # first try + 5 retries


def test_second_run_is_refused_while_first_holds_lock(server):
    with garmin.GarminDownloader():
        with pytest.raises(RuntimeError, match="another download run"):
            garmin.GarminDownloader()
    with garmin.GarminDownloader():  # lock is released on exit
        pass


def test_cached_download_skips_network(server):
    server.routes[DOWNLOAD] = [(200, {}, fit_zip())]
    with garmin.GarminDownloader() as g:
        first = g.download_activity(123)
        second = g.download_activity(123)
    assert first == second
    assert len(server.hits) == 1


def test_invalid_download_is_not_cached(server, tmp_path):
    server.routes[DOWNLOAD] = [(200, {}, b"not a zip")]
    with garmin.GarminDownloader() as g, pytest.raises(ValueError):
        g.download_activity(123)
    assert list((tmp_path / "activities").iterdir()) == []
