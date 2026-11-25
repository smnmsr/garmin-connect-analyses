import io
import json
import os
import urllib.request
import zipfile

from dotenv import load_dotenv
from garmin_fit_sdk import Decoder, Stream

BASE = "https://personal.connectapi.garmin.com"


def _call(url, headers, method="GET"):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method)) as resp:
        return resp.read()


def _auth():
    load_dotenv()
    api_key = os.environ["GARMIN_CONNECT_API_KEY"]
    token = _call(
        f"{BASE}/personalapitoken-service/personal-api/personal-api-key/token",
        {"Authorization": f"ApiKey {api_key}"},
        method="POST",
    ).decode().strip()
    return {"Authorization": f"Bearer {token}"}


def fit_messages(activity_id=None):
    """Return (activity_id, decoded FIT messages). Newest activity if no id given."""
    auth = _auth()
    if activity_id is None:
        activity_id = json.loads(_call(f"{BASE}/activitylist-service/activities/search/activities?start=0&limit=1", auth))[0]["activityId"]

    archive = _call(f"{BASE}/download-service/files/activity/{activity_id}", auth)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        fit_bytes = zf.read(next(n for n in zf.namelist() if n.endswith(".fit")))

    messages, errors = Decoder(Stream.from_byte_array(bytearray(fit_bytes))).read()
    assert not errors, errors
    return activity_id, messages
