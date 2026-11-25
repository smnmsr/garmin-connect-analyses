import fcntl
import json
import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from garmin_fit_sdk import Decoder, Stream
from garmin_personal_api import GarminPersonalApiClient, PersonalApiAuthenticator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA = Path(__file__).resolve().parents[2] / "data"
PAGE_SIZE = 100
BACKOFF_SECONDS = 5  # retry delays grow 0, 10, 20, 40, 80 s


def _session():
    retry = Retry(
        total=5,
        backoff_factor=BACKOFF_SECONDS,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
        raise_on_status=False,  # client raises HttpError with the final status
    )
    session = requests.Session()
    for scheme in ("http://", "https://"):
        session.mount(scheme, HTTPAdapter(max_retries=retry))
    return session


class GarminDownloader:
    """Authenticated, rate-limited, cached access to Garmin activity files."""

    def __init__(self):
        DATA.mkdir(exist_ok=True)
        self._lock = open(DATA / ".lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another download run is active") from None
        load_dotenv()
        self._session = _session()
        token = PersonalApiAuthenticator(session=self._session).obtain_token(os.environ["GARMIN_CONNECT_API_KEY"])
        self._client = GarminPersonalApiClient(token.access_token, session=self._session)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._session.close()
        self._lock.close()  # closing the file releases the lock

    def list_activities(self):
        """Fetch the full activity index, save it under data/, return it."""
        activities, start = [], 0
        while page := self._client.search_activities(start=start, limit=PAGE_SIZE):
            activities += page
            start += PAGE_SIZE
        DATA.mkdir(exist_ok=True)
        (DATA / "activities.json").write_text(json.dumps(activities))
        return activities

    def latest_activity_id(self):
        return self._client.search_activities(limit=1)[0]["activityId"]

    def download_activity(self, activity_id):
        """Return path of the cached FIT ZIP, downloading it only if missing."""
        path = DATA / "activities" / f"{activity_id}.zip"
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(self._client.download_activity(activity_id))
        if not _is_fit_zip(tmp):
            tmp.unlink()
            raise ValueError(f"activity {activity_id}: download is not a valid FIT ZIP")
        os.replace(tmp, path)
        return path


def _is_fit_zip(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None and any(n.endswith(".fit") for n in zf.namelist())
    except zipfile.BadZipFile:
        return False


def decode_zip(path):
    """Decode the FIT file inside a cached activity ZIP."""
    with zipfile.ZipFile(path) as zf:
        fit_bytes = zf.read(next(n for n in zf.namelist() if n.endswith(".fit")))
    messages, errors = Decoder(Stream.from_byte_array(bytearray(fit_bytes))).read()
    if errors:  # not an assert: asserts are stripped under python -O
        raise ValueError(f"FIT decode errors: {errors}")
    return messages


def fit_messages(activity_id=None):
    """Return (activity_id, decoded FIT messages). Newest activity if no id given."""
    with GarminDownloader() as garmin:
        if activity_id is None:
            activity_id = garmin.latest_activity_id()
        path = garmin.download_activity(activity_id)
    return activity_id, decode_zip(path)
