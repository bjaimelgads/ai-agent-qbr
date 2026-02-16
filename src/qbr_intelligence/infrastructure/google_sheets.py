"""Google Sheets API helpers for linked chart extraction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_TOKEN_URI = "https://oauth2.googleapis.com/token"


@dataclass(frozen=True)
class GoogleSheetsAuth:
    client_id: str | None
    client_secret: str | None
    refresh_token: str | None
    token_path: Path
    token_json: str | None
    service_account_file: str | None
    service_account_json: str | None

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "GoogleSheetsAuth":
        token_path = os.getenv("GOOGLE_SHEETS_TOKEN_PATH") or ".google_sheets_token.json"
        if base_dir is not None and not Path(token_path).is_absolute():
            token_path = str(base_dir / token_path)
        return cls(
            client_id=(os.getenv("CLIENT_ID") or "").strip() or None,
            client_secret=(os.getenv("CLIENT_SECRET") or "").strip() or None,
            refresh_token=(os.getenv("GOOGLE_REFRESH_TOKEN") or "").strip() or None,
            token_path=Path(token_path),
            token_json=(os.getenv("GOOGLE_SHEETS_TOKEN_JSON") or "").strip() or None,
            service_account_file=(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or "").strip()
            or None,
            service_account_json=(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
            or None,
        )


def _load_credentials(auth: GoogleSheetsAuth) -> Credentials | None:
    if auth.service_account_file:
        return ServiceAccountCredentials.from_service_account_file(
            auth.service_account_file, scopes=_SCOPES
        )
    if auth.service_account_json:
        try:
            payload = json.loads(auth.service_account_json)
        except json.JSONDecodeError:
            return None
        return ServiceAccountCredentials.from_service_account_info(payload, scopes=_SCOPES)

    if auth.token_json:
        try:
            payload = json.loads(auth.token_json)
        except json.JSONDecodeError:
            payload = None
        if payload:
            return Credentials.from_authorized_user_info(payload, scopes=_SCOPES)

    if auth.token_path.exists():
        try:
            return Credentials.from_authorized_user_file(str(auth.token_path), scopes=_SCOPES)
        except Exception:
            return None

    if auth.client_id and auth.client_secret and auth.refresh_token:
        creds = Credentials(
            token=None,
            refresh_token=auth.refresh_token,
            token_uri=_TOKEN_URI,
            client_id=auth.client_id,
            client_secret=auth.client_secret,
            scopes=_SCOPES,
        )
        creds.refresh(Request())
        return creds

    if auth.client_id and auth.client_secret:
        config = {
            "installed": {
                "client_id": auth.client_id,
                "client_secret": auth.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": _TOKEN_URI,
            }
        }
        flow = InstalledAppFlow.from_client_config(config, _SCOPES)
        creds = flow.run_local_server(port=0)
        try:
            auth.token_path.write_text(creds.to_json(), encoding="utf-8")
        except Exception:
            pass
        return creds

    return None


class GoogleSheetsClient:
    """Minimal Google Sheets client for spreadsheet/chart metadata and values."""

    def __init__(self, credentials: Credentials) -> None:
        self._service = build(
            "sheets",
            "v4",
            credentials=credentials,
            cache_discovery=False,
        )

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "GoogleSheetsClient | None":
        auth = GoogleSheetsAuth.from_env(base_dir=base_dir)
        creds = _load_credentials(auth)
        if not creds:
            return None
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            try:
                auth.token_path.write_text(creds.to_json(), encoding="utf-8")
            except Exception:
                pass
        return cls(creds)

    def get_spreadsheet(self, spreadsheet_id: str) -> dict:
        return (
            self._service.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, includeGridData=False)
            .execute()
        )

    def get_chart(self, spreadsheet_id: str, chart_id: int) -> dict | None:
        spreadsheet = self.get_spreadsheet(spreadsheet_id)
        for sheet in spreadsheet.get("sheets", []):
            for chart in sheet.get("charts", []) or []:
                if chart.get("chartId") == chart_id:
                    return chart
        return None

    def get_values(self, spreadsheet_id: str, a1_range: str) -> list[list[str]]:
        payload = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=a1_range)
            .execute()
        )
        return payload.get("values", [])
