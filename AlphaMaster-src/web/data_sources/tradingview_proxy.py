"""Best-effort proxy wiring for tvDatafeed on Windows."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    scheme: str = "http"

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


_APPLIED: ProxyConfig | None = None


def _parse_proxy_server(raw: str) -> ProxyConfig | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if "=" in text:
        parts = {}
        for item in text.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                parts[k.strip().lower()] = v.strip()
        text = parts.get("https") or parts.get("http") or parts.get("socks") or ""
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text)
    m = re.match(r"^\[?([^:\]]+)\]?:([0-9]+)$", text)
    if not m:
        return None
    try:
        return ProxyConfig(host=m.group(1), port=int(m.group(2)))
    except ValueError:
        return None


def detect_windows_proxy() -> ProxyConfig | None:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not int(enabled):
                return None
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            return _parse_proxy_server(str(server))
    except Exception:
        return None


def apply_tradingview_proxy() -> ProxyConfig | None:
    global _APPLIED
    proxy = detect_windows_proxy()
    if proxy is None:
        return None
    if _APPLIED == proxy:
        return proxy

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = proxy.url
        os.environ[key.lower()] = proxy.url

    try:
        import tvDatafeed.main as tv_main

        original = getattr(tv_main, "_alphamaster_original_create_connection", None)
        if original is None:
            original = tv_main.create_connection
            tv_main._alphamaster_original_create_connection = original

        def proxied_create_connection(url, *args, **kwargs):
            kwargs.setdefault("http_proxy_host", proxy.host)
            kwargs.setdefault("http_proxy_port", proxy.port)
            kwargs.setdefault("proxy_type", "http")
            return original(url, *args, **kwargs)

        tv_main.create_connection = proxied_create_connection
    except Exception:
        pass

    _APPLIED = proxy
    return proxy
