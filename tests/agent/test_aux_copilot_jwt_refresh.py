"""Regression test: auxiliary path must refresh stale Copilot IDE JWT.

Bug: Aux clients (title generation, compression, vision) cached an OpenAI
client with the Copilot IDE JWT baked into its Authorization header. The JWT
expires every ~30 min. The aux client cache never re-consulted the JWT
expiry, so aux calls reused stale tokens and hit:
    HTTP 401: IDE token expired: unauthorized: token expired

Fix:
- Proactive: _get_cached_client() validates JWT freshness for copilot via
  is_copilot_jwt_fresh() and evicts stale aux clients.
- Reactive: _refresh_provider_credentials("copilot") force-refreshes the JWT
  and evicts aux clients on a 401.
"""
import time
from unittest.mock import MagicMock, patch

import pytest


def test_is_copilot_jwt_fresh_returns_false_for_unknown_token():
    from hermes_cli.copilot_auth import is_copilot_jwt_fresh

    assert is_copilot_jwt_fresh("never-seen-this-token") is False


def test_is_copilot_jwt_fresh_respects_margin():
    from hermes_cli.copilot_auth import _jwt_cache, _jwt_cache_lock, is_copilot_jwt_fresh

    fp = "test-fp-fresh"
    fresh_token = "fresh-jwt"
    stale_token = "stale-jwt"

    with _jwt_cache_lock:
        _jwt_cache[fp] = (fresh_token, time.time() + 3600)
        _jwt_cache[fp + "x"] = (stale_token, time.time() + 30)  # within 120s margin
    try:
        assert is_copilot_jwt_fresh(fresh_token) is True
        assert is_copilot_jwt_fresh(stale_token) is False
    finally:
        with _jwt_cache_lock:
            _jwt_cache.pop(fp, None)
            _jwt_cache.pop(fp + "x", None)


def test_get_cached_client_evicts_stale_copilot_jwt():
    """Cache-hit path must drop a stale-JWT aux client before reuse."""
    from agent import auxiliary_client as ac

    stale_jwt = "tid=stale;exp=000"
    fresh_jwt = "tid=fresh;exp=999"

    fake_stale_client = MagicMock()
    fake_stale_client.api_key = stale_jwt
    fake_stale_client.base_url = "https://api.githubcopilot.com/"

    fake_fresh_client = MagicMock()
    fake_fresh_client.api_key = fresh_jwt
    fake_fresh_client.base_url = "https://api.githubcopilot.com/"

    cache_key = ac._client_cache_key(
        "copilot", async_mode=False, api_key="", base_url="", api_mode="",
    )
    with ac._client_cache_lock:
        ac._client_cache[cache_key] = (fake_stale_client, "claude-opus-4.6", None)

    try:
        with patch.object(ac, "resolve_provider_client",
                          return_value=(fake_fresh_client, "claude-opus-4.6")) as rpc, \
             patch("hermes_cli.copilot_auth.is_copilot_jwt_fresh", return_value=False):
            client, _model = ac._get_cached_client("copilot")
        assert client is fake_fresh_client, "stale-JWT client should have been evicted"
        rpc.assert_called_once()
    finally:
        with ac._client_cache_lock:
            ac._client_cache.pop(cache_key, None)


def test_get_cached_client_evicts_stale_copilot_jwt_when_provider_is_auto():
    """Regression: provider='auto' aux path (e.g. title generation) reuses
    the user's main runtime, which for Copilot users points at
    api.githubcopilot.com. The freshness check must fire on URL match, not
    on the resolved provider name — otherwise stale JWTs surface as
    'HTTP 401: IDE token expired'.
    """
    from agent import auxiliary_client as ac

    stale_jwt = "tid=stale;exp=000"
    fresh_jwt = "tid=fresh;exp=999"

    fake_stale_client = MagicMock()
    fake_stale_client.api_key = stale_jwt
    fake_stale_client.base_url = "https://api.githubcopilot.com/"

    fake_fresh_client = MagicMock()
    fake_fresh_client.api_key = fresh_jwt
    fake_fresh_client.base_url = "https://api.githubcopilot.com/"

    main_runtime = {
        "provider": "copilot",
        "model": "claude-opus-4.6",
        "base_url": "https://api.githubcopilot.com",
        "api_key": stale_jwt,
        "api_mode": "",
    }
    cache_key = ac._client_cache_key(
        "auto",
        async_mode=False,
        api_key="",
        base_url="",
        api_mode="",
        main_runtime=main_runtime,
    )
    with ac._client_cache_lock:
        ac._client_cache[cache_key] = (fake_stale_client, "claude-opus-4.6", None)

    try:
        with patch.object(ac, "resolve_provider_client",
                          return_value=(fake_fresh_client, "claude-opus-4.6")) as rpc, \
             patch("hermes_cli.copilot_auth.is_copilot_jwt_fresh", return_value=False):
            client, _model = ac._get_cached_client("auto", main_runtime=main_runtime)
        assert client is fake_fresh_client, (
            "auto-provider Copilot client with stale JWT should have been evicted"
        )
        rpc.assert_called_once()
    finally:
        with ac._client_cache_lock:
            ac._client_cache.pop(cache_key, None)


def test_get_cached_client_skips_jwt_check_for_non_copilot_auto():
    """auto-provider clients pointing at non-Copilot URLs must NOT trigger
    the Copilot JWT freshness check (no false-positive evictions for
    OpenRouter / Anthropic / etc.).
    """
    from agent import auxiliary_client as ac

    fake_client = MagicMock()
    fake_client.api_key = "sk-or-v1-whatever"
    fake_client.base_url = "https://openrouter.ai/api/v1/"

    main_runtime = {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-v1-whatever",
        "api_mode": "",
    }
    cache_key = ac._client_cache_key(
        "auto",
        async_mode=False,
        api_key="",
        base_url="",
        api_mode="",
        main_runtime=main_runtime,
    )
    with ac._client_cache_lock:
        ac._client_cache[cache_key] = (fake_client, "anthropic/claude-opus-4", None)

    try:
        with patch.object(ac, "resolve_provider_client") as rpc, \
             patch("hermes_cli.copilot_auth.is_copilot_jwt_fresh",
                   return_value=False) as fresh_check:
            client, _model = ac._get_cached_client("auto", main_runtime=main_runtime)
        assert client is fake_client, "non-Copilot auto client must not be evicted"
        rpc.assert_not_called()
        fresh_check.assert_not_called()
    finally:
        with ac._client_cache_lock:
            ac._client_cache.pop(cache_key, None)


def test_refresh_provider_credentials_copilot_branch():
    """_refresh_provider_credentials must handle 'copilot' and force-refresh."""
    from agent import auxiliary_client as ac

    with patch("hermes_cli.copilot_auth.resolve_copilot_token",
               return_value=("ghu_fakeraw", "env")), \
         patch("hermes_cli.copilot_auth.get_copilot_api_token",
               return_value="tid=fresh;exp=999") as gca, \
         patch.object(ac, "_evict_cached_clients") as evict:
        ok = ac._refresh_provider_credentials("copilot")
    assert ok is True
    gca.assert_called_once()
    # Must request a force_refresh to bypass the JWT cache after a 401
    assert gca.call_args.kwargs.get("force_refresh") is True
    evict.assert_called_once_with("copilot")


def test_exchange_copilot_token_force_refresh_bypasses_cache():
    from hermes_cli import copilot_auth

    raw = "ghu_test_force_refresh"
    fp = copilot_auth._token_fingerprint(raw)
    with copilot_auth._jwt_cache_lock:
        copilot_auth._jwt_cache[fp] = ("cached-tok", time.time() + 3600)

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"token":"new-tok","expires_at":' + str(int(time.time()) + 1800).encode() + b'}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None

    try:
        # Without force_refresh: returns cached
        tok, _ = copilot_auth.exchange_copilot_token(raw)
        assert tok == "cached-tok"

        # With force_refresh: hits the network
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tok, _ = copilot_auth.exchange_copilot_token(raw, force_refresh=True)
        assert tok == "new-tok"
    finally:
        with copilot_auth._jwt_cache_lock:
            copilot_auth._jwt_cache.pop(fp, None)
