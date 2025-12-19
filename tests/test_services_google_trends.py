import datetime
from types import SimpleNamespace

import pytest

from src.services.google_trends import GoogleTrendsService


class DummyQuery:
    def __init__(self, items):
        self.items = items

    def head(self, n):
        return DummyQuery(self.items[:n])

    def tolist(self):
        return self.items


class DummyTop:
    def __init__(self, items):
        self.query = DummyQuery(items)

    def __getitem__(self, key):
        if key == "query":
            return self.query
        raise KeyError


class DummyTrendReq:
    def __init__(self, related):
        self.related = related
        self.payloads = []

    def build_payload(self, kw_list, timeframe=None):
        self.payloads.append({"kw_list": kw_list, "timeframe": timeframe})

    def related_queries(self):
        return self.related


@pytest.mark.asyncio
async def test_google_trends_cache_miss_fetches_and_saves(monkeypatch, tmp_path):
    related = {"ml": {"top": DummyTop(["Quantum", "AI news"])}}  # casing will be normalized
    trend = DummyTrendReq(related)
    cache_bucket = {}

    monkeypatch.setattr("src.services.google_trends.TrendReq", lambda **_: trend)
    
    async def mock_load(_):
        return cache_bucket.get("data", {})
    
    async def mock_save(_, data):
        cache_bucket.setdefault("data", data)
        
    monkeypatch.setattr("src.services.google_trends.load_cache", mock_load)
    monkeypatch.setattr("src.services.google_trends.save_cache", mock_save)

    svc = GoogleTrendsService()
    topics = await svc.get_trending_topics(["ml"])

    assert set(topics) == {"quantum", "ai news"}
    assert "timestamp" in cache_bucket.get("data", {})


@pytest.mark.asyncio
async def test_google_trends_stale_cache_refetches(monkeypatch):
    old_timestamp = (datetime.datetime.now() - datetime.timedelta(hours=13)).isoformat()
    related = {"ai": {"top": DummyTop(["fresh topic"])}}  # should override stale cache
    trend = DummyTrendReq(related)
    cache_bucket = {"data": {"timestamp": old_timestamp, "topics": ["old"]}}

    monkeypatch.setattr("src.services.google_trends.TrendReq", lambda **_: trend)
    
    async def mock_load(_):
        return cache_bucket["data"]
        
    async def mock_save(_, data):
        cache_bucket["data"] = data

    monkeypatch.setattr("src.services.google_trends.load_cache", mock_load)
    monkeypatch.setattr("src.services.google_trends.save_cache", mock_save)

    svc = GoogleTrendsService()
    topics = await svc.get_trending_topics(["ai"])

    assert topics == ["fresh topic"]
    assert cache_bucket["data"]["topics"] == ["fresh topic"]


@pytest.mark.asyncio
async def test_google_trends_handles_network_failure(monkeypatch):
    class FailingTrendReq:
        def __init__(self, **_):
            pass

        def build_payload(self, *_, **__):
            raise RuntimeError("network down")

    monkeypatch.setattr("src.services.google_trends.TrendReq", FailingTrendReq)
    
    async def mock_load(_):
        return {}
        
    async def mock_save(*_):
        return None
        
    monkeypatch.setattr("src.services.google_trends.load_cache", mock_load)
    monkeypatch.setattr("src.services.google_trends.save_cache", mock_save)

    svc = GoogleTrendsService()
    fallback = await svc.get_trending_topics(["custom"])

    assert fallback == ["custom"]
