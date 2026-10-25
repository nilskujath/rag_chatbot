import pytest
from google.genai import errors

from rag_chatbot.llm import gemini
from rag_chatbot.llm.gemini import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls_by_the_interval():
    clock = FakeClock()
    limiter = RateLimiter(requests_per_minute=6, clock=clock, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert clock.slept == [10.0, 10.0]


def test_rate_limiter_does_not_wait_after_a_pause():
    clock = FakeClock()
    limiter = RateLimiter(requests_per_minute=6, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 60
    limiter.wait()
    assert clock.slept == []


@pytest.fixture()
def no_waits(monkeypatch):
    delays: list[int] = []
    monkeypatch.setattr(gemini.time, "sleep", delays.append)
    monkeypatch.setattr(gemini.limiter, "wait", lambda: None)
    return delays


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": "x", "status": "x"}})


def test_retry_on_503_then_success(no_waits):
    outcomes = iter([api_error(503), api_error(429), "ok"])

    def call():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert gemini._with_retry(call) == "ok"
    assert no_waits == [10, 20]


def test_no_retry_on_other_errors(no_waits):
    calls = []

    def call():
        calls.append(1)
        raise api_error(400)

    with pytest.raises(errors.APIError):
        gemini._with_retry(call)
    assert len(calls) == 1 and no_waits == []


def test_gives_up_after_max_attempts(no_waits, monkeypatch):
    monkeypatch.setattr(gemini.settings, "llm_max_attempts", 3)
    calls = []

    def call():
        calls.append(1)
        raise api_error(503)

    with pytest.raises(errors.APIError):
        gemini._with_retry(call)
    assert len(calls) == 3 and no_waits == [10, 20]


def test_generate_passes_temperature_and_json_mode(monkeypatch, no_waits):
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append((model, config.temperature, config.response_mime_type))
            return type("Response", (), {"text": "ok"})()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gemini, "_client", lambda: FakeClient())
    assert gemini.generate("p", model="m", json_output=True, temperature=0) == "ok"
    assert gemini.generate("p") == "ok"
    assert calls == [
        ("m", 0, "application/json"),
        (gemini.settings.llm_model, None, None),
    ]
