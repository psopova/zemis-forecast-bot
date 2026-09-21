"""Not forecasting the same question twice.

The tournament rules ask for one forecast per question in the bot-only
tournaments. The bot has no memory between runs, and a watcher is restarted
many times inside a question's three hour window, so the only workable source
of truth is the server's own record of what this bot has submitted.

That record only appears in the response when the request asks for it. It did
not, for a while, and the consequence was invisible: every question came back
looking un-forecast, and a second watcher re-forecast a MiniBench question that
had already been answered twenty minutes earlier.
"""

import pytest

from bot import runner
from bot.client import MissingForecastHistory, already_forecast


def test_the_list_request_asks_for_the_forecast_history():
    import inspect

    from bot.client import MetaculusClient

    source = inspect.getsource(MetaculusClient.iter_posts)
    assert '"with_cp"' in source and '"true"' in source, (
        "without with_cp the server omits my_forecasts and the bot silently "
        "re-forecasts the entire tournament"
    )


def test_a_response_without_the_history_is_an_error_not_a_no():
    with pytest.raises(MissingForecastHistory):
        already_forecast({"id": 42, "type": "binary"})


def test_an_empty_history_means_not_yet_forecast():
    q = {"id": 42, "my_forecasts": {"history": [], "latest": None}}
    assert already_forecast(q) is False


def test_a_forecast_in_the_history_counts():
    q = {"id": 42, "my_forecasts": {"history": [{"question_id": 42}], "latest": None}}
    assert already_forecast(q) is True


@pytest.mark.parametrize(
    "latest",
    [
        {"probability_yes": 0.59},
        {"continuous_cdf": [0.0, 0.5, 1.0]},
        {"forecast_values": [0.4, 0.6]},
    ],
)
def test_a_latest_forecast_counts_even_with_no_history(latest):
    q = {"id": 42, "my_forecasts": {"history": [], "latest": latest}}
    assert already_forecast(q) is True


class FakeClient:
    def __init__(self, detail=None, error=None):
        self.detail = detail
        self.error = error
        self.calls = []

    def get_post(self, post_id):
        self.calls.append(post_id)
        if self.error:
            raise self.error
        return self.detail


def test_a_missing_history_falls_back_to_the_post_detail():
    post = {"id": 7}
    question = {"id": 42}
    detail = {"id": 7, "question": {"id": 42, "my_forecasts": {"history": [{"x": 1}]}}}
    client = FakeClient(detail)
    assert runner._seen_before(client, post, question) is True
    assert client.calls == [7]


def test_the_detail_can_also_say_not_yet():
    post = {"id": 7}
    question = {"id": 42}
    detail = {"id": 7, "question": {"id": 42, "my_forecasts": {"history": []}}}
    assert runner._seen_before(FakeClient(detail), post, question) is False


def test_when_nothing_can_answer_the_question_is_forecast_anyway():
    # A second forecast costs no points under spot peer scoring; a question
    # never forecast scores zero, which is the expensive failure.
    from bot.client import MetaculusError

    post = {"id": 7}
    question = {"id": 42}
    client = FakeClient(error=MetaculusError("503"))
    assert runner._seen_before(client, post, question) is False
