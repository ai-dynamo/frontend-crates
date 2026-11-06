# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public delivery check waits bounded for GitHub Pages to serve a new deploy."""

from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from verify_publication_transport import PROPAGATION_STATUSES, open_published


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def http_error(status):
    return HTTPError("https://example.test/index.html", status, "unavailable", {}, None)


def opener_from(outcomes):
    """Return an opener that yields each outcome in turn; exceptions are raised."""
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    opener.calls = calls
    return opener


REQUEST = Request("https://example.test/index.html")


def test_a_healthy_response_is_returned_without_waiting():
    clock = FakeClock()
    opener = opener_from(["ok"])
    assert (
        open_published(
            REQUEST,
            propagation_deadline=300,
            opener=opener,
            clock=clock,
            sleep=clock.sleep,
        )
        == "ok"
    )
    assert len(opener.calls) == 1 and clock.now == 1000.0


def test_propagation_statuses_are_retried_until_the_site_answers():
    clock = FakeClock()
    logged = []
    opener = opener_from([http_error(503), http_error(503), "ok"])
    result = open_published(
        REQUEST,
        propagation_deadline=300,
        backoff=10,
        opener=opener,
        clock=clock,
        sleep=clock.sleep,
        log=logged.append,
    )
    assert result == "ok"
    assert len(opener.calls) == 3 and clock.now == 1020.0
    assert len(logged) == 2 and "HTTP 503" in logged[0] and "attempt 1" in logged[0]


@pytest.mark.parametrize("status", sorted(PROPAGATION_STATUSES))
def test_every_propagation_status_counts_as_not_yet_reachable(status):
    clock = FakeClock()
    opener = opener_from([http_error(status), "ok"])
    assert (
        open_published(
            REQUEST,
            propagation_deadline=60,
            backoff=10,
            opener=opener,
            clock=clock,
            sleep=clock.sleep,
        )
        == "ok"
    )
    assert len(opener.calls) == 2


def test_connection_errors_are_retried_like_propagation():
    clock = FakeClock()
    opener = opener_from([URLError("connection refused"), "ok"])
    assert (
        open_published(
            REQUEST,
            propagation_deadline=60,
            backoff=10,
            opener=opener,
            clock=clock,
            sleep=clock.sleep,
        )
        == "ok"
    )
    assert len(opener.calls) == 2


def test_the_deadline_bounds_the_wait_and_raises_the_last_error():
    clock = FakeClock()
    opener = opener_from([http_error(503)])
    with pytest.raises(HTTPError) as raised:
        open_published(
            REQUEST,
            propagation_deadline=35,
            backoff=10,
            opener=opener,
            clock=clock,
            sleep=clock.sleep,
        )
    assert raised.value.code == 503
    # Attempts at t=0, 10, 20, 30; the next retry would land past the deadline.
    assert len(opener.calls) == 4 and clock.now == 1030.0


def test_other_failures_are_raised_immediately():
    clock = FakeClock()
    opener = opener_from([http_error(500), "ok"])
    with pytest.raises(HTTPError) as raised:
        open_published(
            REQUEST,
            propagation_deadline=300,
            opener=opener,
            clock=clock,
            sleep=clock.sleep,
        )
    assert raised.value.code == 500 and len(opener.calls) == 1 and clock.now == 1000.0


def test_without_a_deadline_the_first_failure_is_final():
    clock = FakeClock()
    opener = opener_from([http_error(503), "ok"])
    with pytest.raises(HTTPError):
        open_published(REQUEST, opener=opener, clock=clock, sleep=clock.sleep)
    assert len(opener.calls) == 1
