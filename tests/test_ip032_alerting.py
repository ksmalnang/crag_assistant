"""Tests for IP-032: Alerting for critical ingestion failures."""

import os
from unittest.mock import MagicMock, patch

from ingestion.alerter import (
    SlackAlerter,
    _get_top_errors,
    check_and_alert,
)

# ─── SlackAlerter Tests ──────────────────────────────────────────────────────


class TestSlackAlerter:
    """Tests for the SlackAlerter class."""

    def test_is_configured_with_url(self):
        """Should be configured when webhook URL is provided."""
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        assert alerter.is_configured is True

    def test_is_not_configured_without_url(self):
        """Should not be configured when webhook URL is missing."""
        alerter = SlackAlerter()
        # If SLACK_WEBHOOK_URL env var is not set, should be False
        if not os.environ.get("SLACK_WEBHOOK_URL"):
            assert alerter.is_configured is False

    @patch("ingestion.alerter.urllib_request.urlopen")
    def test_send_alert_success(self, mock_urlopen):
        """send_alert should return True on success."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        result = alerter.send_alert(
            run_id="run-001",
            error_rate=0.15,
            error_count=3,
            top_errors=[{"reason": "parse_error", "count": 2}],
        )

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("ingestion.alerter.urllib_request.urlopen")
    def test_send_alert_failure(self, mock_urlopen):
        """send_alert should return False on HTTP error."""
        mock_urlopen.side_effect = Exception("Network error")

        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        result = alerter.send_alert(
            run_id="run-001",
            error_rate=0.15,
            error_count=3,
            top_errors=[{"reason": "parse_error", "count": 2}],
        )

        assert result is False

    def test_send_alert_skipped_if_not_configured(self):
        """send_alert should skip gracefully if not configured."""
        alerter = SlackAlerter()
        # Should not raise, just return False
        result = alerter.send_alert(
            run_id="run-001",
            error_rate=0.15,
            error_count=3,
            top_errors=[],
        )
        # Result depends on whether SLACK_WEBHOOK_URL env var is set
        if not os.environ.get("SLACK_WEBHOOK_URL"):
            assert result is False

    def test_alert_message_contains_retry_command(self):
        """Alert message should include retry CLI command."""
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        message = alerter._build_alert_message(
            run_id="run-001",
            error_rate=0.15,
            error_count=3,
            top_errors=[{"reason": "parse_error", "count": 2}],
        )

        assert "python -m ingestion.retry --run-id run-001" in message["text"]


# ─── check_and_alert Tests ───────────────────────────────────────────────────


class TestCheckAndAlert:
    """Tests for the check_and_alert convenience function."""

    @patch("ingestion.alerter.SlackAlerter.send_alert")
    def test_alerts_when_error_rate_exceeds_threshold(self, mock_send):
        """Should alert when error rate > 10%."""
        mock_send.return_value = True

        check_and_alert(
            run_id="run-001",
            total_files=100,
            failed_count=15,  # 15% error rate
            errors=[{"reason": "parse_error", "count": 15}],
        )

        mock_send.assert_called_once()

    @patch("ingestion.alerter.SlackAlerter.send_alert")
    def test_no_alert_when_error_rate_below_threshold(self, mock_send):
        """Should not alert when error rate < 10%."""
        mock_send.return_value = True

        check_and_alert(
            run_id="run-001",
            total_files=100,
            failed_count=5,  # 5% error rate
            errors=[{"reason": "parse_error", "count": 5}],
        )

        mock_send.assert_not_called()

    @patch("ingestion.alerter.SlackAlerter.send_schema_error_alert")
    def test_alerts_on_schema_error(self, mock_send):
        """Should alert on any schema error regardless of rate."""
        mock_send.return_value = True

        check_and_alert(
            run_id="run-001",
            total_files=100,
            failed_count=1,  # 1% error rate
            errors=[],
            has_schema_error=True,
        )

        mock_send.assert_called_once()

    @patch("ingestion.alerter.SlackAlerter.send_health_check_alert")
    def test_alerts_on_degraded_health(self, mock_send):
        """Should alert when health check is degraded."""
        mock_send.return_value = True

        check_and_alert(
            run_id="run-001",
            total_files=100,
            failed_count=5,
            errors=[],
            health_status="degraded",
        )

        mock_send.assert_called_once()

    def test_no_alert_when_no_files(self):
        """Should not alert when no files processed."""
        result = check_and_alert(
            run_id="run-001",
            total_files=0,
            failed_count=0,
            errors=[],
        )
        assert result is False


# ─── Helper Function Tests ───────────────────────────────────────────────────


class TestGetTopErrors:
    """Tests for _get_top_errors helper."""

    def test_returns_top_3_errors(self):
        """Should return top 3 most common errors."""
        errors = [
            {"reason": "parse_error"},
            {"reason": "parse_error"},
            {"reason": "parse_error"},
            {"reason": "empty_file"},
            {"reason": "empty_file"},
            {"reason": "corrupted"},
        ]

        result = _get_top_errors(errors)

        assert len(result) == 3
        assert result[0]["reason"] == "parse_error"
        assert result[0]["count"] == 3
        assert result[1]["reason"] == "empty_file"
        assert result[1]["count"] == 2

    def test_returns_fewer_if_less_than_3(self):
        """Should return fewer than 3 if less errors exist."""
        errors = [{"reason": "parse_error"}]

        result = _get_top_errors(errors, limit=3)

        assert len(result) == 1
        assert result[0]["reason"] == "parse_error"
        assert result[0]["count"] == 1
