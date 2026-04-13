"""Alerting for critical ingestion failures via Slack webhook.

Alert triggers:
- error_rate > 10% of files in run
- any SchemaError
- health check returning 'degraded'

Sent to configured SLACK_WEBHOOK_URL (skipped gracefully if not set).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

# Alert thresholds
ERROR_RATE_THRESHOLD = 0.10  # 10%


class SlackAlerter:
    """
    Sends alert notifications to Slack via incoming webhook.

    Gracefully skips if SLACK_WEBHOOK_URL is not configured.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize the Slack alerter.

        Args:
            webhook_url: Slack incoming webhook URL.
                         Defaults to SLACK_WEBHOOK_URL env var.
        """
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    @property
    def is_configured(self) -> bool:
        """Check if Slack webhook is configured."""
        return bool(self.webhook_url)

    def send_alert(
        self,
        run_id: str,
        error_rate: float,
        error_count: int,
        top_errors: list[dict[str, str]],
        report_url: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Send a Slack alert for a failed ingestion run.

        Args:
            run_id: The ingestion run ID.
            error_rate: Fraction of files that failed (0.0-1.0).
            error_count: Number of failed files.
            top_errors: Top 3 error reasons with counts.
            report_url: URL to the run report.
            extra_context: Additional context for the alert.

        Returns:
            True if alert was sent successfully.
        """
        if not self.is_configured:
            logger.debug("Slack webhook not configured, skipping alert")
            return False

        # Build the alert message
        message = self._build_alert_message(
            run_id=run_id,
            error_rate=error_rate,
            error_count=error_count,
            top_errors=top_errors,
            report_url=report_url,
            extra_context=extra_context,
        )

        # Send to Slack
        return self._send_to_slack(message)

    def send_schema_error_alert(
        self,
        run_id: str,
        collection_name: str,
        report_url: Optional[str] = None,
    ) -> bool:
        """Send alert for a schema error."""
        return self.send_alert(
            run_id=run_id,
            error_rate=1.0,
            error_count=1,
            top_errors=[
                {
                    "reason": f"SchemaError in collection '{collection_name}'",
                    "count": 1,
                }
            ],
            report_url=report_url,
            extra_context={"alert_type": "schema_error"},
        )

    def send_health_check_alert(
        self,
        run_id: str,
        error_rate: float,
        error_count: int,
        top_errors: list[dict[str, str]],
        report_url: Optional[str] = None,
    ) -> bool:
        """Send alert for degraded health check."""
        return self.send_alert(
            run_id=run_id,
            error_rate=error_rate,
            error_count=error_count,
            top_errors=top_errors,
            report_url=report_url,
            extra_context={"alert_type": "health_check_degraded"},
        )

    def _build_alert_message(
        self,
        run_id: str,
        error_rate: float,
        error_count: int,
        top_errors: list[dict[str, str]],
        report_url: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the Slack alert message payload."""
        error_pct = f"{error_rate * 100:.1f}%"

        # Build the main text
        text = (
            f"🚨 *Ingestion Run Alert*\n"
            f"• Run ID: `{run_id}`\n"
            f"• Error rate: {error_pct}\n"
            f"• Failed files: {error_count}\n"
        )

        # Add top error reasons
        if top_errors:
            text += "\n*Top error reasons:*\n"
            for err in top_errors[:3]:
                text += f"  • {err['reason']} ({err['count']} files)\n"

        # Add report link
        if report_url:
            text += f"\n📊 Report: {report_url}"

        # Add retry command
        text += f"\n🔁 *Retry command:*\n`python -m ingestion.retry --run-id {run_id}`"

        # Add extra context if provided
        if extra_context:
            text += f"\n_({extra_context.get('alert_type', 'alert')})_"

        return {
            "text": text,
            "username": "Ingestion Pipeline",
            "icon_emoji": ":warning:",
        }

    def _send_to_slack(self, message: dict[str, Any]) -> bool:
        """
        Send message to Slack webhook.

        Args:
            message: Slack message payload.

        Returns:
            True if sent successfully.
        """
        if not self.webhook_url:
            return False

        data = json.dumps(message).encode("utf-8")
        req = urllib_request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info("Slack alert sent successfully")
                    return True
                logger.warning(f"Slack alert returned status {response.status}")
                return False
        except urllib_error.HTTPError as e:
            logger.error(f"Slack alert failed with HTTP {e.code}: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False


def check_and_alert(
    run_id: str,
    total_files: int,
    failed_count: int,
    errors: list[dict[str, str]],
    health_status: str = "green",
    has_schema_error: bool = False,
    report_url: Optional[str] = None,
) -> bool:
    """
    Convenience function to check alert conditions and send alert if needed.

    Alert triggers:
    - error_rate > 10% of files in run
    - any SchemaError
    - health check returning 'degraded'

    Args:
        run_id: The ingestion run ID.
        total_files: Total files processed.
        failed_count: Number of failed files.
        errors: List of error dictionaries.
        health_status: Health check status ('green', 'degraded', 'unhealthy').
        has_schema_error: Whether a schema error occurred.
        report_url: URL to the run report.

    Returns:
        True if alert was sent.
    """
    if total_files == 0:
        return False

    error_rate = failed_count / total_files
    alerter = SlackAlerter()

    # Check alert conditions
    should_alert = (
        error_rate > ERROR_RATE_THRESHOLD
        or has_schema_error
        or health_status in ("degraded", "unhealthy")
    )

    if not should_alert:
        return False

    # Build top error reasons
    top_errors = _get_top_errors(errors)

    # Send appropriate alert
    if has_schema_error:
        return alerter.send_schema_error_alert(
            run_id=run_id,
            collection_name="unknown",
            report_url=report_url,
        )

    if health_status in ("degraded", "unhealthy"):
        return alerter.send_health_check_alert(
            run_id=run_id,
            error_rate=error_rate,
            error_count=failed_count,
            top_errors=top_errors,
            report_url=report_url,
        )

    return alerter.send_alert(
        run_id=run_id,
        error_rate=error_rate,
        error_count=failed_count,
        top_errors=top_errors,
        report_url=report_url,
    )


def _get_top_errors(
    errors: list[dict[str, str]], limit: int = 3
) -> list[dict[str, str]]:
    """Get the top N most common error reasons."""
    reason_counts: dict[str, int] = {}
    for err in errors:
        reason = err.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    sorted_errors = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    return [
        {"reason": reason, "count": count} for reason, count in sorted_errors[:limit]
    ]
