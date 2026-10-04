"""Edge-case tests for the CLI interface."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from precision_health_os.cli import app, main

runner = CliRunner()

COMMAND_NAMES = ["serve", "status", "register-patient", "ingest-vitals", "alerts", "worker"]


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMAND_NAMES:
        assert cmd in result.output


def test_invalid_command():
    result = runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code != 0


def test_ingest_vitals_all_zeros_no_alerts():
    result = runner.invoke(app, ["ingest-vitals", "--patient", "p1"])
    assert result.exit_code == 0
    assert "No alerts generated" in result.output


def test_alerts_no_active_alerts():
    from precision_health_os.cli import api

    api._alert_manager._alerts.clear()
    api._alert_manager._patient_alerts.clear()
    result = runner.invoke(app, ["alerts"])
    assert result.exit_code == 0
    assert "No active alerts" in result.output


def test_main_invokes_app():
    with patch("precision_health_os.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()


def test_main_entry_point_if_name_main():
    """Cover line 141: main() under if __name__ == '__main__'."""
    import precision_health_os.cli

    cli_path = Path(precision_health_os.cli.__file__)
    with (
        patch.object(sys, "argv", ["phos", "--help"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_path(str(cli_path), run_name="__main__")
    assert exc_info.value.code == 0
