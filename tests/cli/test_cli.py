"""Tests for the CLI interface."""

from unittest.mock import patch

from typer.testing import CliRunner

from precision_health_os.cli import app, main

runner = CliRunner()


def test_status_command():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "patients_registered" in result.output


def test_register_patient_command():
    result = runner.invoke(
        app, ["register-patient", "--id", "p1", "--mrn", "m1", "--name", "Test Patient"]
    )
    assert result.exit_code == 0
    assert "Registered patient" in result.output
    assert "Test Patient" in result.output


def test_register_patient_with_sex():
    result = runner.invoke(
        app,
        ["register-patient", "--id", "p2", "--mrn", "m2", "--name", "Jane", "--sex", "female"],
    )
    assert result.exit_code == 0
    assert "Registered patient" in result.output


def test_register_patient_invalid_sex():
    result = runner.invoke(
        app,
        ["register-patient", "--id", "p3", "--mrn", "m3", "--name", "Bob", "--sex", "invalid"],
    )
    assert result.exit_code != 0


def test_ingest_vitals_no_alerts():
    result = runner.invoke(app, ["ingest-vitals", "--patient", "p1", "--hr", "70", "--spo2", "98"])
    assert result.exit_code == 0
    assert "No alerts generated" in result.output


def test_ingest_vitals_with_alerts():
    result = runner.invoke(app, ["ingest-vitals", "--patient", "p1", "--hr", "160"])
    assert result.exit_code == 0
    assert "Generated" in result.output
    assert "alert(s)" in result.output


def test_ingest_vitals_invalid_hr():
    result = runner.invoke(app, ["ingest-vitals", "--patient", "p1", "--hr", "400"])
    assert result.exit_code != 0


def test_ingest_vitals_all_params():
    result = runner.invoke(
        app,
        [
            "ingest-vitals",
            "--patient",
            "p1",
            "--hr",
            "70",
            "--spo2",
            "98",
            "--sys",
            "120",
            "--dia",
            "80",
            "--temp",
            "37.0",
        ],
    )
    assert result.exit_code == 0
    assert "No alerts generated" in result.output


def test_alerts_no_results():
    result = runner.invoke(app, ["alerts", "--patient", "nonexistent"])
    assert result.exit_code == 0
    assert "No active alerts" in result.output


def test_alerts_with_results():
    runner.invoke(app, ["ingest-vitals", "--patient", "alert-patient", "--hr", "160"])
    result = runner.invoke(app, ["alerts", "--patient", "alert-patient"])
    assert result.exit_code == 0
    assert "Active Alerts" in result.output


def test_alerts_with_severity_filter():
    runner.invoke(app, ["ingest-vitals", "--patient", "severity-patient", "--hr", "160"])
    result = runner.invoke(
        app, ["alerts", "--patient", "severity-patient", "--severity", "critical"]
    )
    assert result.exit_code == 0
    assert "Active Alerts" in result.output


def test_alerts_invalid_severity():
    result = runner.invoke(app, ["alerts", "--severity", "invalid"])
    assert result.exit_code != 0


def test_serve_command():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert "Starting Precision Health OS" in result.output


def test_serve_with_custom_host_port():
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert result.exit_code == 0
    assert "127.0.0.1:9000" in result.output


def test_worker_command():
    result = runner.invoke(app, ["worker"])
    assert result.exit_code == 0
    assert "Starting background worker" in result.output


def test_main_callable():
    with patch("precision_health_os.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()
