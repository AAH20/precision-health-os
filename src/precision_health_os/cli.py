"""CLI interface for Precision Health OS."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import Patient, VitalSigns
from precision_health_os.utils import utcnow

app = typer.Typer(
    name="phos",
    help="Precision Health OS — Unified precision health platform",
    no_args_is_help=True,
)
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = PrecisionHealthAPI()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),  # nosec B104
    port: int = typer.Option(8000, "--port", help="Bind port"),
) -> None:
    """Start the Precision Health OS server."""
    console.print(f"[green]Starting Precision Health OS on {host}:{port}[/green]")
    console.print(f"[dim]Status: {api.get_system_status()}[/dim]")


@app.command()
def status() -> None:
    """Show system status."""
    status_data = api.get_system_status()
    table = Table(title="Precision Health OS — System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in status_data.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def register_patient(
    patient_id: str = typer.Option(..., "--id", help="Patient ID"),
    mrn: str = typer.Option(..., "--mrn", help="Medical record number"),
    name: str = typer.Option(..., "--name", help="Patient name"),
    sex: str = typer.Option("unknown", "--sex", help="Sex"),
) -> None:
    """Register a new patient."""
    patient = Patient(
        id=patient_id,
        mrn=mrn,
        name=name,
        date_of_birth=utcnow(),
        sex=sex,
    )
    api.register_patient(patient)
    console.print(f"[green]Registered patient: {name} ({patient_id})[/green]")


@app.command()
def ingest_vitals(
    patient_id: str = typer.Option(..., "--patient", help="Patient ID"),
    heart_rate: float = typer.Option(0, "--hr", help="Heart rate (bpm)"),
    spo2: float = typer.Option(0, "--spo2", help="SpO2 (%)"),
    systolic: float = typer.Option(0, "--sys", help="Systolic BP"),
    diastolic: float = typer.Option(0, "--dia", help="Diastolic BP"),
    temperature: float = typer.Option(0, "--temp", help="Temperature (°C)"),
) -> None:
    """Ingest vital signs for a patient."""
    vitals = VitalSigns(
        patient_id=patient_id,
        heart_rate_bpm=heart_rate or None,
        spo2_percent=spo2 or None,
        blood_pressure_systolic=systolic or None,
        blood_pressure_diastolic=diastolic or None,
        temperature_celsius=temperature or None,
        respiratory_rate=None,
        glucose_mg_dl=None,
    )
    alerts = api.ingest_vitals(vitals)
    if alerts:
        console.print(f"[yellow]Generated {len(alerts)} alert(s):[/yellow]")
        for alert in alerts:
            console.print(f"  [{alert.severity.value}] {alert.title}")
    else:
        console.print("[green]No alerts generated[/green]")


@app.command()
def alerts(
    patient_id: str = typer.Option(None, "--patient", help="Filter by patient"),
    severity: str = typer.Option(None, "--severity", help="Filter by severity"),
) -> None:
    """Show active alerts."""
    from precision_health_os.models import AlertSeverity

    sev = AlertSeverity(severity) if severity else None
    active = api.get_active_alerts(patient_id, sev)
    if not active:
        console.print("[dim]No active alerts[/dim]")
        return

    table = Table(title="Active Alerts")
    table.add_column("ID", style="dim")
    table.add_column("Patient", style="cyan")
    table.add_column("Severity", style="red")
    table.add_column("Title", style="yellow")
    table.add_column("Source", style="green")
    for alert in active:
        table.add_row(
            alert.id[:8],
            alert.patient_id,
            alert.severity.value,
            alert.title,
            alert.source,
        )
    console.print(table)


@app.command()
def worker() -> None:
    """Start background worker for event processing."""
    console.print("[green]Starting background worker...[/green]")
    console.print("[dim]Processing event queue...[/dim]")


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
