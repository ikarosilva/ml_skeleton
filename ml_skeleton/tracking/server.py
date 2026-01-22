"""
MLflow server management with auto-start capability.

Handles starting, stopping, and health checking of the MLflow tracking server.
"""

from __future__ import annotations

import atexit
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


class MLflowServer:
    """
    Manages MLflow tracking server lifecycle.

    Can automatically start an MLflow server if one is not already running,
    and provides methods for checking server health.

    Example:
        # Auto-start server if needed
        server = MLflowServer.ensure_running()

        # Or manual management
        with MLflowServer(port=5000) as server:
            # Server is running
            print(f"MLflow UI: {server.tracking_uri}")
    """

    _instance: Optional["MLflowServer"] = None

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        backend_store_uri: str = "sqlite:///mlflow.db",
        artifact_root: str = "./mlruns",
        workers: int = 2,
    ):
        """
        Initialize MLflow server configuration.

        Args:
            host: Host to bind the server to
            port: Port to run the server on
            backend_store_uri: URI for the backend store (SQLite, PostgreSQL, etc.)
            artifact_root: Root directory for artifact storage
            workers: Number of gunicorn workers
        """
        self.host = host
        self.port = port
        self.backend_store_uri = backend_store_uri
        self.artifact_root = artifact_root
        self.workers = workers
        self._process: Optional[subprocess.Popen] = None

    @property
    def tracking_uri(self) -> str:
        """Get the tracking URI for this server."""
        return f"http://localhost:{self.port}"

    @classmethod
    def ensure_running(
        cls,
        host: str = "0.0.0.0",
        port: int = 5000,
        backend_store_uri: str = "sqlite:///mlflow.db",
        artifact_root: str = "./mlruns",
    ) -> "MLflowServer":
        """
        Ensure an MLflow server is running, starting one if needed.

        This is the recommended way to get an MLflow server when auto-start
        is enabled. It will reuse an existing server if one is already running.

        Args:
            host: Host to bind the server to
            port: Port to check/use
            backend_store_uri: Backend store URI if starting new server
            artifact_root: Artifact root if starting new server

        Returns:
            MLflowServer instance (may be existing or newly started)
        """
        # Check if server is already running at this port
        tracking_uri = f"http://localhost:{port}"
        if cls._check_health(tracking_uri):
            # Server already running, return a wrapper
            server = cls(host, port, backend_store_uri, artifact_root)
            server._process = None  # Not managed by us
            return server

        # Need to start a new server
        if cls._instance is not None and cls._instance._process is not None:
            # We have a managed instance, return it
            return cls._instance

        # Start new server
        server = cls(host, port, backend_store_uri, artifact_root)
        server.start(wait=True)
        cls._instance = server

        # Register cleanup on exit
        atexit.register(server.stop)

        return server

    @staticmethod
    def _check_health(tracking_uri: str, timeout: float = 2.0) -> bool:
        """Check if an MLflow server is healthy at the given URI."""
        try:
            response = requests.get(
                f"{tracking_uri}/health", timeout=timeout
            )
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def is_running(self) -> bool:
        """Check if this server instance is running."""
        return self._check_health(self.tracking_uri)

    def start(self, wait: bool = True, timeout: int = 30) -> None:
        """
        Start the MLflow server.

        Args:
            wait: Whether to wait for the server to be ready
            timeout: Maximum time to wait for server startup (seconds)
        """
        # Ensure artifact directory exists
        Path(self.artifact_root).mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "mlflow",
            "server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--backend-store-uri",
            self.backend_store_uri,
            "--default-artifact-root",
            self.artifact_root,
            "--workers",
            str(self.workers),
        ]

        # Start server process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if wait:
            self._wait_for_ready(timeout)

        print(f"MLflow server started at {self.tracking_uri}")

    def _wait_for_ready(self, timeout: int) -> None:
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_health(self.tracking_uri):
                return
            time.sleep(0.5)
        raise TimeoutError(f"MLflow server not ready after {timeout}s")

    def stop(self) -> None:
        """Stop the MLflow server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("MLflow server stopped")

    def __enter__(self) -> "MLflowServer":
        """Context manager entry - start the server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the server."""
        self.stop()
        return False
