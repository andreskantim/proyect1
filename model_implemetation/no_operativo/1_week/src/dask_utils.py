"""Dask utilities for distributed computing with web dashboard."""

from dask.distributed import Client, LocalCluster, progress
import dask
import os

def get_dask_client(n_workers=None, threads_per_worker=2, dashboard_address=':8787'):
    """
    Get or create Dask client with web dashboard.

    Args:
        n_workers: Number of workers (None = auto)
        threads_per_worker: Threads per worker
        dashboard_address: Dashboard address (default: localhost:8787)

    Returns:
        Dask client

    Usage:
        client = get_dask_client()
        print(f"Dashboard: {client.dashboard_link}")
    """
    try:
        # Try to connect to existing scheduler
        client = Client.current()
        print(f"✓ Connected to existing Dask cluster")
    except (ValueError, OSError):
        # Create new cluster
        if n_workers is None:
            n_workers = os.cpu_count()

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            dashboard_address=dashboard_address,
            silence_logs=False
        )
        client = Client(cluster)
        print(f"✓ Created Dask cluster: {n_workers} workers, {threads_per_worker} threads/worker")

    print(f"✓ Dashboard: {client.dashboard_link}")
    return client

def close_dask_client(client):
    """Close Dask client and cluster."""
    client.close()
    print("✓ Dask client closed")
