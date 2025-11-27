#!/bin/bash
# Start Dask scheduler with web dashboard

PORT=${1:-8787}
WORKERS=${2:-$(nproc)}

echo "=========================================="
echo "Starting Dask Dashboard"
echo "=========================================="
echo "Dashboard URL: http://localhost:$PORT"
echo "Workers: $WORKERS"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop"
echo ""

dask scheduler --port 8786 --dashboard-address :$PORT &
SCHEDULER_PID=$!

sleep 2

for i in $(seq 1 $WORKERS); do
    dask worker tcp://localhost:8786 &
done

echo ""
echo "✓ Dask cluster started"
echo "✓ Dashboard: http://localhost:$PORT"
echo ""

wait $SCHEDULER_PID
