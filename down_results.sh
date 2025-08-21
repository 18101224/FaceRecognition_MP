#!/bin/bash

# Define local results directory
LOCAL_RESULTS_DIR="./results"

# Define servers (same as sync.sh)
SERVERS=(
    "server5:/home/gpuadmin/mj/rl/results/"
    "server7:/home/gpuadmin/mj/rl/results/"
    "server6:/home/gpuadmin/mj/rl/results/"
    "server3:/home/hcir-server3/mj/rl/results/"
    "mine1:/home/mj/rl/results/"
    "pm:/pscratch/sd/s/sgkim/hcir/rl/results"
)

# Create local results directory if it doesn't exist
mkdir -p "$LOCAL_RESULTS_DIR"

# Separate pm from the rest for prioritized download
PM_SERVER="pm:/pscratch/sd/s/sgkim/hcir/rl/results"
OTHER_SERVERS=()
for SERVER in "${SERVERS[@]}"; do
    if [[ "$SERVER" != "$PM_SERVER" ]]; then
        OTHER_SERVERS+=("$SERVER")
    fi
done

# Download results from all servers except pm in parallel
PIDS=()
for SERVER in "${OTHER_SERVERS[@]}"; do
    SERVER_NAME=$(echo $SERVER | cut -d':' -f1)
    echo "📥 Downloading results from: $SERVER_NAME"
    
    # Create server-specific directory
    SERVER_DIR="$LOCAL_RESULTS_DIR/$SERVER_NAME"
    mkdir -p "$SERVER_DIR"
    
    # Download using rsync in background (only new files)
    rsync -avz --ignore-existing "$SERVER" "$SERVER_DIR" &
    PIDS+=($!)
done

# Wait for all parallel downloads to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "****************************************"
echo "*   🚀🚀 Download from main servers done 🚀🚀   *"
echo "****************************************"
echo ""

# Now download from pm (serially, as it's slow)
PM_SERVER_NAME=$(echo $PM_SERVER | cut -d':' -f1)
echo "📥 Downloading results from: $PM_SERVER_NAME"
PM_SERVER_DIR="$LOCAL_RESULTS_DIR/$PM_SERVER_NAME"
mkdir -p "$PM_SERVER_DIR"
rsync -avz --ignore-existing "$PM_SERVER" "$PM_SERVER_DIR"

echo "****************************************"
echo "*      🚀🚀 ALL DOWNLOADS COMPLETE 🚀🚀      *"
echo "****************************************"
echo ""
echo "Download complete! Results are in $LOCAL_RESULTS_DIR"