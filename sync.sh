# Exclude patterns, including those from .gitignore and project-specific paths
EXCLUDE_OPTS="--exclude=checkpoint/ --exclude=results/ --exclude=gits/ --exclude=logs/ \
 --exclude=repos/ --exclude=.gits/ --exclude=wandb/ --exclude=.venv/ --exclude=.git/ \
 --exclude=InstantID/checkpoints/ --exclude=InstantID/models/ --exclude=ConsistentID/SG161222/ --exclude=ConsistentID/JackAILab/ \
 --exclude=ConsistentID/laion/ --exclude=ConsistentID/JackAILab/ \
 --exclude=QCS/models/pretrain/ --exclude=datas/ --exclude=QCS/checkpoint_raf_db/ --exclude=QCS/log_raf_db/ --exclude=*.pth"

SOURCE_DIR="."

SERVERS=(
    "server5:/home/gpuadmin/mj/rl/"
    "server7:/home/gpuadmin/mj/rl/"
    "server6:/home/gpuadmin/mj/rl/"
    "server3:/home/hcir-server3/mj/rl/"
    "mine1:/home/mj/rl/"
    
)

SERVERS=(
    "mine1:/home/mj/rl/"
    "server5:/home/gpuadmin/mj/rl/"
    "server6:/home/gpuadmin/mj/rl/"
    "server7:/home/gpuadmin/mj/rl/"
)
# Exclude checkpoint directory from syncing


for SERVER in "${SERVERS[@]}"; do

    rsync -avz --delete --checksum $EXCLUDE_OPTS "$SOURCE_DIR" "$SERVER" &
    echo "synchroized to $SERVER"

done 
wait
exit 0
# wait
# echo "****************************************"
# echo "*                                      *"
# echo "*   🚀🚀 SYNCHRONIZATION DONE 🚀🚀     *"
# echo "*                                      *"
# echo "****************************************"
# echo ""


# rsync -avz --delete --checksum $EXCLUDE_OPTS "$SOURCE_DIR" "pm:/pscratch/sd/s/sgkim/hcir/rl"


