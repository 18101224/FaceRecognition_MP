EXCLUDE_OPTS="--exclude=checkpoint/ --exclude=results/ --exclude=gits/ --exclude=logs/  \
 --exclude=repos/ --exclude=.gits/ --exclude=wandb/ --exclude=.venv/ --exclude=.git/ \
 --exclude=InstantID/checkpoints/ --exclude=InstantID/models/ --exclude=ConsistentID/SG161222/ --exclude=ConsistentID/JackAILab/ \
 --exclude=ConsistentID/laion/ --exclude=ConsistentID/JackAILab/"
SOURCE_DIR="."

SERVERS=(
    "server5:/home/gpuadmin/mj/rl/"
    "server7:/home/gpuadmin/mj/rl/"
    "server6:/home/gpuadmin/mj/rl/"
    "server3:/home/hcir-server3/mj/rl/"
    "mine1:/home/mj/rl/"
    
)
SERVERS=(
    "server7:/home/gpuadmin/mj/rl/"
)
# Exclude checkpoint directory from syncing


for SERVER in "${SERVERS[@]}"; do

    rsync -avz --delete --checksum $EXCLUDE_OPTS "$SOURCE_DIR" "$SERVER" &
done 

# wait
# echo "****************************************"
# echo "*                                      *"
# echo "*      🚀🚀 SYNCHRONIZATION DONE 🚀🚀      *"
# echo "*                                      *"
# echo "****************************************"
# echo ""


# rsync -avz --delete --checksum $EXCLUDE_OPTS "$SOURCE_DIR" "pm:/pscratch/sd/s/sgkim/hcir/rl"


