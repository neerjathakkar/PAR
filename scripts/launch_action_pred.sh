NUM_AGENTS=2
AG_EMB=true
LOSS_SAME_AG=true

MAX_EPOCHS=500
BATCH_SIZE=32
LR="5e-5"
WEIGHT_DECAY="5e-2"
WARMUP_EPOCHS=5
GRAD_CLIP_VAL=2.0

LIMIT_TRAIN_BATCHES=2000
LIMIT_VAL_BATCHES=1.0

HEADS=8 
DEPTH=8
HIDDEN_SIZE=128

RANDOM_SEED=1
DISABLE_EMA=false
EMA_DECAY=0.999

TASK_NAME="action_pred_ava_${NUM_AGENTS}_agent"

# Print parsed values for debugging
echo "Task name: $TASK_NAME"
echo "NUM AGENTS: $NUM_AGENTS"
echo "LEARNING RATE: $LR"
echo "MAX EPOCHS: $MAX_EPOCHS"
echo "BATCH SIZE: $BATCH_SIZE"
echo "WEIGHT DECAY: $WEIGHT_DECAY"
echo "WARMUP EPOCHS: $WARMUP_EPOCHS"
echo "GRADIENT CLIP VALUE: $GRAD_CLIP_VAL"
echo "LIMIT TRAIN BATCHES: $LIMIT_TRAIN_BATCHES"
echo "LIMIT VAL BATCHES: $LIMIT_VAL_BATCHES"
echo "HEADS: $HEADS"
echo "DEPTH: $DEPTH"
echo "HIDDEN SIZE: $HIDDEN_SIZE"
echo "RANDOM SEED: $RANDOM_SEED"
echo "DISABLE EMA: $DISABLE_EMA"
echo "EMA DECAY: $EMA_DECAY"


if [ "$HEADS" != "8" ]; then TASK_NAME+="_heads_${HEADS}"; fi
if [ "$DEPTH" != "8" ]; then TASK_NAME+="_depth_${DEPTH}"; fi
if [ "$HIDDEN_SIZE" != "128" ]; then TASK_NAME+="_hsize_${HIDDEN_SIZE}"; fi
if [ "$AG_EMB" = false ]; then TASK_NAME+="_no_ag_id_emb"; fi
if [ "$LOSS_SAME_AG" = false ]; then TASK_NAME+="_no_same_ag_loss"; fi
if [ "$DISABLE_EMA" = true ]; then TASK_NAME+="_no_ema"; fi
if [ "$WARMUP_EPOCHS" != "5" ]; then TASK_NAME+="_warmup_${WARMUP_EPOCHS}"; fi
TASK_NAME+="_lr_${LR}"
TASK_NAME+="_ema_decay_${EMA_DECAY}"
TASK_NAME+="_seed_${RANDOM_SEED}"

echo "TASK NAME: $TASK_NAME"

if [ "$NUM_AGENTS" = "1" ]; then
    AG_EMB=false
    LOSS_SAME_AG=false
fi

python -m PAR.train -m \
--config-name ava.yaml \
task_name=$TASK_NAME \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.train_batch_size=$BATCH_SIZE \
configs.test_batch_size=1 \
configs.train_num_workers=1 \
configs.test_num_workers=1 \
configs.solver.lr=$LR \
configs.solver.weight_decay=$WEIGHT_DECAY \
configs.solver.warmup_epochs=$WARMUP_EPOCHS \
trainer.max_epochs=$MAX_EPOCHS \
trainer.gradient_clip_val=$GRAD_CLIP_VAL \
callbacks.rich_progress_bar.refresh_rate=-1  \
trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
configs.do_vis=False \
configs.num_agents=$NUM_AGENTS \
configs.transformer.depth=$DEPTH \
configs.transformer.heads=$HEADS \
configs.transformer.hsize=$HIDDEN_SIZE \
configs.use_multiagent_pos_emb=$AG_EMB \
configs.loss_on_same_agent=$LOSS_SAME_AG \
seed=$RANDOM_SEED \
callbacks.ema.validate_original_weights=$DISABLE_EMA \
callbacks.ema.decay=$EMA_DECAY 