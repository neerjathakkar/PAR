NUM_AGENTS=3
AG_EMB=true
LOSS_SAME_AG=true

ACC_TOKENS=true

MAX_EPOCHS=300
BATCH_SIZE=32
LR="1e-4"
WEIGHT_DECAY="5e-2"
WARMUP_EPOCHS=5
GRAD_CLIP_VAL=2.0

LIMIT_TRAIN_BATCHES=1000
LIMIT_VAL_BATCHES=1.0

DO_ROT_AUG=false
ACC_TOKEN_SIZE=13
HEADS=8
DEPTH=8
HIDDEN_SIZE=128

MULTINOMIAL_SAMPLING=false
RANDOM_SEED=1
DISABLE_EMA=false
EMA_DECAY=0.9999

USE_LOC_POS_EMB=true
REL_POS_LOCS=true
CAT_LOC_EMB=true
ENCODING_TYPE="sin_cos"
LOC_ENC_SIZE=100

TASK_NAME="car_traj_${NUM_AGENTS}_agent"

# Print parsed values for debugging
echo "Task name: $TASK_NAME"
echo "LEARNING RATE: $LR"
echo "MAX EPOCHS: $MAX_EPOCHS"
echo "BATCH SIZE: $BATCH_SIZE"
echo "WEIGHT DECAY: $WEIGHT_DECAY"
echo "WARMUP EPOCHS: $WARMUP_EPOCHS"
echo "GRADIENT CLIP VALUE: $GRAD_CLIP_VAL"
echo "LIMIT TRAIN BATCHES: $LIMIT_TRAIN_BATCHES"
echo "LIMIT VAL BATCHES: $LIMIT_VAL_BATCHES"
echo "DO ROTATION AUGMENTATION: $DO_ROT_AUG"
echo "ACC TOKEN SIZE: $ACC_TOKEN_SIZE"
echo "HEADS: $HEADS"
echo "DEPTH: $DEPTH"
echo "HIDDEN SIZE: $HIDDEN_SIZE"
echo "RANDOM SEED: $RANDOM_SEED"
echo "DISABLE EMA: $DISABLE_EMA"
echo "EMA DECAY: $EMA_DECAY"
echo "MULTINOMIAL SAMPLING: $MULTINOMIAL_SAMPLING"
echo "ACC TOKENS: $ACC_TOKENS"
echo "USE LOC POS EMB: $USE_LOC_POS_EMB"
echo "REL POS LOCS: $REL_POS_LOCS"
echo "CAT LOC EMB: $CAT_LOC_EMB"
echo "ENCODING TYPE: $ENCODING_TYPE"
echo "LOC ENC SIZE: $LOC_ENC_SIZE"

if [ "$ACC_TOKENS" = false ]; then TASK_NAME+="_vel_tokens"; fi
if [ "$ACC_TOKENS" = true ]; then TASK_NAME+="_${ACC_TOKEN_SIZE}_tok_size"; fi
if [ "$DO_ROT_AUG" = true ]; then TASK_NAME+="_rot_aug"; fi
if [ "$HEADS" != "8" ]; then TASK_NAME+="_heads_${HEADS}"; fi
if [ "$DEPTH" != "8" ]; then TASK_NAME+="_depth_${DEPTH}"; fi
if [ "$HIDDEN_SIZE" != "128" ]; then TASK_NAME+="_hsize_${HIDDEN_SIZE}"; fi
if [ "$AG_EMB" = false ]; then TASK_NAME+="_no_ag_id_emb"; fi
if [ "$LOSS_SAME_AG" = false ]; then TASK_NAME+="_no_same_ag_loss"; fi
if [ "$DISABLE_EMA" = true ]; then TASK_NAME+="_no_ema"; fi
if [ "$MULTINOMIAL_SAMPLING" = true ]; then TASK_NAME+="_multinomial_sampling"; fi
if [ "$USE_LOC_POS_EMB" = true ]; then TASK_NAME+="_loc_pos_emb"; fi
if [ "$USE_LOC_POS_EMB" = true ] && [ "$REL_POS_LOCS" = false ]; then TASK_NAME+="_global_locs"; else TASK_NAME+="_rel_locs"; fi
if [ "$USE_LOC_POS_EMB" = true ] && [ "$CAT_LOC_EMB" = false ]; then TASK_NAME+="_sum_loc_emb"; else TASK_NAME+="_cat_loc_emb"; fi
if [ "$USE_LOC_POS_EMB" = true ]; then TASK_NAME+="_${LOC_ENC_SIZE}_loc_enc_size"; fi
TASK_NAME+="_ema_decay_${EMA_DECAY}"
TASK_NAME+="_lr_${LR}"
TASK_NAME+="_seed_${RANDOM_SEED}"

echo "TASK NAME: $TASK_NAME"

VELOCITY_TOKS=false
if [ "$ACC_TOKENS" = false ]; then VELOCITY_TOKS=true; fi
if [ "$NUM_AGENTS" = "1" ]; then
    echo "Single agent"
    AG_EMB=false
    LOSS_SAME_AG=false
fi

python -m PAR.train -m \
--config-name car_traj.yaml \
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
configs.velocity_tokens=$VELOCITY_TOKS \
configs.do_rotation_aug=$DO_ROT_AUG \
configs.acc_token_size=$ACC_TOKEN_SIZE \
configs.transformer.depth=$DEPTH \
configs.transformer.heads=$HEADS \
configs.transformer.hsize=$HIDDEN_SIZE \
configs.use_multiagent_pos_emb=$AG_EMB \
configs.loss_on_same_agent=$LOSS_SAME_AG \
seed=$RANDOM_SEED \
callbacks.ema.validate_original_weights=$DISABLE_EMA \
callbacks.ema.decay=$EMA_DECAY \
configs.multinomial_sampling=$MULTINOMIAL_SAMPLING \
configs.location_pos_embedding=$USE_LOC_POS_EMB \
configs.relative_pos_locs=$REL_POS_LOCS \
configs.cat_loc_emb=$CAT_LOC_EMB \
configs.encoding_type=$ENCODING_TYPE \
configs.loc_enc_size=$LOC_ENC_SIZE