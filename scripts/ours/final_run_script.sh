#!/bin/bash

set -e

STRATEGY=$1
SELECTION_STRAT=$2
MODEL=$3
TASK=$4
SEED=$5
MAX_REALIGNMENT_STEPS=$6
REALIGNMENT_BATCH_SIZE=$7
ADD_ARGS=$8

DATA_DIR=./data
CACHE_DIR=./cache
CKPT_DIR=./results/

DATASET="mix_opus100_nllb"

#********************************************REALIGNMENT LANGUAGE SETTING********************************************

#=================================BASELINE=================================
if [ "$SELECTION_STRAT" == "xt_afri" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
#=================================URIEL_DIVERSITY=================================
#=======================40_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_40" ]; then
    langs="af ar az eu mya_Mymr zh ewe_Latn fon_Latn fr lug_Latn ka el hau_Latn he ibo_Latn ja kaz_Cyrl kin_Latn ko lin_Latn ms ml mar_Deva nya_Latn gaz_Latn fa ru sna_Latn es tgl_Latn ta tel_Telu th tr twi_Latn ur vi xho_Latn yor_Latn zul_Latn"
#=================================RANDOM=================================
elif [ "$STRATEGY" == "baseline" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
    echo "Fine-tuning baseline."
    SELECTION_STRAT="ft_only"
else
    echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Exitting."
    exit 1
fi
#********************************************END REALIGNMENT LANGUAGE SETTING********************************************

#********************************************TASK SETTING********************************************
# THERE ARE DUPLICATED LANGS IN EVAL SET!!!
if [ "$TASK" == "xnli" ]; then
    # 14 XTREME-R + 18 AfriXNLI + 10 AmericasNLI + Ind + Mya 
    n_epochs=2
    eval_langs="ar bg de el es fr hi ru sw th tr ur vi zh amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul aym bzd cni gn hch nah oto quy shp tar ind mya"
elif [ "$TASK" == "wikiann" ]; then
    # 47 XTREME-R + 20 MasakhaNER2
    n_epochs=5
    eval_langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
elif [ "$TASK" == "xtreme_r.udpos" ]; then
    # 37 XTREME-R + 20 MasakhaNER2
    n_epochs=5
    eval_langs="af ar bg de el es et eu fa fi fr he hi hu id it ja kk ko lt mr nl pl pt ro ru ta te th tl tr uk ur vi wo yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
fi
#********************************************END TASK SETTING********************************************

# Print for confirmation
echo "ADD_ARGS: $ADD_ARGS"
echo "model: $MODEL"
echo "task: $TASK"
echo "Max realignment steps: $MAX_REALIGNMENT_STEPS"
echo "Selected realignment set: $SELECTION_STRAT, realignment langs: $langs"
echo "seeds: $SEED"
echo "epoch: $n_epochs"
echo "eval_langs: $eval_langs"

mkdir -p $DATA_DIR

ADD_ARGS_SAVE_NAME="${ADD_ARGS//--enable_weighted_sampling/__weighted}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--weighted_sampling_method /__}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--ucb_exploration_coef /__excoef}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--inner_batches_before_outer /__innerb}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--meta_learning_rate /__metalr}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lbsmoothing_eps/__smeps}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--with_regularization/__withreg}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lambda_entropy /__lamdae}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--use_adapter/__adapt}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--adapter_approach /__}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--noise_mixing_strat /__noisem}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--softmax_temp /__temp}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--meta_loss_type /__mlstp}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--decouple_meta_and_model_updates /__decouple}"

TEMP_NAME=""
for word in $ADD_ARGS_SAVE_NAME; do
    if [[ $word == __* ]]; then
        TEMP_NAME+="$word"
    fi
done
ADD_ARGS_SAVE_NAME="$TEMP_NAME"

SAVE_FOLDER_NAME=${SELECTION_STRAT}${ADD_ARGS_SAVE_NAME}
echo "SAVE_FOLDER_NAME: $SAVE_FOLDER_NAME"

TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=scripts/ours/results/
SUB_DIR=aggregated_results/$SAVE_FOLDER_NAME

mkdir -p $CACHE_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR

export DATA_DIR=$DATA_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR
export RESULT_DIR=$RESULT_DIR

uv run scripts/ours/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies $STRATEGY \
    --models $MODEL \
    --tasks $TASK \
    --cache_dir $CACHE_DIR \
    --n_epochs $n_epochs \
    --seed $SEED\
    --realignment_steps $MAX_REALIGNMENT_STEPS \
    --realignment_batch_size $REALIGNMENT_BATCH_SIZE\
    --right_langs $langs \
    --eval_langs $eval_langs \
    --output_file $RESULT_DIR/$SUB_DIR/${MODEL##*/}__${DATASET}__${STRATEGY}__${TASK}__${MAX_REALIGNMENT_STEPS}__${REALIGNMENT_BATCH_SIZE}.csv \
    --checkpoint_path $CKPT_DIR/$DATASET/$STRATEGY/$SUB_DIR/rebs_${REALIGNMENT_BATCH_SIZE} \
    --large_gpu $ADD_ARGS