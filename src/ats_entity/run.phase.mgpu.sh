#!/bin/bash


help()
{
    echo "Usage: run.phase.mgpu.sh --run_session_type [train/test]"
    echo "                         --datasource       [cnn/xsum/gigaword]"
    echo "                         --alias            [cnndm/xsum/gigaword]"
    echo "                         --token_type       [_bpe/'']"
    echo "                         --data_build_type  [struct/struct.ner_coref]"
    echo "                         --pretrained_model [facebook/bart-base|facebook/bart-large-cnn|facebook/bart-large-xsum]"
    echo "                         --split_type       [\"train\",\"dev\"/\"test\"]/[\"train\",\"validation\"/\"test\"]"
    echo "                         --pair_type        [\"document\",\"summary\"]/[\"article\",\"highlights\"]"
    echo "                         --data_file_type   [doc/struct]"
    echo "                         --sd_vocab_set     [super/sub]"
    echo "                         --run_phase        [pretraining/fulltraining]"
    echo "                         --dataset_changed  [true/false]"
    echo "                         --validate_summary [true/false]"
}

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=26
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

while :
do
  case "$1" in
    --run_session_type )
      RUN_SESSION_TYPE="$2"
      shift 2
      ;;
    --datasource )
      DATASOURCE="$2"
      shift 2
      ;;
    --alias )
      DATASOURCE_ALIAS="$2"
      shift 2
      ;;
    --token_type )
      TOKEN_TYPE="$2"
      shift 2
      ;;
    --data_build_type )
      DATASET_BUILD_TYPE="$2"
      shift 2
      ;;
    --pretrained_model )
      PRETRAINED_MODEL_TYPE="$2"
      shift 2
      ;;
    --split_type )
      DATA_SPLIT_TYPE="$2"
      shift 2
      ;;
    --pair_type )
      DATA_PAIR_TYPE="$2"
      shift 2
      ;;
    --data_file_type )
      DATA_FILE_TYPE="$2"
      shift 2
      ;;
    --run_phase )
      RUN_PHASE="$2"
      shift 2
      ;;
    --sd_vocab_set )
      SD_VOCAB_SET="$2"
      shift 2
      ;;
    --validate_summary )
      VALIDATE_SUMMARY="$2"
      shift 2
      ;;
    --dataset_changed )
      DATASET_CHANGED="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      # echo "Unexpected option: $1"
      # help
      break
      ;;
  esac
done


source /data/your_directory/dev/pyvenv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CPATH=/usr/local/cuda-11.3/include:$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO


FOLDER_NAME="`basename $PWD`"
MATE_DIR=/home/your_directory/Data/n210801.mate/src/ml

DATASOURCE_DIR=dataset/${DATASOURCE_ALIAS}/by_stanford-corenlp-4.4.0

if [[ ${DATASET_BUILD_TYPE} == "struct" ]]; then
    SD_VOCAB_FOLDER="."
    COREF_VOCAB_FOLDER="../struct.ner_coref"
else
    if [[ ${SD_VOCAB_SET} == "super" ]]; then
        SD_VOCAB_FOLDER="../struct"
    else
        SD_VOCAB_FOLDER="."
    fi
    COREF_VOCAB_FOLDER="."
fi

DATA_DOC=""
if [[ ${DATA_FILE_TYPE} == "doc" ]]; then
    DATA_DOC=".doc"
fi

DATASET_CHANGED=[ ${DATASET_CHANGED} == "true" ]

VALIDATE_SUMMARY_DIR="${MATE_DIR}/${FOLDER_NAME}/validation.smr"
[ -d ${VALIDATE_SUMMARY_DIR} ] || mkdir -p ${VALIDATE_SUMMARY_DIR}

DATASET_ROOT=${MATE_DIR}/${DATASOURCE_DIR}/${DATASOURCE_ALIAS}${TOKEN_TYPE}

RUN_TRACE_DIR="${MATE_DIR}/${FOLDER_NAME}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_TRACE_DIR}/${DATASOURCE_ALIAS}_${RUN_SESSION_TYPE}_results_$today.out"

echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

echo "--modeldata_root:             ${MATE_DIR}/${FOLDER_NAME}"
echo "--dataset_root:               ${DATASET_ROOT}"
echo "--config_folder:              ${DATASOURCE_ALIAS}"
echo "--dataset_folder:             ${DATASET_BUILD_TYPE}"
echo "--base_model_pretrained_name: ${PRETRAINED_MODEL_TYPE}"
echo "--tokenizer_name:             ${PRETRAINED_MODEL_TYPE}"
echo "--split_type:                 ${DATA_SPLIT_TYPE}"
echo "--pair_type:                  ${DATA_PAIR_TYPE}"
echo "--dataset_file:               {split_type}.{pair_type}${DATA_DOC}.dataset.json"
echo "--regularize_phase:           ${RUN_PHASE}"
echo "--coref_animacy_vocab_file:   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json"
echo "--coref_gender_vocab_file:    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json"
echo "--coref_number_vocab_file:    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json"
echo "--coref_type_vocab_file:      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json"
echo "--validate_summary:           ${VALIDATE_SUMMARY}"
echo "--dataset_changed:            ${DATASET_CHANGED}"


if [ "${VALIDATE_SUMMARY}" = "true" ]; then
    VSS_MAX_FREQ=2
else
    VSS_MAX_FREQ=0
fi


if [ "${RUN_SESSION_TYPE}" = "train" ]; then
    accelerate launch --config_file ./accelerate_config.${DATASOURCE_ALIAS}.yaml train_main.py \
        --modeldata_root             ${MATE_DIR}/${FOLDER_NAME} \
        --dataset_root               ${DATASET_ROOT} \
        --config_folder              ${DATASOURCE_ALIAS} \
        --dataset_folder             ${DATASET_BUILD_TYPE} \
        --base_model_pretrained_name ${PRETRAINED_MODEL_TYPE} \
        --tokenizer_name             ${PRETRAINED_MODEL_TYPE} \
        --use_slow_tokenizer                                  \
        --split_type                 ${DATA_SPLIT_TYPE} \
        --pair_type                  ${DATA_PAIR_TYPE} \
        --dataset_file               {split_type}.{pair_type}${DATA_DOC}.dataset.json \
        --regularize_phase           ${RUN_PHASE} \
        --modeling_choice            model_mrl \
        --coref_animacy_vocab_file   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json \
        --coref_gender_vocab_file    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json \
        --coref_number_vocab_file    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json \
        --coref_type_vocab_file      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json \
        --seed 19786403 \
        --time_limit -1 \
        --dataset_changed            ${DATASET_CHANGED} \
        --vss_freq 10 \
        --vss_max_freq               ${VSS_MAX_FREQ} \
        --vss_n_samples 4 \
        --vss_folder                 ${VALIDATE_SUMMARY_DIR} \
        --query_model_size \
        --early_stop_count_on_rouge 3 >>${RUN_LOG} 2>&1 &
else
    accelerate launch --config_file ./accelerate_config.${DATASOURCE_ALIAS}.yaml test_main.py \
        --modeldata_root             ${MATE_DIR}/${FOLDER_NAME} \
        --dataset_root               ${DATASET_ROOT} \
        --config_folder              ${DATASOURCE_ALIAS} \
        --dataset_folder             ${DATASET_BUILD_TYPE} \
        --base_model_pretrained_name ${PRETRAINED_MODEL_TYPE} \
        --tokenizer_name             ${PRETRAINED_MODEL_TYPE} \
        --use_slow_tokenizer                                  \
        --split_type                 ${DATA_SPLIT_TYPE} \
        --pair_type                  ${DATA_PAIR_TYPE} \
        --dataset_file               {split_type}.{pair_type}${DATA_DOC}.dataset.json \
        --regularize_phase           ${RUN_PHASE} \
        --modeling_choice            model_mrl \
        --coref_animacy_vocab_file   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json \
        --coref_gender_vocab_file    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json \
        --coref_number_vocab_file    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json \
        --coref_type_vocab_file      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json \
        --test_batch_size 4 >>${RUN_LOG} 2>&1 &
fi
