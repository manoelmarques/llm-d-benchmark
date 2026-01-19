# IMPORTANT NOTE
# All parameters not defined here or exported externally will be the default values found in setup/env.sh
# Many commonly defined values were left blank (default) so that this scenario is applicable to as many environments as possible.

# Model parameters
#export LLMDBENCH_DEPLOY_MODEL_LIST="Qwen/Qwen3-0.6B"
#export LLMDBENCH_DEPLOY_MODEL_LIST="Qwen/Qwen3-32B"
#export LLMDBENCH_DEPLOY_MODEL_LIST=Qwen/Qwen2.5-0.5B-Instruct
#export LLMDBENCH_DEPLOY_MODEL_LIST="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
#export LLMDBENCH_DEPLOY_MODEL_LIST=ibm-granite/granite-vision-3.3-2b
#export LLMDBENCH_DEPLOY_MODEL_LIST=ibm-granite/granite-speech-3.3-8b
#export LLMDBENCH_DEPLOY_MODEL_LIST=ibm-granite/granite-3.3-8b-instruct
#export LLMDBENCH_DEPLOY_MODEL_LIST=ibm-granite/granite-3.3-2b-instruct
#export LLMDBENCH_DEPLOY_MODEL_LIST=ibm-ai-platform/micro-g3.3-8b-instruct-1b
#export LLMDBENCH_DEPLOY_MODEL_LIST="facebook/opt-125m"
#export LLMDBENCH_DEPLOY_MODEL_LIST="meta-llama/Llama-3.1-8B-Instruct"
#export LLMDBENCH_DEPLOY_MODEL_LIST="meta-llama/Llama-3.1-70B-Instruct"
#export LLMDBENCH_DEPLOY_MODEL_LIST="deepseek-ai/DeepSeek-R1-0528"

# Deploy methods
export LLMDBENCH_DEPLOY_METHODS=fma

# harness image
export LLMDBENCH_IMAGE_REGISTRY=quay.io
export LLMDBENCH_IMAGE_REPO=manoelmrqs
export LLMDBENCH_IMAGE_NAME=llm-d-benchmark
export LLMDBENCH_IMAGE_TAG=0.0.1

# readwrite pvc
export LLMDBENCH_VLLM_COMMON_VLLM_CACHE_ROOT=/model-cache
export LLMDBENCH_VLLM_COMMON_EXTRA_PVC_NAME=model-pvc
export LLMDBENCH_VLLM_COMMON_EXTRA_VOLUME_MOUNTS="$(cat <<EOF
- name: ${LLMDBENCH_VLLM_COMMON_EXTRA_PVC_NAME}
  mountPath: ${LLMDBENCH_VLLM_COMMON_VLLM_CACHE_ROOT}
EOF
)"
export LLMDBENCH_VLLM_COMMON_EXTRA_VOLUMES="$(cat <<EOF
- name: ${LLMDBENCH_VLLM_COMMON_EXTRA_PVC_NAME}
  persistentVolumeClaim:
    claimName: ${LLMDBENCH_VLLM_COMMON_EXTRA_PVC_NAME}
EOF
)"

# Workload parameters
export LLMDBENCH_FMA_ITERATIONS=3
#export LLMDBENCH_VLLM_COMMON_NAMESPACE=???
#export LLMDBENCH_HARNESS_NAMESPACE=???
export LLMDBENCH_CONTROL_WORK_DIR=~/data/fma-llm-d-benchmark
#export LLMDBENCH_HARNESS_NAME=nop
#export LLMDBENCH_HARNESS_EXPERIMENT_PROFILE=nop.yaml

export LLMDBENCH_VLLM_COMMON_VLLM_LOGGING_LEVEL=DEBUG
export LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG="{ \\\"enable_multithread_load\\\": true, \\\"num_threads\\\": 8 }"
export LLMDBENCH_VLLM_COMMON_ENABLE_SLEEP_MODE=true
