#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.insert(0, str(project_root))

# Import from functions.py
from functions import (
    announce,
    llmdbench_execute_cmd,
    model_attribute,
    check_storage_class,
    check_accelerator,
    check_network,
    discover_node_resources,
    add_annotations,
    add_config,
    check_priority_class,
    is_fma_deployment,
    environment_variable_to_dict,
    wait_for_pods_created_running_ready,
    kube_connect,
    collect_logs,
    propagate_standup_parameters,
    install_fma_crds,
    install_fma_clusterole,
    install_fma_components,
)


def main():
    """Deploy vLLM standalone models with Kubernetes Deployment, Service, and HTTPRoute."""

    ev = {"current_step_name": os.path.splitext(os.path.basename(__file__))[0]}
    environment_variable_to_dict(ev)

    # Check if standalone environment is active
    if not is_fma_deployment(ev):
        deploy_methods = ev["deploy_methods"]
        announce(f'⏭️  Environment types are "{deploy_methods}". Skipping this step.')
        return 1

    # Check storage class
    if not check_storage_class(ev):
        announce("ERROR: Failed to check storage class")
        sys.exit(1)

    if not discover_node_resources(ev):
        announce("ERROR: Failed to discover resources on nodes")
        sys.exit(1)

    if not check_accelerator(ev):
        announce("ERROR: Failed to check accelerator")
        sys.exit(1)

    if not check_network(ev):
        announce("ERROR: Failed to check network")
        sys.exit(1)

    if not check_priority_class(ev):
        announce("ERROR: Failed to check priority class")
        sys.exit(1)

    # Create yamls directory
    yamls_dir = Path(ev["control_work_dir"]) / "setup" / "yamls"
    yamls_dir.mkdir(parents=True, exist_ok=True)

    api, client = kube_connect(f"{ev['control_work_dir']}/environment/context.ctx")
    api_client = client.CoreV1Api()

    srl = (
        "InferenceServerConfig,LauncherConfig,LauncherPopulationPolicy,ReplicaSet,pods"
    )

    # Fast Model Actuation CRDS
    result = install_fma_crds(api, ev)
    if result != 0:
        return result

    # Fast Model Actuation ClusterRole
    result = install_fma_clusterole(api, ev)
    if result != 0:
        return result

    # Process each model - First pass: Deploy resources
    model_list = ev["deploy_model_list"].replace(",", " ").split()
    for model in model_list:
        # Generate filename-safe model name
        modelfn = model.replace("/", "___")

        # Set current model environment variable
        current_model = model_attribute(model, "model", ev)
        ev["deploy_current_model"] = current_model

        # Fast Model Actuation chart
        result = install_fma_components(model, ev)
        if result != 0:
            return result

        # Wait for fma dual pod to be created, running, and ready
        result = wait_for_pods_created_running_ready(api_client, ev, 1, "fma_dual_pod")
        if result != 0:
            return result

        # Wait for fma launcher populator pod to be created, running, and ready
        result = wait_for_pods_created_running_ready(
            api_client, ev, 1, "fma_launcher_populator"
        )
        if result != 0:
            return result

        # Get model attributes
        model_label = model_attribute(model, "label", ev)

        # Generate Deployment YAML
        deployment_yaml = generate_deployment_yaml(ev, model_label)
        deployment_file = (
            yamls_dir / f"{ev['current_step']}_a_deployment_{modelfn}.yaml"
        )
        with open(deployment_file, "w") as f:
            f.write(deployment_yaml)

        announce(
            f'🚚 Deploying model "{model}" and associated service (from files located at {ev["control_work_dir"]})...'
        )

        # Apply deployment
        kubectl_deploy_cmd = f"{ev['control_kcmd']} apply -f {deployment_file}"
        llmdbench_execute_cmd(
            actual_cmd=kubectl_deploy_cmd,
            dry_run=ev["control_dry_run"],
            verbose=ev["control_verbose"],
            fatal=True,
        )

        announce(f'✅ Model "{model}" and associated resources deployed.')

    # Show resource snapshot
    announce(
        f'ℹ️ A snapshot of the relevant (model-specific) resources on namespace "{ev["vllm_common_namespace"]}":'
    )
    kubectl_get_cmd = (
        f"{ev['control_kcmd']} get --namespace {ev['vllm_common_namespace']} {srl}"
    )
    llmdbench_execute_cmd(
        actual_cmd=kubectl_get_cmd,
        dry_run=ev["control_dry_run"],
        verbose=ev["control_verbose"],
        fatal=False,
    )
    propagate_standup_parameters(ev, api)

    return 0


def generate_deployment_yaml(ev, model_label):
    """Generate Kubernetes Deployment YAML for FMA model."""

    # Generate annotations
    annotations = add_annotations(ev, "LLMDBENCH_VLLM_COMMON_ANNOTATIONS")

    cache_root = ev["vllm_common_vllm_cache_root"]
    extra_volume_mounts = add_config("vllm_common_extra_volume_mounts", 12, "", ev)
    extra_volumes = add_config("vllm_common_extra_volumes", 8, "", ev)
    extra_config = ev["vllm_common_model_loader_extra_config"]
    extra_config = json.dumps(
        json.loads(extra_config.encode().decode("unicode_escape")),
        separators=(",", ":"),
    ).replace('"', '\\"')

    config_opt = (
        f"--model {ev['deploy_current_model']} "
        "--no-enable-prefix-caching "
        "--load-format auto "
        "--max-model-len 16384 "
        "--gpu-memory-utilization 0.95 "
        "--tensor-parallel-size 1 "
        f"--model-loader-extra-config {extra_config}"
    )
    if "vllm_common_enable_sleep_mode" in ev and ev["vllm_common_enable_sleep_mode"]:
        config_opt += " --enable-sleep-mode"

    deployment_yaml = f"""apiVersion: fma.llm-d.ai/v1alpha1
kind: InferenceServerConfig
metadata:
  name: fma-{model_label}
  labels:
    app: fma-{model_label}
    stood-up-by: "{ev["control_username"]}"
    stood-up-from: llm-d-benchmark
    stood-up-via: "{ev["deploy_methods"]}"
  namespace: {ev["vllm_common_namespace"]}
spec:
  launcherConfigName: fma-{model_label}
  modelServerConfig:
    labels:
      component: inference
    env_vars:
      HF_HOME: "{cache_root}"
      VLLM_LOGGING_LEVEL: {ev["vllm_common_vllm_logging_level"]}
      VLLM_SERVER_DEV_MODE: "{ev["vllm_common_vllm_server_dev_mode"]}"
      VLLM_USE_V1: "1"
    options: "{config_opt}"
    port: 8005
---
apiVersion: fma.llm-d.ai/v1alpha1
kind: LauncherConfig
metadata:
  name: fma-{model_label}
  labels:
    app: fma-{model_label}
    stood-up-by: "{ev["control_username"]}"
    stood-up-from: llm-d-benchmark
    stood-up-via: "{ev["deploy_methods"]}"
  namespace: {ev["vllm_common_namespace"]}
spec:
  maxSleepingInstances: 3
  podTemplate:
    spec:
      runtimeClassName: nvidia-legacy
      containers:
        - name: inference-server
          image: {ev["fma_launcher_image_repository"]}:{ev["fma_launcher_image_tag"]}
          imagePullPolicy: IfNotPresent
          command:
            - /app/launcher.py
            - --host=0.0.0.0
            - --log-level=info
            - --port={ev["fma_launcher_config_port"]}
          env:
          - name: HOME
            value: "{cache_root}"
          - name: VLLM_CACHE_ROOT
            value: "{cache_root}/vllm"
          - name: FLASHINFER_WORKSPACE_DIR
            value: "{cache_root}/flashinfer"
          - name: TRITON_CACHE_DIR
            value: "{cache_root}/triton"
          - name: XDG_CACHE_HOME
            value: "{cache_root}"
          - name: XDG_CONFIG_HOME
            value: "{cache_root}/config"
          - name: HF_HOME
            value: "{cache_root}"
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: {ev["vllm_common_hf_token_name"]}
                key: HF_TOKEN
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
          volumeMounts:
          {extra_volume_mounts}
      volumes:
      {extra_volumes}
---
apiVersion: fma.llm-d.ai/v1alpha1
kind: LauncherPopulationPolicy
metadata:
  name: fma-{model_label}
  labels:
    app: fma-{model_label}
    stood-up-by: "{ev["control_username"]}"
    stood-up-from: llm-d-benchmark
    stood-up-via: "{ev["deploy_methods"]}"
  namespace: {ev["vllm_common_namespace"]}
spec:
  enhancedNodeSelector:
    labelSelector:
      matchLabels:
        nvidia.com/gpu.present: "true"
  countForLauncher:
    - launcherConfigName: fma-{model_label}
      launcherCount: 1
---
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: fma-requester-{model_label}
  labels:
    app: fma-requester-{model_label}
    stood-up-by: "{ev["control_username"]}"
    stood-up-from: llm-d-benchmark
    stood-up-via: "{ev["deploy_methods"]}"
  namespace: {ev["vllm_common_namespace"]}
  annotations:
{annotations}
spec:
  replicas: 0
  selector:
    matchLabels:
      app: dp-app
      llm-d.ai/inference-serving: "true"
      llm-d.ai/model: {model_label}
      llm-d.ai/role: requester
  template:
    metadata:
      labels:
        app: dp-app
        llm-d.ai/inference-serving: "true"
        llm-d.ai/model: {model_label}
        llm-d.ai/role: requester
      annotations:
        dual-pods.llm-d.ai/admin-port: "8081"
        dual-pods.llm-d.ai/inference-server-config: fma-{model_label}
    spec:
      containers:
        - name: inference-server
          image: {ev["fma_requester_image_repository"]}:{ev["fma_requester_image_tag"]}
          imagePullPolicy: Always
          command:
          - /app/requester
          ports:
          - name: probes
            containerPort: {ev["fma_requester_probe_port"]}
          - name: spi
            containerPort: {ev["fma_requester_spi_port"]}
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 2
            periodSeconds: 5
          resources:
            limits:
              nvidia.com/gpu: {ev["fma_requester_limits_gpu"]}
              cpu: {ev["fma_requester_limits_cpu"]}
              memory: {ev["fma_requester_limits_memory"]}
"""
    return deployment_yaml


if __name__ == "__main__":
    sys.exit(main())
