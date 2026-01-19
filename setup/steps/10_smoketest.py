#!/usr/bin/env python3

import os
import sys
import tempfile
import re
from pathlib import Path
import pykube
import ipaddress
from pykube.objects import NamespacedAPIObject

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.insert(0, str(project_root))

# ---------------- Import local packages ----------------

from functions import announce, \
        environment_variable_to_dict, \
        is_standalone_deployment, \
        get_accelerator_type, \
        llmdbench_execute_cmd, \
        model_attribute, \
        get_model_name_from_pod, \
        get_image, \
        kubectl_get, \
        kube_connect

# ---------------- Helpers ----------------

def check_deployment(api: pykube.HTTPClient, client: any, ev: dict):
    """
    Checking if current deployment was successful
    """

    announce("🔍 Checking if current deployment was successful...")
    dry_run = ev["control_dry_run"]
    verbose = ev["control_verbose"]

    """
    Checking if service/gateway was successfully deployed
    """
    service_ip = "N/A"
    service_hostname = "N/A"
    service_name = "N/A"

    gateway_port = "80"

    if is_standalone_deployment(ev):
        pod_string = "standalone"
        try:
            all_services = client.CoreV1Api().list_namespaced_service(namespace=ev["vllm_common_namespace"], watch=False)
            for service in all_services.items:
                if pod_string in service.metadata.name:
                    service_name = service.metadata.name
            service_name = service.metadata.name
            service_ip=service.spec.cluster_ip
            service_type = "service"
            route_string = service_name + '-route'
        except client.ApiException as e:
            announce(f"ERROR: unable to find service: {e}")
            return 1
    else:
        pod_string = "decode"
        route_string=f"{ev.get('vllm_modelservice_release', '')}-inference-gateway-route"
        service_type = "gateway"
        try:
            gateways = client.CustomObjectsApi().list_namespaced_custom_object(
                group="gateway.networking.k8s.io",
                version="v1",
                namespace=ev["vllm_common_namespace"],
                plural="gateways"
            )
            for service in gateways['items']:

                for mf in service["metadata"]["managedFields"] :
                    if 'fieldsV1' in mf :
                        if 'f:status' in mf['fieldsV1'] :
                            if 'f:listeners' in mf['fieldsV1']['f:status'] :
                                for k in mf['fieldsV1']['f:status']['f:listeners'].keys() :
                                    if k.count('https') :
                                        gateway_port = "443"

                if service['metadata']['name'] == f"infra-{ev.get('vllm_modelservice_release', '')}-inference-gateway":
                    service_name = service['metadata']['name']
                    if "addresses" in service["status"] :
                        for address in service["status"]["addresses"]:
                            if address.get("type") == "IPAddress":
                                service_ip = address.get("value")
                            if address.get("type") == "Hostname":
                                service_ip = address.get("value")
                                service_hostname = address.get("value")
                                break
                    else:
                        announce(f"ERROR: Unable to find an address for gateway {service_name}")
                        return 1
                    break
        except client.ApiException as e:
            announce(f"ERROR: Unable to find a gateway: {e}")
            return 1

    if dry_run:
        service_name = "localhost"
        service_ip = "127.0.0.8"
    else:
        if not service_name:
            announce(f"ERROR: No {service_type} found with string \"{pod_string}\"!")
            return 1
        elif not service_ip:
            announce(f"ERROR: Unable to find IP for service/gateway \"{service}\"!")
            return 1
        else:
            if service_hostname == "N/A":
                try:
                    ipaddress.ip_address(service_ip)
                except ValueError:
                    announce(f"ERROR: Invalid IP (\"{service_ip}\") for service/gateway \"{service_name}\"!")
                    return 1

    """
    Checking if pods were successfully deployed
    """
    model_list = ev.get("deploy_model_list", "").replace(",", " ").split()
    for model in model_list:
        current_model = model_attribute(model, "model", ev)
        current_model_ID = model_attribute(model, "modelid", ev)
        current_model_ID_label = model_attribute(model, "modelid_label", ev)

    if dry_run:
        pod_ip_list = ["127.0.0.4"]
    else :
        try:
            pod_ip_list = []
            if is_standalone_deployment(ev):
                pods = client.CoreV1Api().list_namespaced_pod(namespace=ev["vllm_common_namespace"])
                for pod in pods.items:
                    if pod_string in pod.metadata.name:
                        pod_ip_list.append(pod.status.pod_ip)
            else:
                pods = client.CoreV1Api().list_namespaced_pod(namespace=ev["vllm_common_namespace"], label_selector=f"llm-d.ai/model={current_model_ID_label},llm-d.ai/role={pod_string}")
                for pod in pods.items:
                    pod_ip_list.append(pod.status.pod_ip)
        except client.ApiException as e:
            announce(f"ERROR: Unable to find pods in namespace {ev['vllm_common_namespace']}: {e}")
            return 1

    if not pod_ip_list:
        announce(f"ERROR: Unable to find IPs for pods \"{pod_string}\"!")
        return 1

    announce(f"🚀 Testing all pods \"{pod_string}\" (port {ev['vllm_common_inference_port']})...")
    for pod_ip in pod_ip_list:
        announce(f"       🚀 Testing pod ip \"{pod_ip}\" ...")
        if dry_run:
            announce(f"       ✅ [DRY RUN] Pod ip \"{pod_ip}\" responded successfully ({current_model})")
        else:
            received_model_name, curl_command_used = get_model_name_from_pod(api, client, ev, pod_ip)
            if received_model_name == current_model:
                announce(f"       ✅ Pod ip \"{pod_ip}\" responded successfully ({received_model_name})")
            else:
                announce(f"       ERROR: Pod ip \"{pod_ip}\" responded to \"{curl_command_used}\" with model name \"{received_model_name}\" (instead of {current_model})!")
                return 1

    announce(f"✅ All pods respond successfully")
    announce(f"🚀 Testing service/gateway \"{service_ip}\" (port {gateway_port})...")

    if dry_run:
        announce(f"✅ [DRY RUN] Service responds successfully ({current_model})")
    else:
        received_model_name, curl_command_used = get_model_name_from_pod(api, client, ev, service_ip, gateway_port)
        if received_model_name == current_model:
            announce(f"✅ Service responds successfully ({received_model_name})")
        else:
            announce(f"ERROR: Service responded to \"{curl_command_used}\" with model name \"{received_model_name}\" (instead of {current_model})!")
            return 1

    route_url = ""
    if dry_run:
        True
    else:
        if ev['control_deploy_is_openshift'] == "1":

            route_instances, route_names = kubectl_get(api=api, \
                                                       object_api='route.openshift.io/v1', \
                                                       object_kind="Route", \
                                                       object_name = '', \
                                                       object_namespace=ev['vllm_common_namespace'])

            if route_instances:
                # TODO handle multiple routes, for now grab first
                for i in route_instances:
                    route_url = i.obj["spec"]["host"]
                    break

            if not route_url:
                announce(f"WARNING: unable to fetch route")

    if ev['control_deploy_is_openshift'] == "1" and route_url:
        announce(f"🚀 Testing external route \"{route_url}\"...")
        if is_standalone_deployment(ev):
            received_model_name, curl_command_used = get_model_name_from_pod(api, client, ev, route_url, gateway_port)
        else:
            received_model_name, curl_command_used = get_model_name_from_pod(api, client, ev, route_url, gateway_port)
        if received_model_name == current_model:
            announce(f"✅ External route responds successfully ({received_model_name})")
        else:
            announce(f"ERROR: External route responded to \"{curl_command_used}\" with model name \"{received_model_name}\" (instead of {current_model})!")
            return 1
    return 0

def check_deployment_fma(api: pykube.HTTPClient, client: any, ev: dict) -> int:
    """
    Checking if current FMA deployment was successful
    """

    class InferenceServerConfig(NamespacedAPIObject):
        """ Inference Server Config"""
        version = "fma.llm-d.ai/v1alpha1"
        endpoint = "inferenceserverconfigs"
        kind = "InferenceServerConfig"

    announce("🔍 Checking if current FMA deployment was successful...")
    dry_run = ev["control_dry_run"]

    pod_string = "decode"

    rc = 0
    model_list = ev.get("deploy_model_list", "").replace(",", " ").split()
    for model in model_list:
        current_model = model_attribute(model, "model", ev)
        if dry_run:
            announce(f"✅ [DRY RUN] FMA deployment successfull ({current_model})")
            continue

        current_model_id_label = model_attribute(model, "modelid_label", ev)
        label_sel = f"llm-d.ai/model={current_model_id_label},llm-d.ai/role={pod_string}"
        requester_pods = \
            client.CoreV1Api().list_namespaced_pod(namespace=ev["vllm_common_namespace"],
                                                    label_selector=label_sel)
        if not requester_pods.items:
            announce(f"ERROR: no requester pods found for model '{current_model}'!")
            rc = 1
            continue

        for pod in requester_pods.items:
            requester_pod_name = pod.metadata.name
            launcher_pod_name = pod.metadata.labels.get("dual-pods.llm-d.ai/dual")
            if launcher_pod_name is None or launcher_pod_name == "":
                announce(f"ERROR: no launcher pod found for requester pod '{pod.metadata.name}'!")
                rc = 1
                continue

            inference_server_config_name = \
                pod.metadata.annotations.get("dual-pods.llm-d.ai/inference-server-config")
            if inference_server_config_name is None or inference_server_config_name == "":
                announce("ERROR: no Inference Server Config found for requester pod "
                         f"'{requester_pod_name}'!")
                rc = 1
                continue

            try:
                inference_server = InferenceServerConfig.objects(api).get(
                    name=inference_server_config_name,
                    namespace=ev["vllm_common_namespace"],
                )
            except client.ApiException as e:
                announce("ERROR: Unable to find a inference_server "
                         f"'{inference_server_config_name}' "
                         f"for requester pod: '{requester_pod_name}': {e}")
                rc = 1
                continue

            port = inference_server.obj.get("spec", {}).get("modelServerConfig", {}).get("port")
            try:
                launcher_pod = \
                    client.CoreV1Api().read_namespaced_pod(name=launcher_pod_name,
                                                           namespace=ev["vllm_common_namespace"])
                announce(f"✅ Requester pod {requester_pod_name} connected to vLLM server "
                         f"at http://{launcher_pod.status.pod_ip}:{port}")
            except client.ApiException as e:
                announce(f"ERROR: Unable to find a launcher pod '{launcher_pod_name}' "
                         f"for requester pod: '{requester_pod_name}': {e}")
                rc = 1
                continue

        if rc == 0:
            announce(f"✅ FMA deployment successfull ({current_model})")

    return rc


def main():
    """Main function following the pattern from other Python steps"""

    ev = {'current_step_name': os.path.splitext(os.path.basename(__file__))[0] }
    environment_variable_to_dict(ev)

    if ev["control_dry_run"]:
        announce("DRY RUN enabled. No actual changes will be made.")

    api, client = kube_connect(f'{ev["control_work_dir"]}/environment/context.ctx')

    # Execute the main logic
    if ev.get("fma_enabled", False):
        return check_deployment_fma(api, client, ev)

    return check_deployment(api, client, ev)


if __name__ == "__main__":
    sys.exit(main())
