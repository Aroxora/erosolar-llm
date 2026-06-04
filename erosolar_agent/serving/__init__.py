"""Serving for HF/vLLM-backed models (additive; the legacy MiniGPT serve.py is
left untouched).

  vllm_server.py  : build/launch a vLLM OpenAI server for a registered model
  agent_server.py : an OpenAI-compatible server that runs the AGENT LOOP on top
                    of the vLLM endpoint (so the website gets agentic behavior)
  registry_ext.py : resolve registry entries (backend/hf_model_id) to a source

Topology:  website -> agent_server (:8080, agent loop) -> vLLM (:8000, raw model)
"""
