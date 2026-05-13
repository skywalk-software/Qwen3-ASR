# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .qwen3_asr import Qwen3ASRForConditionalGeneration

def register():
    """Wire Qwen3-ASR into vLLM via the vllm.general_plugins entry point.

    Without this, vLLM resolves Qwen3ASRForConditionalGeneration to its
    built-in module which uses the wrong audio encoder.
    """
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        "qwen_asr.core.vllm_backend.qwen3_asr:Qwen3ASRForConditionalGeneration",
    )
