# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "envAssets" -}}
- name: ASSETS_BASE_URL
  value: "{{ include "assets.baseUrl" . }}"
- name: ASSETS_STORAGE_ROOT
  value: {{ .Values.assets.storageRoot | quote }}
- name: ASSETS_STORAGE_PROTOCOL
  value: {{ .Values.assets.storageProtocol | quote }}
{{- end -}}
