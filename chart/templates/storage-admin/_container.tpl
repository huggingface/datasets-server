# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

{{- define "containerStorageAdmin" -}}
- name: "{{ include "name" . }}-storage-admin"
  image: {{ include "services.storageAdmin.image" . }}
  imagePullPolicy: {{ .Values.images.pullPolicy }}
  volumeMounts:
  {{ include "volumeMountDuckDBIndexRW" . | nindent 2 }}
  {{ include "volumeMountParquetMetadataRW" . | nindent 2 }}
  - mountPath: /volumes/duckdb-index
    mountPropagation: None
    name: volume-duckdb-index
    readOnly: false
  - mountPath: /volumes/parquet-metadata
    mountPropagation: None
    name: volume-parquet-metadata
    readOnly: false
  securityContext:
    runAsNonRoot: false
    runAsUser: 0
    runAsGroup: 0
  resources: {{ toYaml .Values.storageAdmin.resources | nindent 4 }}
{{- end -}}
