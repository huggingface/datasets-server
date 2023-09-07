# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

{{- define "envCachedAssetsS3" -}}
- name: CACHED_ASSETS_S3_BUCKET
  value: {{ .Values.cachedAssetsS3.bucket | quote }}
- name: CACHED_ASSETS_S3_REGION
  value: {{ .Values.cachedAssetsS3.region | quote }}
- name: CACHED_ASSETS_S3_FOLDER_NAME
  value: {{ .Values.cachedAssetsS3.folderName | quote }}
- name: CACHED_ASSETS_S3_ACCESS_KEY_ID
  {{- if .Values.secrets.cachedAssetsS3.accessKeyId.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cachedAssetsS3.accessKeyId.secretName | quote }}
      key: CACHED_ASSETS_S3_ACCESS_KEY_ID
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cachedAssetsS3.accessKeyId.value | quote }}
  {{- end }}
- name: CACHED_ASSETS_S3_SECRET_ACCESS_KEY
  {{- if .Values.secrets.cachedAssetsS3.secretAccessKey.fromSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.cachedAssetsS3.secretAccessKey.secretName | quote }}
      key: CACHED_ASSETS_S3_SECRET_ACCESS_KEY
      optional: false
  {{- else }}
  value: {{ .Values.secrets.cachedAssetsS3.secretAccessKey.value | quote }}
  {{- end }}
{{- end -}}
