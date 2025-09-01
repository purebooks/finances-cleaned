#!/usr/bin/env python3

import io
import json
import time
import uuid

import pandas as pd
import numpy as np
from flask import request, jsonify, send_file, make_response

from flexible_column_detector import FlexibleColumnDetector
from common_cleaner import CommonCleaner
from cleaning_config import build_cleaner_config
from ai.overlook import apply_ai_overlook
from infra.metrics import timed


def register_routes(app, APP_CONFIG, get_llm_client):
	@app.route('/upload', methods=['POST'])
	def upload_file():
		"""Upload CSV/XLSX content, clean non-destructively, and return cleaned data.
		Preserves original column headers and order; does not add or rename columns.
		Accepts multipart form-data with file under key 'file', or raw bytes with
		query arg format=csv|xlsx. Optional JSON field 'config' for flags.
		"""
		request_id = f"upl-{uuid.uuid4().hex[:8]}"
		start_ts = time.time()

		with timed('upload'):
			try:
				cfg = {}
				if 'config' in request.form:
					try:
						cfg = json.loads(request.form.get('config', '{}') or '{}')
					except Exception:
						cfg = {}
				elif request.is_json and isinstance(request.get_json(silent=True), dict):
					cfg = request.get_json(silent=True) or {}

				# Get bytes and format
				file_storage = request.files.get('file')
				fmt = (request.args.get('format') or '').lower()
				content_bytes: bytes
				if file_storage:
					filename = file_storage.filename or ''
					content_bytes = file_storage.read()
					if not fmt:
						if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
							fmt = 'xlsx'
						else:
							fmt = 'csv'
				else:
					content_bytes = request.get_data() or b''
					if not fmt:
						fmt = 'csv'

				if not content_bytes:
					return jsonify({'error': 'No file content provided.'}), 400

				# Read into DataFrame without altering headers
				if fmt == 'xlsx':
					df = pd.read_excel(io.BytesIO(content_bytes), dtype=object)
				else:
					df = pd.read_csv(io.BytesIO(content_bytes), dtype=object)

				# Normalize various inbound shapes to DataFrame as the rest of the system expects
				detector = FlexibleColumnDetector()
				if not isinstance(df, pd.DataFrame):
					df = detector.normalize_to_dataframe(df)

				# Non-destructive cleaning
				cleaner = CommonCleaner(config=build_cleaner_config(APP_CONFIG, cfg))
				cleaned_df, summary = cleaner.clean(df)

				# Optional AI overlook (fill blanks only)
				try:
					apply_ai_overlook(cleaned_df, cfg, get_llm_client())
				except Exception:
					pass

				# Return JSON with same columns and order
				cleaned_records = cleaned_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')
				elapsed = time.time() - start_ts
				return jsonify({
					'cleaned_data': cleaned_records,
					'summary_report': {
						'schema_analysis': summary.schema_analysis,
						'processing_summary': summary.processing_summary,
						'math_checks': summary.math_checks,
						'performance_metrics': summary.performance_metrics,
					},
					'insights': {
						'processing_time': elapsed,
						'rows_processed': int(len(cleaned_df)),
						'ai_requests': 0,
						'ai_cost': 0.0
					},
					'request_id': request_id,
				})

			except Exception as e:
				app.logger.error(f"[{request_id}] Upload processing failed: {e}", exc_info=True)
				return jsonify({'error': 'Upload processing failed', 'details': str(e)}), 500

	@app.route('/export', methods=['POST'])
	def export_cleaned():
		"""Accept JSON data, clean non-destructively, and return a CSV/XLSX file.
		Body: { data: [ ... ] | {col:[...]} , format: 'csv'|'xlsx' }
		"""
		request_id = f"exp-{uuid.uuid4().hex[:8]}"
		with timed('export'):
			try:
				payload = request.get_json()
				if not isinstance(payload, dict):
					return jsonify({'error': 'Expected JSON object body'}), 400
				data = payload.get('data', payload)
				out_fmt = str(payload.get('format', 'csv')).lower()
				detector = FlexibleColumnDetector()
				df = detector.normalize_to_dataframe(data)
				cleaner = CommonCleaner(config=build_cleaner_config(APP_CONFIG, payload))
				cleaned_df, _ = cleaner.clean(df)

				# Optional AI overlook (fill blanks only)
				try:
					apply_ai_overlook(cleaned_df, payload or {}, get_llm_client())
				except Exception:
					pass

				buf = io.BytesIO()
				if out_fmt == 'xlsx':
					with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
						cleaned_df.to_excel(writer, index=False)
					mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
					filename = f"cleaned_{request_id}.xlsx"
				else:
					csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')
					buf.write(csv_bytes)
					mime = 'text/csv; charset=utf-8'
					filename = f"cleaned_{request_id}.csv"

				buf.seek(0)
				resp = make_response(send_file(buf, as_attachment=True, download_name=filename, mimetype=mime))
				return resp
			except Exception as e:
				app.logger.error(f"[{request_id}] Export failed: {e}", exc_info=True)
				return jsonify({'error': 'Export failed', 'details': str(e)}), 500

