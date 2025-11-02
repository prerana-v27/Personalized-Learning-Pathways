"""Convert JSON file(s) in Data/raw into a single CSV in Data/processed.

Defaults
  Input:  ../raw/course_extractions_20.json
  Output: ../processed/NPTEL.csv

Features
  - Accepts JSON arrays, a single JSON object, or newline-delimited JSON (ndjson)
  - Flattens nested objects into dot-separated column names
  - Joins simple lists into a pipe-separated string; complex lists are JSON-dumped
  - CLI to specify input and output paths

Usage
  python jsonToCsv.py            # uses default input and output
  python jsonToCsv.py -i path -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List


def flatten(value: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
	"""Flatten a JSON object (dict) into a single-level dict with dotted keys.

	Special handling:
	- Lists of primitives are joined with '|'
	- Lists with non-primitives are JSON-dumped
	- Non-dict values are returned as-is under parent_key
	"""

	def is_primitive(v):
		return v is None or isinstance(v, (str, int, float, bool))

	out: Dict[str, Any] = {}

	if isinstance(value, dict):
		for k, v in value.items():
			new_key = f"{parent_key}{sep}{k}" if parent_key else k
			if isinstance(v, dict):
				out.update(flatten(v, new_key, sep=sep))
			elif isinstance(v, list):
				# if list of primitives, join; else dump
				if all(is_primitive(x) for x in v):
					out[new_key] = "|".join("" if x is None else str(x) for x in v)
				else:
					try:
						out[new_key] = json.dumps(v, ensure_ascii=False)
					except Exception:
						out[new_key] = str(v)
			else:
				out[new_key] = v

	else:
		# value itself is primitive or list
		if parent_key:
			if isinstance(value, list):
				if all(is_primitive(x) for x in value):
					out[parent_key] = "|".join("" if x is None else str(x) for x in value)
				else:
					out[parent_key] = json.dumps(value, ensure_ascii=False)
			else:
				out[parent_key] = value
		else:
			# No parent key and not a dict: return empty mapping
			out[""] = value

	return out


def read_json_records(path: str) -> List[Dict[str, Any]]:
	"""Read a JSON file and return a list of records (dicts).

	Tries these formats in order:
	- a JSON array of objects
	- a single JSON object (wrapped into a list)
	- newline-delimited JSON (one JSON object per line)
	"""

	if not os.path.exists(path):
		raise FileNotFoundError(f"Input file not found: {path}")

	with open(path, "r", encoding="utf-8") as f:
		text = f.read()

	text = text.strip()
	if not text:
		return []

	# First try to parse whole file as JSON
	try:
		data = json.loads(text)
	except json.JSONDecodeError:
		# Try ndjson: parse line by line
		records: List[Dict[str, Any]] = []
		for line in text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				# Skip malformed lines but warn
				print(f"Warning: skipping malformed JSON line: {line[:80]}")
				continue
			if isinstance(obj, dict):
				records.append(obj)
			else:
				# If the line is a primitive or list, wrap into dict
				records.append({"value": obj})
		return records

	# data parsed successfully
	if isinstance(data, list):
		# ensure each element is a dict; if not, wrap
		out: List[Dict[str, Any]] = []
		for el in data:
			if isinstance(el, dict):
				out.append(el)
			else:
				out.append({"value": el})
		return out

	if isinstance(data, dict):
		# If dict contains a top-level list under a common key, try to find it
		# Common keys: 'data', 'courses', 'results'
		for key in ("data", "courses", "results", "items"):
			if key in data and isinstance(data[key], list):
				items = data[key]
				out = []
				for el in items:
					if isinstance(el, dict):
						out.append(el)
					else:
						out.append({"value": el})
				return out
		# Otherwise treat the dict itself as a single record
		return [data]

	# Fallback
	return [{"value": data}]


def records_to_csv(records: Iterable[Dict[str, Any]], out_path: str) -> None:
	records = list(records)
	if not records:
		print("No records to write. Exiting.")
		return

	# Flatten all records and collect fieldnames
	flat_records: List[Dict[str, Any]] = []
	fieldnames: List[str] = []
	fieldset = set()

	for rec in records:
		flat = flatten(rec)
		flat_records.append(flat)
		for k in flat.keys():
			if k not in fieldset:
				fieldset.add(k)
				fieldnames.append(k)

	# Ensure output directory exists
	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	# Write CSV
	with open(out_path, "w", encoding="utf-8", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		for frec in flat_records:
			# Ensure all fields exist
			row = {k: ("" if frec.get(k) is None else frec.get(k)) for k in fieldnames}
			writer.writerow(row)


def main(argv: List[str] | None = None) -> int:
	argv = argv if argv is not None else sys.argv[1:]
	parser = argparse.ArgumentParser(description="Convert JSON to CSV (flattens nested fields)")
	script_dir = os.path.dirname(os.path.abspath(__file__))
	default_input = os.path.abspath(os.path.join(script_dir, "..", "raw", "course_extractions_20.json"))
	default_output = os.path.abspath(os.path.join(script_dir, "..", "processed", "NPTEL.csv"))

	parser.add_argument("-i", "--input", default=default_input, help=f"Input JSON file (default: {default_input})")
	parser.add_argument("-o", "--output", default=default_output, help=f"Output CSV file (default: {default_output})")
	args = parser.parse_args(argv)

	try:
		print(f"Reading JSON from: {args.input}")
		records = read_json_records(args.input)
		print(f"Parsed {len(records)} records. Writing CSV to: {args.output}")
		records_to_csv(records, args.output)
		print("Done.")
		return 0
	except FileNotFoundError as e:
		print(f"Error: {e}")
		return 2
	except Exception as e:
		print(f"Unexpected error: {e}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())

