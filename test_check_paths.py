#!/usr/bin/env python3
"""
Test script to check paths in a RunPod endpoint.

Requires environment variables (see .env.example):
  RUNPOD_API_KEY
  RUNPOD_ENDPOINT_ID
"""

import json
import os
import sys
import time

import requests

ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()

if not ENDPOINT_ID or not API_KEY:
    print(
        "Missing RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY.\n"
        "Set them in your environment or .env file (see .env.example).",
        file=sys.stderr,
    )
    sys.exit(1)

ENDPOINT_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"


def check_paths() -> None:
    """Check what paths exist in the container."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "action": "check_paths",
        }
    }

    print("Checking paths in RunPod container...")
    print("=" * 60)

    response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    result = response.json()
    job_id = result.get("id")
    print(f"\nJob started: {job_id}")
    print("Waiting for completion...")

    for _ in range(60):
        time.sleep(2)
        status_response = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=10)
        if status_response.status_code != 200:
            continue

        status_result = status_response.json()
        status_text = status_result.get("status", "UNKNOWN")
        print(f"   Status: {status_text}")

        if status_text in ["COMPLETED", "FAILED"]:
            print("\nFinal result:")
            print(json.dumps(status_result, indent=2))

            output = status_result.get("output") or {}
            paths = output.get("paths")
            if paths:
                print("\nPath analysis:")
                print("=" * 60)
                for path, info in paths.items():
                    print(f"\n{path}")
                    print(f"   Exists: {info.get('exists', False)}")
                    print(f"   Is Link: {info.get('is_link', False)}")
                    if info.get("link_target"):
                        print(f"   Points to: {info['link_target']}")
                    if info.get("contents"):
                        print(f"   Contents: {info['contents']}")
                    if info.get("error"):
                        print(f"   Error: {info['error']}")
            break


if __name__ == "__main__":
    check_paths()
