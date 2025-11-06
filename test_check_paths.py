#!/usr/bin/env python3
"""
Test script to check paths in RunPod endpoint
"""

import requests
import json

# Your RunPod endpoint
ENDPOINT_URL = "https://api.runpod.ai/v2/ghkm8uyf5fqv4p/run"
STATUS_URL = "https://api.runpod.ai/v2/ghkm8uyf5fqv4p/status"
API_KEY = "rpa_CA6IUT83T2H7JZG0TPO9Z4C6NGCZMPDVUCHV3FL91s6a2x"

def check_paths():
    """Check what paths exist in the container"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "action": "check_paths"
        }
    }
    
    print("🔍 Checking paths in RunPod container...")
    print("=" * 60)
    
    # Start async job
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get("id")
        print(f"\n📝 Job started: {job_id}")
        print("⏳ Waiting for completion...")
        
        # Poll for results
        import time
        for i in range(60):
            time.sleep(2)
            status_response = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=10)
            if status_response.status_code == 200:
                status_result = status_response.json()
                status_text = status_result.get("status", "UNKNOWN")
                print(f"   Status: {status_text}")
                
                if status_text in ["COMPLETED", "FAILED"]:
                    print("\n✅ Final result:")
                    print(json.dumps(status_result, indent=2))
                    
                    if "output" in status_result and "paths" in status_result["output"]:
                        print("\n📁 Path Analysis:")
                        print("=" * 60)
                        for path, info in status_result["output"]["paths"].items():
                            print(f"\n📍 {path}")
                            print(f"   Exists: {info.get('exists', False)}")
                            print(f"   Is Link: {info.get('is_link', False)}")
                            if info.get('link_target'):
                                print(f"   Points to: {info['link_target']}")
                            if info.get('contents'):
                                print(f"   Contents: {info['contents']}")
                            if info.get('error'):
                                print(f"   ❌ Error: {info['error']}")
                    break
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    check_paths()

