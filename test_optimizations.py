#!/usr/bin/env python3
"""
Performance testing script to validate optimizations
"""

import requests
import time
import json
from pathlib import Path

def test_performance():
    """Test performance with different batch sizes"""
    print("🧪 Testing Performance Optimizations...")
    
    # Test configurations
    test_configs = [
        {"batch_size": 16, "name": "baseline"},
        {"batch_size": 32, "name": "optimized_32"},
        {"batch_size": 64, "name": "optimized_64"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🚀 Testing batch size: {config['batch_size']}")
        
        # Update batch size in simple_api.py (this would need to be done manually)
        print(f"⚠️  Note: Manually update BATCH_SIZE = {config['batch_size']} in simple_api.py")
        
        # Test files
        video_file = "data/video/sun.mp4"
        audio_file = "data/audio/sun.wav"
        
        if not Path(video_file).exists() or not Path(audio_file).exists():
            print("❌ Test files not found, skipping...")
            continue
        
        try:
            # Send request
            url = "http://localhost:8000/generate"
            files = {
                'video': open(video_file, 'rb'),
                'audio': open(audio_file, 'rb')
            }
            data = {'avatar_id': f"test_{config['name']}"}
            
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=600)
            duration = time.time() - start_time
            
            files['video'].close()
            files['audio'].close()
            
            if response.status_code == 200:
                results[config['name']] = {
                    'batch_size': config['batch_size'],
                    'duration': duration,
                    'success': True
                }
                print(f"✅ Success: {duration:.2f}s")
            else:
                print(f"❌ Failed: {response.status_code}")
                results[config['name']] = {
                    'batch_size': config['batch_size'],
                    'duration': duration,
                    'success': False,
                    'error': response.text
                }
                
        except Exception as e:
            print(f"💥 Error: {e}")
            results[config['name']] = {
                'batch_size': config['batch_size'],
                'duration': 0,
                'success': False,
                'error': str(e)
            }
    
    # Save results
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("=" * 50)
    for name, result in results.items():
        if result['success']:
            print(f"{name:20} | {result['batch_size']:2d} | {result['duration']:8.2f}s")
        else:
            print(f"{name:20} | {result['batch_size']:2d} | FAILED")
    
    # Find best configuration
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        best = min(successful_results.items(), key=lambda x: x[1]['duration'])
        print(f"\n🏆 BEST CONFIGURATION: {best[0]} (batch_size={best[1]['batch_size']}, {best[1]['duration']:.2f}s)")
        
        # Calculate speedup
        baseline = successful_results.get('baseline', {}).get('duration', 0)
        if baseline > 0:
            speedup = baseline / best[1]['duration']
            print(f"🚀 SPEEDUP: {speedup:.2f}x faster than baseline")

if __name__ == "__main__":
    print("🚀 Performance Optimization Test")
    print("Make sure to run: uvicorn simple_api:app --host 0.0.0.0 --port 8000")
    print()
    
    test_performance()
