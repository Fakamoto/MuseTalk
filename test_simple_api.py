#!/usr/bin/env python3
"""
Test script for simple_api.py to verify the fix works
"""

import requests
import time
from pathlib import Path

def test_simple_api():
    """Test the simple API endpoint"""
    print("🧪 Testing Simple MuseTalk API...")
    
    # API endpoint
    url = "http://localhost:8000/generate"
    
    # Test files (use existing files from the project)
    video_file = "data/video/sun.mp4"
    audio_file = "data/audio/sun.wav"
    
    if not Path(video_file).exists():
        print(f"❌ Video file not found: {video_file}")
        return False
        
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        print(f"📤 Sending request to {url}")
        print(f"📁 Video: {video_file}")
        print(f"🎵 Audio: {audio_file}")
        
        # Prepare files for upload
        files = {
            'video': open(video_file, 'rb'),
            'audio': open(audio_file, 'rb')
        }
        
        data = {
            'avatar_id': 'test_simple_api'
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(url, files=files, data=data, timeout=300)
        duration = time.time() - start_time
        
        # Close files
        files['video'].close()
        files['audio'].close()
        
        if response.status_code == 200:
            print(f"✅ Success! Duration: {duration:.2f}s")
            print(f"📁 Response size: {len(response.content)} bytes")
            
            # Save the response video
            output_file = "test_output_simple_api.mp4"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"💾 Saved output video: {output_file}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple API Test")
    print("Make sure to run: uvicorn simple_api:app --host 0.0.0.0 --port 8000")
    print()
    
    success = test_simple_api()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
