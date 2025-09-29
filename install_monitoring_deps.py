#!/usr/bin/env python3
"""
Install performance monitoring dependencies for MuseTalk APIs
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🔧 Installing performance monitoring dependencies...")
    
    # Required packages for performance monitoring
    packages = [
        "psutil",      # System and process monitoring
        "GPUtil",      # GPU monitoring
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"  Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("✅ All monitoring dependencies installed successfully!")
        print("\n🚀 You can now run the enhanced APIs with performance monitoring:")
        print("  - simple_api.py (with subprocess monitoring)")
        print("  - realtime_api.py (with direct inference monitoring)")
    else:
        print("⚠️  Some packages failed to install. Monitoring will be limited.")
        print("   You can still use the APIs, but some metrics may not be available.")

if __name__ == "__main__":
    main()
