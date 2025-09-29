# MuseTalk Performance Monitoring

This document describes the comprehensive performance monitoring system implemented for MuseTalk APIs.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python install_monitoring_deps.py
```

### 2. Run Enhanced APIs
```bash
# Simple API with subprocess monitoring
uvicorn simple_api:app --host 0.0.0.0 --port 8000

# Realtime API with direct inference monitoring  
uvicorn realtime_api:app --host 0.0.0.0 --port 8001
```

## 📊 Monitoring Features

### System Resource Monitoring
- **CPU Usage**: Real-time CPU utilization tracking
- **Memory Usage**: RAM consumption monitoring
- **GPU Metrics**: GPU utilization, memory usage, temperature
- **Background Monitoring**: Continuous monitoring every 500ms

### Step-by-Step Timing
- **File Processing**: Upload and save timing
- **Configuration Creation**: YAML config generation
- **Inference Execution**: Main processing bottleneck analysis
- **Output Discovery**: File search and verification
- **Response Preparation**: Final response creation

### Detailed Metrics
- **File Size Analysis**: Input/output file sizes and compression ratios
- **Subprocess Metrics**: External process execution details
- **Resource Analysis**: Peak and average resource usage
- **JSON Export**: Detailed metrics saved to files

## 📈 Sample Output

```
📊 SIMPLE API PERFORMANCE ANALYSIS REPORT
================================================================================
🎯 Avatar ID: test_avatar
🎯 Mode: simple_subprocess
⏱️ Total Duration: 45.234s

📋 STEP BREAKDOWN:
  file_processing: 2.145s (4.7%)
  config_creation: 0.156s (0.3%)
  subprocess_execution: 40.234s (89.0%)
  output_discovery: 1.234s (2.7%)
  response_preparation: 1.465s (3.2%)

🚀 SUBPROCESS ANALYSIS:
  Subprocess Duration: 40.234s (89.0%)
  Return Code: 0

📁 FILE ANALYSIS:
  Input Video: 15.67 MB
  Input Audio: 2.34 MB
  Output Video: 8.45 MB
  Compression Ratio: 0.48 (output/input)

💻 CPU ANALYSIS:
  Average CPU Usage: 78.5%
  Peak CPU Usage: 95.2%
  CPU Cores: 8

🧠 MEMORY ANALYSIS:
  Average Memory Usage: 12.45 GB
  Peak Memory Usage: 15.67 GB

🎮 GPU ANALYSIS:
  Average GPU Utilization: 85.3%
  Peak GPU Utilization: 98.1%
  Average GPU Memory: 78.2%
  Peak GPU Memory: 89.4%
```

## 🔧 Configuration

### Log Files
- **Console Output**: Real-time monitoring in terminal
- **Log File**: `simple_api_performance.log` (detailed logs)
- **Metrics JSON**: `simple_api_metrics_{avatar_id}_{timestamp}.json`

### Monitoring Frequency
- **System Resources**: Every 500ms
- **Step Timing**: Per operation
- **File Metrics**: Per request

## 🎯 Bottleneck Identification

### Common Bottlenecks
1. **Subprocess Execution** (89% of time)
   - GPU inference processing
   - Model loading and computation
   - Video generation

2. **File I/O Operations** (5-10% of time)
   - Upload processing
   - Output file discovery
   - Response preparation

3. **System Resources**
   - GPU memory constraints
   - CPU utilization limits
   - RAM availability

### Optimization Strategies
- **GPU Optimization**: Monitor GPU utilization, adjust batch sizes
- **Memory Management**: Track memory usage, optimize model loading
- **File Processing**: Optimize upload/download speeds
- **Subprocess Tuning**: Adjust timeout, batch size, GPU settings

## 📁 File Structure

```
MuseTalk/
├── simple_api.py                    # Enhanced with monitoring
├── realtime_api.py                  # Enhanced with monitoring  
├── install_monitoring_deps.py       # Dependency installer
├── PERFORMANCE_MONITORING.md        # This documentation
├── simple_api_performance.log       # Log file
└── simple_api_metrics_*.json        # Detailed metrics
```

## 🛠️ Troubleshooting

### Missing Dependencies
```bash
# Install monitoring packages
pip install psutil GPUtil

# Or use the installer
python install_monitoring_deps.py
```

### No GPU Monitoring
- Install `GPUtil`: `pip install GPUtil`
- Ensure NVIDIA drivers are installed
- Check GPU availability: `nvidia-smi`

### High Memory Usage
- Monitor memory metrics in logs
- Consider reducing batch size
- Check for memory leaks in subprocess

## 📊 Metrics Analysis

### Key Performance Indicators (KPIs)
- **Total Duration**: Overall request time
- **Subprocess Percentage**: How much time spent in inference
- **GPU Utilization**: GPU efficiency
- **Memory Peak**: Maximum RAM usage
- **File Compression**: Output efficiency

### Performance Trends
- Compare metrics across different requests
- Identify performance regressions
- Track optimization improvements
- Monitor resource usage patterns

## 🚀 Advanced Usage

### Custom Monitoring
```python
# Access the performance monitor
from simple_api import performance_monitor

# Start custom monitoring
performance_monitor.start_monitoring("custom_avatar")

# Log custom steps
performance_monitor.log_step("custom_step", duration, details)

# Stop and get metrics
performance_monitor.stop_monitoring()
```

### Metrics Export
```python
# Metrics are automatically saved to JSON files
# Access programmatically:
metrics = performance_monitor.metrics
print(f"Total duration: {metrics['total_duration']:.2f}s")
```

## 📝 Log Levels

- **INFO**: Standard monitoring output
- **ERROR**: Failed operations
- **WARNING**: Non-critical issues
- **DEBUG**: Detailed step information

## 🔍 Monitoring Best Practices

1. **Regular Monitoring**: Run performance tests regularly
2. **Baseline Metrics**: Establish performance baselines
3. **Resource Limits**: Set appropriate timeouts and limits
4. **Cleanup**: Monitor disk space for log files
5. **Analysis**: Review metrics for optimization opportunities

## 🆘 Support

For issues with performance monitoring:
1. Check log files for error messages
2. Verify all dependencies are installed
3. Ensure sufficient system resources
4. Review subprocess execution logs
