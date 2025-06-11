#!/usr/bin/env python3
import psutil
import time
import os
from datetime import datetime, timedelta

def monitor_downloads():
    """
    Advanced monitoring for download processes with detailed status
    """
    keywords = ['reddit', 'cve_historical', 'submissions', 'comments']
    
    print("🔍 Advanced Download Process Monitor")
    print("=" * 70)
    
    while True:
        print(f'\n📊 Status at {datetime.now().strftime("%H:%M:%S")}')
        print('-' * 70)
        
        found_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time', 'status']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    
                    if any(keyword in cmdline.lower() for keyword in keywords):
                        runtime = datetime.now() - datetime.fromtimestamp(proc.info['create_time'])
                        memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                        
                        process_info = {
                            'pid': proc.info['pid'],
                            'memory_mb': memory_mb,
                            'runtime': runtime,
                            'status': proc.info['status'],
                            'cmdline': cmdline
                        }
                        found_processes.append(process_info)
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if found_processes:
            for proc in found_processes:
                status_icon = "✅" if proc['status'] == 'running' else "⚠️"
                print(f'{status_icon} PID {proc["pid"]} - {proc["memory_mb"]:.1f}MB - Runtime: {proc["runtime"]}')
                
                # Identify process type
                if 'submissions' in proc['cmdline'].lower():
                    print(f'   📝 Reddit Submissions Scraper')
                elif 'comments' in proc['cmdline'].lower():
                    print(f'   💬 Reddit Comments Scraper')
                elif 'cve_historical' in proc['cmdline'].lower():
                    print(f'   🔐 CVE Historical Reconstructor')
                else:
                    print(f'   🔄 Download Process')
                
                print(f'   Status: {proc["status"]}')
                print()
        else:
            print('❌ No download processes found running')
        
        # Check log files for recent activity
        log_files = ['logs/bot.log', 'output.log']
        print('\n📄 Recent Log Activity:')
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    stat = os.stat(log_file)
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    time_diff = datetime.now() - mod_time
                    
                    if time_diff < timedelta(minutes=5):
                        status = "🟢 Active"
                    elif time_diff < timedelta(minutes=30):
                        status = "🟡 Recent" 
                    else:
                        status = "🔴 Stale"
                    
                    print(f'   {status} {log_file}: Last modified {time_diff} ago')
                except:
                    print(f'   ❓ {log_file}: Unable to check')
            else:
                print(f'   ❌ {log_file}: Not found')
        
        print('\n' + '=' * 70)
        print('Press Ctrl+C to stop monitoring...')
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print('\n👋 Monitoring stopped')
            break

if __name__ == "__main__":
    try:
        monitor_downloads()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...") 