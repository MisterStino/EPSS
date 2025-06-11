#!/usr/bin/env python3
import psutil
import time
from datetime import datetime, timedelta

def get_process_info():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if any(keyword in cmdline for keyword in ['reddit', 'cve_historical', 'submissions', 'comments']):
                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                    runtime = datetime.now() - create_time
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    
                    processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'runtime': runtime,
                        'memory_mb': memory_mb,
                        'create_time': create_time
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def format_timedelta(td):
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def main():
    print("🔍 Monitoring Running Download Processes")
    print("=" * 60)
    
    while True:
        processes = get_process_info()
        
        print(f"\n📊 Status at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        if not processes:
            print("❌ No download processes found running")
        else:
            for proc in processes:
                script_name = "Unknown"
                if "submissions" in proc['cmdline']:
                    script_name = "🔗 Reddit Submissions Scraper"
                elif "comments" in proc['cmdline']:
                    script_name = "💬 Reddit Comments Scraper"
                elif "cve_historical" in proc['cmdline']:
                    script_name = "🛡️  CVE Historical Reconstructor"
                
                print(f"{script_name}")
                print(f"   PID: {proc['pid']}")
                print(f"   Runtime: {format_timedelta(proc['runtime'])}")
                print(f"   Memory: {proc['memory_mb']:.1f} MB")
                print(f"   Started: {proc['create_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        
        # Check log file for recent activity
        try:
            with open('logs/bot.log', 'r') as f:
                lines = f.readlines()
                if lines:
                    latest = lines[-1].strip()
                    print(f"📝 Latest log entry:")
                    print(f"   {latest}")
        except FileNotFoundError:
            print("📝 No log file found")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring...")
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.") 