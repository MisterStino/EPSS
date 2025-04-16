import subprocess
import os
import pandas as pd
import shutil
import time
import errno
from pathlib import Path
from subprocess import Popen, PIPE

def remove_directory_with_retry(directory, max_retries=5, delay=2):
    """Remove a directory, handling Windows-specific attributes and errors."""
    # First, remove Hidden and Read-Only attributes recursively
    try:
        subprocess.run(
            ["attrib", "-h", "-r", f"{directory}\\*.*", "/s", "/d"],
            check=True,
            shell=True,  # Required for attrib on Windows
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Cleared Hidden and Read-Only attributes for {directory}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clear attributes for {directory}: {e.stderr.decode()}")

    def handle_error(func, path, exc_info):
        exc_type, exc_value, _ = exc_info
        if (func in (os.rmdir, os.remove, os.unlink) and 
            (exc_value.errno == errno.EACCES or exc_value.errno == errno.ENOTEMPTY)):
            time.sleep(1)
            try:
                if func == os.rmdir:
                    for root, dirs, files in os.walk(path, topdown=False):
                        for file in files:
                            os.unlink(os.path.join(root, file))
                        for dir in dirs:
                            os.rmdir(os.path.join(root, dir))
                    os.rmdir(path)
                else:
                    os.unlink(path)
                print(f"Retried and removed {path}")
            except OSError as e:
                print(f"Failed to retry removal of {path}: {e}")
        else:
            raise

    for attempt in range(max_retries):
        try:
            shutil.rmtree(directory, onerror=handle_error)
            print(f"Deleted cloned repository {directory}")
            return True
        except OSError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed to delete {directory}: {e}")
            time.sleep(delay)
    
    # Fallback: Use CMD rmdir /s /q if rmtree fails
    try:
        subprocess.run(["rmdir", "/s", "/q", directory], check=True, shell=True)
        print(f"Force-deleted {directory} using rmdir /s /q")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to force-delete {directory} with rmdir: {e}. Please delete manually.")
        return False

def get_commit_date(repo_url, commit_sha, clone_dir="temp_repo"):
    if not repo_url.endswith('.git'):
        repo_url = f"{repo_url}.git"
        print(f"Adjusted repo_url to: {repo_url}")

    if os.path.exists(clone_dir):
        remove_directory_with_retry(clone_dir)

    clone_process = None
    try:
        clone_process = Popen(["git", "clone", repo_url, clone_dir], stdout=PIPE, stderr=PIPE)
        stdout, stderr = clone_process.communicate()
        if clone_process.returncode != 0:
            print(f"Failed to clone {repo_url}: {stderr.decode()}")
            return None
        print(f"Cloned {repo_url} to {clone_dir}")
    except Exception as e:
        print(f"Exception during cloning {repo_url}: {e}")
        return None
    finally:
        if clone_process:
            if clone_process.stdout:
                clone_process.stdout.close()
            if clone_process.stderr:
                clone_process.stderr.close()
            clone_process.terminate()
            clone_process.wait()
            time.sleep(1)

    os.chdir(clone_dir)

    check_process = None
    try:
        check_process = Popen(
            ["git", "cat-file", "-t", commit_sha],
            stdout=PIPE,
            stderr=PIPE,
            text=True
        )
        stdout, stderr = check_process.communicate()
        if check_process.returncode != 0 or stdout.strip() != "commit":
            print(f"SHA {commit_sha} not found in {repo_url}: {stderr}")
            return None
        print(f"SHA {commit_sha} exists in repository")
    except Exception as e:
        print(f"Exception checking SHA {commit_sha}: {e}")
        return None
    finally:
        if check_process:
            if check_process.stdout:
                check_process.stdout.close()
            if check_process.stderr:
                check_process.stderr.close()
            check_process.terminate()
            check_process.wait()

    show_process = None
    commit_date = None
    try:
        show_process = Popen(
            ["git", "show", "-s", "--format=%ci", commit_sha],
            stdout=PIPE,
            stderr=PIPE,
            text=True
        )
        stdout, stderr = show_process.communicate()
        if show_process.returncode == 0:
            commit_date = stdout.strip()
            print(f"Retrieved commit date for {commit_sha}: {commit_date}")
        else:
            print(f"Failed to retrieve date for SHA {commit_sha} in {repo_url}: {stderr}")
    except Exception as e:
        print(f"Exception during git show for {repo_url}: {e}")
    finally:
        if show_process:
            if show_process.stdout:
                show_process.stdout.close()
            if show_process.stderr:
                show_process.stderr.close()
            show_process.terminate()
            show_process.wait()
            time.sleep(1)
        os.chdir("..")
        time.sleep(1)

    return commit_date

def process_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Failed to read input CSV {input_file}: {e}")
        return

    if not os.path.exists(output_file):
        pd.DataFrame(columns=["cve_id", "count", "repo_url", "sha", "commit_date"]).to_csv(output_file, index=False)

    for index, row in df.iterrows():
        cve_id = row["cve_id"]
        count = row["count"]
        repo_url = row["repo_url"]
        sha = row["sha"]

        print(f"\nProcessing {cve_id} - {repo_url} - {sha}")
        
        commit_date = get_commit_date(repo_url, sha)
        
        result = pd.DataFrame([{
            "cve_id": cve_id,
            "count": count,
            "repo_url": row["repo_url"],
            "sha": sha,
            "commit_date": commit_date if commit_date else "Not Found"
        }])
        
        result.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Appended result for {cve_id} to {output_file}")

        remove_directory_with_retry("temp_repo")

    print(f"\nProcessing complete. All results saved to {output_file}")

# Example usage
input_file = "cve_results.csv"  # Replace with your input CSV file path
output_file = "output.csv"  # Replace with your desired output CSV file path
process_csv(input_file, output_file)