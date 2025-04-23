import subprocess

# List your scripts here
scripts = [
    "data/catalogs_scrapping/CVE_scrapper.py",
    "data/catalogs_scrapping/exploitDB_scrapper.py",
    "data/catalogs_scrapping/KEV_scrapper.py",
    "data/catalogs_scrapping/metasploit_scrapper.py",
    "data/catalogs_scrapping/ZDI_scrapper.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)

    print(f"--- Output of {script} ---")
    print(result.stdout)
    if result.stderr:
        print(f"--- Errors in {script} ---")
        print(result.stderr)
    print("\n")