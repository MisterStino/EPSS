# epss-data
## Project Setup

### Requirements
Make sure you have Python installed. Then, create and activate a virtual environment named `epss-env`:
```bash
python -m venv epss-env
source epss-env/bin/activate  # On Windows use `epss-env\Scripts\activate`
```
Next, install the required packages using:
```bash
pip install -r requirements.txt
```

### Files
- `get_epps_data.py`: Script to fetch EPSS data.
- `inspect.py`: Script to inspect the fetched data.

### Usage
1. **Fetch Data**: Run the following command to fetch the EPSS data:
    ```bash
    python -m get_epps_data
    ```
2. **Inspect Data**: After fetching the data, you can inspect it using:
    ```bash
    python -m inspect
    ```

### Notes
- Ensure you have an active internet connection to fetch the data.
- Modify the scripts as needed to fit your specific use case.



### Additional Information
- **Data Storage**: The fetched data will be stored in the `data` directory. Ensure this directory exists or modify the script to create it if it doesn't.
- **Logging**: The scripts include logging functionality to help you debug any issues. We write CVE days that could not be downloaded to the temp_error.json. Should later inspect it. 

Also we should maybe verify that everything went correct with sample checks.




