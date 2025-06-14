# EPSS Forecasting System Overview

## System Architecture

The EPSS (Exploit Prediction Scoring System) forecasting system is a comprehensive solution for predicting vulnerability exploit probabilities. The system is organized into several key components:

### 1. Data Engineering Layer

#### 1.1 Data Ingestion and Processing
- **Raw Data Sources**
  - NVD (National Vulnerability Database) data
  - CVE List (Common Vulnerabilities and Exposures)
  - ExploitDB data
  - KEV (Known Exploited Vulnerabilities) data
  - Reddit data for social media analysis

- **Core Data Processing Pipeline**
  - **CVE Daily Wrangling (`cve_daily_wrangling.py`)**
    - Robust JSON parsing with fallback mechanisms
    - CVSS standardization across versions (V2, V3.0, V3.1, V4.0)
    - Configuration parsing for CPE data
    - Description extraction and analysis
    - Weakness information processing
    - Reference counting and analysis
    - Temporal integrity validation
    - Schema uniqueness validation
    - Feature consistency validation

  - **CSAF Integration (`csaf_merge_analysis.py`)**
    - Memory-efficient CSAF data processing
    - Chunked data analysis for large datasets
    - CSAF coverage analysis
    - Source data validation
    - Detailed CVE analysis
    - Temporal event tracking
    - Vendor and product mapping
    - Impact assessment integration

  - **Temporal Dynamics Analysis (`temporal_dynamics_eda.py`)**
    - Snapshot distribution analysis
    - Feature change tracking
    - Temporal pattern classification
    - CVE evolution analysis
    - LSTM modeling potential assessment
    - Dynamic vs static CVE categorization
    - Feature change rate analysis
    - Temporal pattern visualization

  - **Feature Engineering Pipeline**
    - **Base Features (`features.py`)**
      - Temporal feature extraction
      - CVSS score categorization
      - Vendor-specific features
      - KEV product features
      - Recent update indicators
      - Date-based features

    - **Advanced Features (`features_v2.py`)**
      - Modification history analysis
      - Inter-modification intervals
      - Recency and frequency metrics
      - Calendar-based features
      - CWE analysis
      - Vendor rate analysis
      - Cross-feature generation
      - Advanced temporal features

  - **Data Quality Controls**
    - Temporal integrity validation
    - Schema uniqueness checks
    - Feature consistency validation
    - Duplicate detection and removal
    - Data type validation
    - Missing value handling
    - CSAF data coverage validation
    - Source data cross-validation

  - **Data Storage Architecture**
    - **Directory Structure**
      - `catalogs_processed/`: Processed vulnerability data
      - `catalogs_raw/`: Raw vulnerability data
      - `processed_data/`: Final processed datasets
      - `enriched_vulnerability_catalog/`: Enriched data
      - `models/`: Trained model storage
      - `logs/`: System logs
      - `features/`: Feature storage
      - `ray/`: Distributed computing data
      - `t3_spark/`: Spark processing data

    - **Storage Formats**
      - Parquet for efficient columnar storage
      - CSV for compatibility
      - JSON for raw data
      - Binary formats for models
      - Log files for monitoring

    - **Data Organization**
      - Versioned data storage
      - Temporal snapshots
      - Feature-specific storage
      - Model-specific storage
      - Log-based tracking

    - **Access Patterns**
      - Chunked data access
      - Column-based retrieval
      - Temporal range queries
      - Feature-specific access
      - Model-based access

    - **Optimization Strategies**
      - ZSTD compression
      - Columnar storage
      - Chunked processing
      - Memory-efficient access
      - Distributed storage

### 2. Machine Learning Pipeline
- LSTM-based prediction models
- Model optimization
- Training and validation
- Prediction pipeline

### 3. System Components

#### Data Processing
- `cve_daily_wrangling.py`: Core CVE data processing
- `csaf_merge_analysis.py`: CSAF data analysis and merging
- `temporal_dynamics_eda.py`: Temporal analysis
- `validate_master_dataset.py`: Dataset validation

#### Feature Engineering
- `features.py` and `features_v2.py`: Feature processing
- `comprehensive_analyzer.py`: Feature analysis

#### Machine Learning
- `lstm_optimized.py`: Optimized LSTM implementation
- `lstm_window/`: LSTM window processing

#### Validation and Testing
- `test_equivalence.py`: Component testing
- `validate_master_dataset.py`: Dataset validation
- `rigorous_merge_eda.py`: Merge analysis

### 4. Data Storage
- `processed_data/`: Processed datasets
- `catalogs_processed/`: Processed vulnerability catalogs
- `catalogs_raw/`: Raw vulnerability data
- `models/`: Trained model storage

### 5. Monitoring and Logging
- `monitor_processes.py`: Process monitoring
- `advanced_process_monitor.py`: Advanced monitoring
- `logs/`: System logs

## System Flow

1. **Data Ingestion**
   - Raw data collection from multiple sources
   - Initial data validation and cleaning
   - JSON parsing and standardization
   - CVSS score normalization
   - CSAF data integration

2. **Data Processing**
   - CVE data processing and enrichment
   - Temporal alignment
   - Duplicate detection and removal
   - Feature extraction and standardization
   - CSAF data merging and validation
   - Temporal pattern analysis
   - Feature change tracking

3. **Feature Engineering**
   - CVSS feature processing
   - Configuration feature extraction
   - Temporal feature generation
   - CSAF feature integration
   - Feature validation and quality control
   - Dynamic feature evolution tracking
   - Pattern-based feature creation
   - Cross-feature generation
   - Advanced temporal analysis

4. **Model Training**
   - LSTM model training
   - Model optimization
   - Model validation

5. **Prediction**
   - Real-time prediction generation
   - Result validation
   - Output formatting

## Technical Stack
- Python for core processing
- PySpark for large-scale data processing
- LSTM for prediction
- Ray for distributed computing
- Various data processing libraries:
  - pandas for data manipulation
  - pyarrow for efficient storage
  - cvss for vulnerability scoring
  - python-dateutil for temporal processing
  - matplotlib and seaborn for visualization

## Data Quality Controls
1. **Input Validation**
   - JSON parsing robustness
   - Data type checking
   - Missing value handling
   - CSAF data coverage validation

2. **Processing Validation**
   - Temporal integrity checks
   - Schema uniqueness validation
   - Feature consistency validation
   - Source data cross-validation
   - Pattern-based validation

3. **Output Validation**
   - Data completeness checks
   - Feature range validation
   - Cross-reference validation
   - CSAF integration validation
   - Temporal pattern validation

## Performance Optimizations
1. **Memory Efficiency**
   - Chunked data processing
   - Parquet format usage
   - Efficient compression
   - Selective column loading
   - Sample-based analysis

2. **Processing Efficiency**
   - Parallel processing support
   - Distributed computing
   - Optimized data structures
   - Efficient algorithms
   - Pattern-based optimization

## Storage Architecture
1. **Data Organization**
   - Hierarchical directory structure
   - Versioned data storage
   - Feature-specific organization
   - Model-specific storage
   - Log-based tracking

2. **Storage Formats**
   - Parquet for efficiency
   - CSV for compatibility
   - JSON for raw data
   - Binary for models
   - Text for logs

3. **Access Patterns**
   - Chunked access
   - Column-based retrieval
   - Temporal queries
   - Feature access
   - Model access

4. **Optimization**
   - Compression
   - Columnar storage
   - Chunked processing
   - Memory efficiency
   - Distributed storage

## Next Steps
This is an initial overview that will be expanded as we analyze each component in detail. The next phase will involve:
1. Detailed analysis of remaining data processing components
2. Feature engineering pipeline analysis
3. Machine learning model architecture review
4. System integration analysis
