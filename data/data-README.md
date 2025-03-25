Currently database is set up as following:
+----------------+------------+-------+----------+----------+------+
|      cve       |    date    | epss  | feature1 | feature2 | ...  |
+----------------+------------+-------+----------+----------+------+
| CVE-2021-42013 | 2022-02-04 | 0.816 |   ...    |   ...    | ...  |
| CVE-2021-42013 | 2022-02-05 | 0.820 |   ...    |   ...    | ...  |
| CVE-2021-1732  | 2022-02-04 | 0.091 |   ...    |   ...    | ...  |
|       ...      |    ...     |  ...  |   ...    |   ...    | ...  |
+----------------+------------+-------+----------+----------+------+

- Each row still represents a single observation for a CVE at a given date.

## Columns:

cve: The identifier of the vulnerability (e.g., "CVE-2021-42013").

date: The calendar date corresponding to that observation. This is extracted from the filename and represents the day the EPSS score was recorded.

epss: The EPSS score, which is your target variable.

## Ordering:

The data is sorted by cve and then by date so that each CVE’s time-series is in chronological order.

## Primary Key (Conceptually):

A composite key of (cve, date) uniquely identifies each row.


## Note on data storage design and file type. 
The current set up is for dataset development purposes in the initial phase. Depending on when we run into issues we can convert to more efficient approaches. .csv is chosen here because it is easy to interpret, view and commonly used. We can easily check manually if we are getting expected results. 

A long table is chosen because it allows for relatively easily addition of features and do inspection and cleaning. 

If we run into memory or performance issues I want to move to using .parquet files.

## for training the models and testing:
- we create a preprocessing step or datapipeline:
- here we either create a set of preprocessed files on disk, where each chunk contains data for subset of cve. Or we do it just in time with a streaming approach. 
- we might store in parquet. 
- add padding to time series sequences.
- oflload computations to gpu(s).
- Main idea: find a way to handle the big amount of data and have fast and efficient computations. 