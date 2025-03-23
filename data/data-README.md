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