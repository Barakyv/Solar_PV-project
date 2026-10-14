"""Stage 6b: download and process GFS 1-degree forecasts at 30N 35E (PROJECT_MASTER.md §6.12).

Source: https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.1p00.fXXX
Only the needed GRIB messages are fetched, using the .idx byte offsets and HTTP Range requests.
Run choice: 12Z of the previous UTC day, allowed only if init + 6 h <= T0 (test T-17).
DSWRF is a 6-hour block average and must be de-averaged into 3-hour means.
Output columns use the prefix fcst_gfs_ and one column set per horizon.
"""
