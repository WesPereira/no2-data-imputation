# #!/bin/bash

# Change the file datasets.yml to select
# bands, datasets and output file path.
# download_data.py

# Apply filter to eliminate invalid data
./format_csvs.sh

# Convert csvs to time series in range of 1 year
./csv2numpy.sh

# Divide data into train, val and test
./normalize_ds.sh