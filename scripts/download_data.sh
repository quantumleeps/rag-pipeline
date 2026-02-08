#!/usr/bin/env bash
# Download EPA water treatment corpus for RAG chunking comparison
# All documents are publicly available from epa.gov

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DATA_DIR"

# filename|url pairs
DOCS="
01-understanding-sdwa.pdf|https://www.epa.gov/sites/default/files/2015-04/documents/epa816f04030.pdf
02-npdwr-complete-table.pdf|https://19january2021snapshot.epa.gov/sites/static/files/2016-06/documents/npwdr_complete_table.pdf
03-pfas-treatment-options.pdf|https://www.epa.gov/system/files/documents/2024-04/pfas-npdwr_fact-sheet_treatment_4.8.24.pdf
04-pfas-bat-ssct.pdf|https://www.epa.gov/system/files/documents/2024-04/2024-final-pfas-bat-ssct_final-508.pdf
05-dbpr-plain-english-guide.pdf|https://www.epa.gov/sites/default/files/2020-06/documents/dbpr_plain_english_guide_final_508.pdf
06-disinfection-profiling-benchmarking.pdf|https://www.epa.gov/system/files/documents/2022-02/disprof_bench_3rules_final_508.pdf
07-ozone-disinfection-factsheet.pdf|https://www.epa.gov/sites/default/files/2015-06/documents/ozon.pdf
08-chemistry-ozone-disinfection.pdf|https://www.epa.gov/sites/default/files/2016-12/documents/mcchristian-ozone.pdf
09-package-plants-factsheet.pdf|https://www.epa.gov/sites/default/files/2015-06/documents/package_plant.pdf
10-sbr-factsheet.pdf|https://www.epa.gov/system/files/documents/2022-10/sequencing-batch-reactors-factsheet.pdf
11-emerging-technologies-wastewater.pdf|https://www.epa.gov/sites/default/files/2019-02/documents/emerging-tech-wastewater-treatment-management.pdf
12-wqs-handbook-ch3-criteria.pdf|https://www.epa.gov/sites/default/files/2014-10/documents/handbook-chapter3.pdf
"

count=0
echo "Downloading EPA documents to $DATA_DIR..."

echo "$DOCS" | while IFS='|' read -r filename url; do
  [ -z "$filename" ] && continue
  filepath="$DATA_DIR/$filename"

  if [ -f "$filepath" ]; then
    echo "  [skip] $filename (already exists)"
  else
    echo "  [download] $filename"
    curl -sL -o "$filepath" "$url"
  fi
done

echo "Done. $(ls "$DATA_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs in $DATA_DIR"
