"""Fetch the DAFD 3.0 single-emulsion dataset from OSF (project 938rs).

The raw data is not committed (it is other people's dataset; cite it, don't
vendor it).  This pulls the two files the study needs into normalization/data/.

    python normalization/fetch_data.py
"""
import os
import urllib.request

HERE = os.path.dirname(__file__)
DEST = os.path.join(HERE, "data")
os.makedirs(DEST, exist_ok=True)

# OSF download endpoints (project https://osf.io/938rs/)
FILES = {
    "Comprehensive_normalized.xlsx": "https://osf.io/download/68f43/",
    "Generalizability_data_normalized.xlsx": "https://osf.io/download/zb5mu/",
    "Raw_SE_dataset.xlsx": "https://osf.io/download/awvq3/",
}


def main():
    for name, url in FILES.items():
        out = os.path.join(DEST, name)
        if os.path.exists(out):
            print(f"have {name}")
            continue
        print(f"fetching {name} <- {url}")
        urllib.request.urlretrieve(url, out)
        print(f"  {os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
