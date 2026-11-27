"""Make short-lived download links (id <TAB> ext <TAB> url) for S3 objects, on the owner's machine; credentials never leave it.

Reads lines "id<TAB>ext<TAB>s3://bucket/key" and writes the links file. Never prints a link.
    /home/yassine/inkscript/.venv/bin/python scripts/presign_links.py keys.tsv links.tsv --hours 3
"""
import argparse

import awscli  # noqa: F401  (makes its bundled botocore importable)
import botocore.session
from botocore.config import Config

ap = argparse.ArgumentParser(); ap.add_argument("keys"); ap.add_argument("out"); ap.add_argument("--hours", type=float, default=3)
a = ap.parse_args()
s3 = botocore.session.get_session().create_client("s3", config=Config(signature_version="s3v4"))
n = 0
with open(a.out, "w") as f:
    for l in open(a.keys):
        if not l.strip(): continue
        i, ext, uri = l.rstrip("\n").split("\t")
        bucket, key = uri[5:].split("/", 1)
        f.write(f"{i}\t{ext}\t{s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=int(a.hours * 3600))}\n"); n += 1
print(f"{n} links written")
