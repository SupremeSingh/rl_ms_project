"""JSON protocol for a separately installed, pinned Math-Verify interpreter."""
from contextlib import redirect_stdout
import json
import sys
from .grading import grade, provenance


def main():
    rows=json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        identity=provenance()
        results=[grade(row['response'],row['reference']) for row in rows]
    json.dump(dict(provenance=identity,results=results),sys.stdout,allow_nan=False)


if __name__=='__main__': main()
