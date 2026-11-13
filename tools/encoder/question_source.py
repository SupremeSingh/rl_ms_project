"""Standard-library adapters for immutable package and historical Numina corpora."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(path):
    path=Path(path).resolve()
    for name in ('run.json','identity.json','manifest.json'):
        meta=path.parent/name
        if not meta.exists():continue
        record=read(meta)
        if not isinstance(record,dict) or 'questions_sha256' not in record:continue
        if record['questions_sha256']!=digest(path):
            raise ValueError(f'Question hash differs from active provenance: {meta}')
        return dict(record,metadata_path=str(meta),metadata_sha256=digest(meta))
    raise ValueError(f'No matching active question provenance for {path}')


def load_questions(path):
    obj=read(path);rows=obj if isinstance(obj,list) else obj.get('questions') if isinstance(obj,dict) else None
    if not isinstance(rows,list):raise ValueError('Expected question list or questions wrapper')
    result=[];ids=set()
    for q in rows:
        if q['id'] in ids:raise ValueError('Duplicate question ID')
        ids.add(q['id'])
        if not isinstance(q.get('question'),str) or not q['question'].strip():raise ValueError('Empty question')
        refs=[str(q[k]) for k in ('reference','ground_truth') if k in q]
        if not refs or any(v!=refs[0] for v in refs):raise ValueError('Missing or conflicting numeric reference fields')
        result.append(dict(id=q['id'],question=q['question'],reference=refs[0],split=q['split']))
    return result


def main_budget(rows):
    counts=Counter(q['split'] for q in rows)
    for target in (dict(train=10000,val=1000,test=2000),dict(train=7800,val=2600,test=2600)):
        if all(counts[k]>=v for k,v in target.items()):return target
    raise ValueError(f'Existing splits cannot supply a supported 13,000-question budget: {dict(counts)}')
