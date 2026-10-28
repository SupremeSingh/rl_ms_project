"""Pinned dataset loading and outcome-independent question-family splits."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from .numbers import parse_number
from .config import immutable_revision
from .references import reference_decision
from .storage import sha256


from .text import family_key


def load_rows(config):
    path=Path(config.dataset)
    if path.is_file():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows,dict(kind='jsonl',path=str(path.resolve()),sha256=sha256(path))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(config.dataset,revision=config.dataset_revision).sha
    if not immutable_revision(revision): raise ValueError('Dataset revision did not resolve to an immutable commit SHA')
    data=load_dataset(config.dataset,config.dataset_config,split=config.dataset_split,revision=revision)
    return data,dict(kind='huggingface',id=config.dataset,revision=revision,
                     config=config.dataset_config,split=config.dataset_split)


def prepare_questions(rows, config, backend):
    eligible=[];seen=set();excluded={}
    def reject(reason): excluded[reason]=excluded.get(reason,0)+1
    for index,row in enumerate(rows):
        for column in [config.question_column,config.reference_column]+([config.solution_column] if config.solution_column else []):
            if column not in row: raise ValueError(f'Missing configured column: {column}')
        question=row[config.question_column];answer=row[config.reference_column]
        if not isinstance(question,str) or not question.strip(): reject('empty_question');continue
        number=parse_number(answer)
        if number is None: reject('unsupported_reference');continue
        if config.solution_column and not reference_decision(question,str(row[config.solution_column]),str(answer))['eligible']:
            reject('reference_audit');continue
        if re.search(r'!\[|https?://|\[asy\]|\\includegraphics|<img|\bdiagram\b|\bfigure\b',question,re.I):
            reject('diagram');continue
        if re.search(r'(?:find|enter|give|list|determine)\s+all\s+(?:the\s+)?(?:possible\s+)?(?:values|roots|solutions)|separated\s+by\s+commas',question,re.I):
            reject('multiple_answers');continue
        family=family_key(question)
        if family in seen: reject('duplicate_family');continue
        tokens=backend.prompt_ids(question)
        if not tokens or len(tokens)>config.max_prompt_tokens: reject('prompt_length');continue
        if len(tokens)+config.max_new_tokens>backend.max_context: reject('native_context');continue
        seen.add(family)
        key=hashlib.sha256(family.encode()).hexdigest()
        eligible.append(dict(id=key,source_index=index,family=family,question=question,
                             reference=str(number),prompt_token_ids=tokens))
    ordered=sorted(eligible,key=lambda q:hashlib.sha256(f'{config.seed}:{q["id"]}'.encode()).digest())
    if len(ordered)<config.max_questions:
        raise ValueError(f'Only {len(ordered)} eligible families; requested {config.max_questions}. Exclusions: {excluded}')
    chosen=ordered[:config.max_questions];n=len(chosen);train=int(.6*n);val=int(.2*n)
    for i,q in enumerate(chosen): q['split']='train' if i<train else 'val' if i<train+val else 'test'
    return chosen,dict(eligible=len(eligible),selected=n,exclusions=excluded,
                      counts={split:sum(q['split']==split for q in chosen) for split in ('train','val','test')},
                      limitation='Number-template grouping is not semantic or pretraining decontamination')
