"""Resumable supervised training; no actor updates, VERL, or scheduler dependency."""
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .config import TrainingConfig, select_layers, INSTRUCTION
from .data import load_rows, prepare_questions
from .storage import implementation_identity, read_json, write_json, save_tensor, load_tensor, sha256, publish, read_artifact
from .bottleneck import Bottleneck, loss, encode
from .features import isotropic_normalization
from .cumulative import trajectory_statistics, raw_design
from .linear import fit_system
from .evaluation import calibrate, metrics


class VerificationClient:
    def __init__(self, python=None):
        # Virtualenv executables are often symlinks to system Python. Realpath /
        # Path.resolve() discards the virtualenv identity and its installed packages.
        self.python=os.path.abspath(os.path.expanduser(os.fspath(python or sys.executable)))

    def check(self):
        """Validate interpreter, package imports and pinned versions before GPU work."""
        self([], '0')

    def __call__(self, responses, reference):
        from .grading import PINS
        rows=[dict(response=r,reference=reference) for r in responses]
        try:
            result=subprocess.run([self.python,'-m','slice_critic.verifier'],input=json.dumps(rows),
                text=True,capture_output=True,check=True,timeout=15+8*len(rows))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Verifier failed using {self.python} (exit {exc.returncode}):\n'
                               f'{(exc.stderr or "No stderr returned")[-6000:]}') from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'Verifier timed out after {exc.timeout}s using {self.python}') from exc
        record=json.loads(result.stdout)
        if record['provenance']['packages']!=PINS or len(record['results'])!=len(rows):
            raise ValueError('Verifier identity or batch count mismatch')
        self.provenance=record['provenance']
        return record['results']


def _save_record(path, payload):
    canonical=json.dumps(payload,sort_keys=True,allow_nan=False).encode()
    write_json(path,dict(payload=payload,sha256=hashlib.sha256(canonical).hexdigest()))


def _read_record(path):
    record=read_json(path)
    if hashlib.sha256(json.dumps(record['payload'],sort_keys=True,allow_nan=False).encode()).hexdigest()!=record['sha256']:
        raise ValueError(f'Changed stage record: {path}')
    return record['payload']


def _generate(config, out, questions, backend, grader):
    for q in questions:
        path=out/'trajectories'/f'{q["id"]}.json'
        if path.exists():
            record=_read_record(path)
            if record['question_id']!=q['id'] or len(record['responses'])!=config.responses_per_question:
                raise ValueError('Incompatible saved generation')
            continue
        seed=int(hashlib.sha256(f'{config.seed}:{q["id"]}'.encode()).hexdigest()[:8],16)
        print(f'Generating {q["split"]} question {q["id"][:12]}: '
              f'{config.responses_per_question} answers, up to {config.max_new_tokens} tokens each', flush=True)
        generated=backend.generate(q['prompt_token_ids'],config.responses_per_question,config.max_new_tokens,seed)
        if len(generated)!=config.responses_per_question: raise ValueError('Incomplete generated batch')
        grades=grader([r['response'] for r in generated],q['reference'])
        if len(grades)!=len(generated): raise ValueError('Incomplete verifier batch')
        for row,grade in zip(generated,grades):
            if grade['status']=='correct' and grade['diagnostic_reward']!=1: raise ValueError('Invalid correct label')
            if grade['status']=='incorrect' and grade['diagnostic_reward']!=0: raise ValueError('Invalid incorrect label')
            row.update(score=grade['diagnostic_reward'],verifier_status=grade['status'],extracted=grade.get('extracted'))
        _save_record(path,dict(question_id=q['id'],responses=generated,
                               verifier=getattr(grader,'provenance',{'kind':'custom supplied grader'})))
        print(f'Generated {q["split"]} question {q["id"][:12]}',flush=True)


def _answers(out,q):
    record=_read_record(out/'trajectories'/f'{q["id"]}.json')
    for i,row in enumerate(record['responses']):
        if row['verifier_status'] in ('correct','incorrect') and row['token_ids']:
            yield i,row


def _sampled(out, questions, config, backend, layers, split):
    xs=[];ys=[];groups=[];positions=[]
    for q in questions:
        if q['split']!=split: continue
        for i,row in _answers(out,q):
            cache=out/'prefix-features'/f'{q["id"]}-{i}.pt'
            source_hash=sha256(out/'trajectories'/f'{q["id"]}.json')
            if cache.exists():
                record=load_tensor(cache)
                if record['source_sha256']!=source_hash: raise ValueError('Trajectory changed after feature extraction')
            else:
                seed=int(hashlib.sha256(f'{config.seed}:prefix:{q["id"]}:{i}'.encode()).hexdigest()[:8],16)
                generator=torch.Generator().manual_seed(seed)
                # Uniform random pre-action lengths L in {0,...,T-1}, without replacement.
                sampled=torch.randperm(len(row['token_ids']),generator=generator)[:config.prefixes_per_answer].sort().values.tolist()
                x=backend.trajectory_features(q['prompt_token_ids'],row['token_ids'],sampled,layers).float().cpu()
                if x.shape!=(len(sampled),len(layers)*backend.hidden_size) or not torch.isfinite(x).all():
                    raise ValueError('Invalid sampled frozen features')
                record=dict(x=x,positions=sampled,source_sha256=source_hash)
                save_tensor(cache,record)
            xs.append(record['x']);ys.extend([float(row['score'])]*len(record['x']))
            groups.extend([q['id']]*len(record['x']));positions.extend(record['positions'])
    if not xs: raise ValueError(f'No verifiable nonempty answers for {split}')
    return dict(x=torch.cat(xs),y=torch.tensor(ys),question=groups,position=positions)


def _fit_compressor(out, train, val, config):
    checkpoint=out/'compressor.pt'
    if checkpoint.exists(): return load_tensor(checkpoint)
    if set(train['y'].tolist())!={0.,1.} or set(val['y'].tolist())!={0.,1.}:
        raise ValueError('Train and validation retained prefixes must each contain correct and incorrect outcomes')
    torch.manual_seed(config.seed)
    mean,scale=isotropic_normalization(train['x'])
    model=Bottleneck(train['x'].shape[1],config.compression_width)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.001)
    generator=torch.Generator().manual_seed(config.seed)
    best=float('inf');history=[];selected=None
    penalty=.01 if config.compression=='sparse' else 0.
    for epoch in range(config.epochs):
        model.train()
        for ids in torch.randperm(len(train['y']),generator=generator).split(config.batch_size):
            objective=loss(model,(train['x'][ids]-mean)/scale,train['y'][ids],penalty)
            if not torch.isfinite(objective): raise ValueError('Nonfinite compressor loss')
            optimizer.zero_grad();objective.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            probability=torch.cat([model((x-mean)/scale)[0].sigmoid() for x in val['x'].split(config.batch_size)])
            brier=float((probability-val['y']).square().mean())
        history.append(dict(epoch=epoch+1,validation_brier=brier))
        if brier<best:
            best=brier;selected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print('Compressor epoch',epoch+1,'validation Brier',brier,flush=True)
    record=dict(state=selected,mean=mean,scale=scale,width=config.compression_width,history=history,
                selection='Minimum validation sigmoid Brier at a fixed learning rate; no test access')
    save_tensor(checkpoint,record)
    return record


def _model(record):
    model=Bottleneck(len(record['mean']),record['width'])
    model.load_state_dict(record['state']);return model.eval().requires_grad_(False)


def _fit_head(out, questions, config, backend, layers, fitted):
    path=out/'head-statistics.pt';model=_model(fitted);k=config.compression_width+1
    if path.exists(): state=load_tensor(path)
    else: state=dict(a=torch.zeros(k,k,dtype=torch.float64),b=torch.zeros(k,dtype=torch.float64),
                     processed={},transitions=0,answers=0,compressor_sha256=sha256(out/'compressor.pt'))
    if state['compressor_sha256']!=sha256(out/'compressor.pt'): raise ValueError('Compressor changed after head accumulation')
    # Whole-trajectory checkpoints prevent double counting on retries.
    for q in questions:
        if q['split']!='train': continue
        source_hash=sha256(out/'trajectories'/f'{q["id"]}.json')
        for i,row in _answers(out,q):
            key=f'{q["id"]}:{i}'
            if key in state['processed']:
                if state['processed'][key]!=source_hash: raise ValueError('Changed accumulated trajectory')
                continue
            x=backend.trajectory_features(q['prompt_token_ids'],row['token_ids'],list(range(len(row['token_ids']))),layers)
            z=encode(model,x,fitted['mean'],fitted['scale'],'cpu')
            if config.method=='ridge':
                phi=raw_design(z);a=phi.T@phi;b=phi.sum(0)*row['score']
            else:
                a,b,_=trajectory_statistics(z,float(row['score']),config.critic_lambda)
            state['a']+=a;state['b']+=b;state['processed'][key]=source_hash
            state['transitions']+=len(z);state['answers']+=1
            if state['answers']%32==0: save_tensor(path,state)
    save_tensor(path,state)
    weights,residual=fit_system(state['a'],state['b'],config.epsilon)
    return weights,dict(answers=state['answers'],transitions=state['transitions'],relative_residual=residual,
                        method=config.method,critic_lambda=config.critic_lambda,gamma=1,epsilon=config.epsilon,
                        scope='All retained complete training trajectories; raw sums, penalized intercept')


def _run(config, out, device, backend, grader):
    from .backend import TransformersBackend, resolve_model
    identity_path=out/'run.json'
    previous=read_json(identity_path) if identity_path.exists() else None
    runtime = dict(python=sys.version)
    for name in ('torch','numpy','transformers','datasets','huggingface-hub'):
        try: runtime[name] = version(name)
        except PackageNotFoundError: runtime[name] = None
    if previous and previous.get('runtime') != runtime:
        raise ValueError('Training runtime changed; use a new output directory')
    if previous and previous['config']!=asdict(config): raise ValueError('Configuration changed; use a new output directory')
    if previous and previous.get('implementation')!=implementation_identity():
        raise ValueError('Package code changed; use a new training output directory')
    if grader is None:
        grader=VerificationClient(config.verifier_python)
        grader.check()
    if backend is None:
        model_config=replace(config,model_revision=previous['model']['revision']) if previous else config
        if previous and Path(previous['model']['local_path']).is_dir():
            path=Path(previous['model']['local_path']);revision=previous['model']['revision']
        else: path,revision=resolve_model(model_config)
        backend=TransformersBackend(path,device=device)
    else: revision=config.model_revision
    layers=select_layers(backend.num_layers,config.layer_base,config.num_slices)
    model=dict(id=config.model_id,revision=revision,local_path=str(backend.path),files=backend.model_files)
    if previous:
        if model['files']!=previous['model']['files']: raise ValueError('Model/tokenizer changed during resume')
        if sha256(out/'questions.json')!=previous['questions_sha256']: raise ValueError('Saved questions changed')
        if previous['dataset']['kind']=='jsonl' and sha256(previous['dataset']['path'])!=previous['dataset']['sha256']:
            raise ValueError('Local dataset changed')
        questions=read_json(out/'questions.json');identity=previous
    else:
        rows,dataset=load_rows(config)
        questions,selection=prepare_questions(rows,config,backend)
        write_json(out/'questions.json',questions)
        identity=dict(config=asdict(config),dataset=dataset,model=model,selection=selection,
                      layers_zero_based=layers,questions_sha256=sha256(out/'questions.json'),
                      implementation=implementation_identity(),runtime=runtime)
        write_json(identity_path,identity)
    if (out/'artifact').exists():
        read_artifact(out/'artifact');return out/'artifact'
    _generate(config,out,questions,backend,grader)
    train=_sampled(out,questions,config,backend,layers,'train')
    val=_sampled(out,questions,config,backend,layers,'val')
    fitted=_fit_compressor(out,train,val,config)
    weights,head=_fit_head(out,questions,config,backend,layers,fitted)
    compressor=_model(fitted)
    def raw(data):
        return raw_design(encode(compressor,data['x'],fitted['mean'],fitted['scale'],'cpu'))@weights
    calibration=calibrate(raw(val),val['y'])
    # Test prefixes are loaded only after compressor/head/validation decisions are fixed.
    test=_sampled(out,questions,config,backend,layers,'test')
    statuses=Counter();capped=0;answers=0;verifiers=[]
    for q in questions:
        generated=_read_record(out/'trajectories'/f'{q["id"]}.json')
        if generated['verifier'] not in verifiers: verifiers.append(generated['verifier'])
        for row in generated['responses']:
            statuses[row['verifier_status']]+=1;capped+=row['capped'];answers+=1
    predictions=raw(test)
    evaluation=dict(test=metrics(test['y'],predictions,calibration,float(train['y'].mean())),
                    validation=metrics(val['y'],raw(val),calibration,float(train['y'].mean())),
                    test_questions=len(set(test['question'])),status_counts=dict(statuses),
                    generated_answers=answers,truncation_rate=capped/answers,head_fit=head,
                    limitation='Metrics condition on correct/incorrect retained answers; sampled prefixes are correlated within questions')
    manifest=dict(version=out.name,package_version='0.1.0',model=model,prompt_instruction=INSTRUCTION,
                  generation=dict(engine='transformers',temperature=1.,top_p=1.,top_k=0,do_sample=True,
                                  terminal='EOS or configured response cap'),
                  prefix_sampling='Uniform without replacement from 0,...,T-1; at most configured prefixes_per_answer',
                  features=dict(layers_zero_based=layers,blocks_one_based=[i+1 for i in layers],
                                rms_eps=backend.rms_eps,width=config.compression_width,
                                input_dimension=len(layers)*backend.hidden_size,compression=config.compression,
                                source='Block outputs before learned final norm; per-block RMS then training-only isotropic normalization',
                                projection_row_normalized=True,activation='relu',
                                sparsity_penalty=.01 if config.compression=='sparse' else 0.),
                  training=identity,verifier=verifiers,
                  value_semantics='Expected terminal correctness under the frozen generation policy; not a proof certificate',
                  split_question_ids={split:[q['id'] for q in questions if q['split']==split] for split in ('train','val','test')},
                  representation_training_families=[q['family'] for q in questions if q['split']=='train'],
                  selection_families=[q['family'] for q in questions if q['split']=='val'])
    checkpoint=dict(compressor=fitted['state'],mean=fitted['mean'],scale=fitted['scale'],weights=weights)
    # Commit the audit before publishing, so a completed artifact never skips it on recovery.

    write_json(out/'test-predictions.json',dict(question=test['question'],position=test['position'],
                                              label=test['y'].tolist(),value=predictions.tolist()))
    return publish(out/'artifact',manifest,checkpoint,calibration,evaluation)


def train_critic(config=None, output_dir=None, *, device='cuda', backend=None, grader=None, **kwargs):
    """Train from explicit columns. Resume complete stage records with identical config.

    backend/grader injection supports testing or specialist integrations. Standard
    use loads the pinned Transformers model and isolated Math-Verify interpreter.
    """
    if config is not None and kwargs: raise ValueError('Supply a TrainingConfig OR keyword configuration')
    config=(config or TrainingConfig(**kwargs)).validate()
    if output_dir is None: raise ValueError('An explicit output_dir is required')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'.training.lock'
    # Advisory lock is released even if the process dies; only one local writer.
    import fcntl
    with lock.open('a+') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another trainer owns this output directory') from None
        started=time.perf_counter()
        write_json(out/'status.json',dict(state='running'))
        try:
            artifact=_run(config,out,device,backend,grader)
            write_json(out/'status.json',dict(state='complete',artifact=str(artifact),seconds=time.perf_counter()-started))
            return artifact
        except BaseException as exc:
            write_json(out/'status.json',dict(state='failed',error=str(exc),seconds=time.perf_counter()-started));raise
