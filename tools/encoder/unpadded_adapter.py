"""Execution adapter: singleton official encodings, one joint contrastive loss.

No model/attention/pooling/loss changes and no installed/upstream file edits.
This explicitly changes wrapper execution, rather than claiming stock batching.
"""
import torch


def require_singleton(features):
    ids=features['input_ids'];mask=features['attention_mask']
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]==0:
        raise ValueError('Unpadded execution requires one nonempty sequence per forward')
    if mask.shape!=ids.shape or not bool((mask==1).all()):
        raise ValueError('Padding is forbidden in singleton encoding')


def make_unpadded(base):
    class UnpaddedLLM2Vec(base):
        def tokenize(self,texts):
            if not texts:raise ValueError('Empty contrastive batch')
            sequences=[]
            for text in texts:
                if not isinstance(text,str) or not text.strip():raise ValueError('Empty prefix text')
                full=self.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
                if len(full)>self.max_length:raise ValueError('Prefix exceeds context; truncation forbidden')
                encoded=super().tokenize([text])
                require_singleton(encoded)
                sequences.append(encoded)
            return {'sequences':sequences}

        def forward(self,features):
            if self.skip_instruction:raise ValueError('All prefix tokens must be pooled; skip_instruction must be False')
            sequences=features.get('sequences',[features])
            if not sequences:raise ValueError('Empty contrastive batch')
            representations=[]
            for sequence in sequences:
                require_singleton(sequence)
                # Calls the official forward with exactly the same tensor shape
                # and position numbering as single-prefix feature extraction.
                representations.append(super().forward(sequence))
            # Keep every graph alive. The official trainer computes ONE loss on
            # these eight embeddings, then gathers 32 across four ranks.
            return torch.cat(representations,dim=0)
    return UnpaddedLLM2Vec
