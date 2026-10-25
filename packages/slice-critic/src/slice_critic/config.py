"""Explicit model, layer and resource limits; no GPU work during preview."""
from dataclasses import asdict, dataclass
import math

MODEL_ID = 'Qwen/Qwen2.5-Math-1.5B'
INSTRUCTION = r'Solve the math problem step by step. Put your final numeric answer inside a single \boxed{...}.'


def immutable_revision(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


def completion_prompt(question):
    return INSTRUCTION+'\n\nQuestion: '+question+'\n\nSolution:'


def select_layers(num_layers, base=2., count=5):
    """Zero-based blocks, final first. Floor powers; resolve ties consecutively."""
    if (type(num_layers) is not int or type(count) is not int or count<1
            or isinstance(base,bool) or not isinstance(base,(int,float)) or not math.isfinite(base) or base<1 or count>num_layers):
        raise ValueError('Invalid geometric layer selection')
    distances=[]
    for i in range(count):
        try: distance=max(int(math.floor(base**i)), distances[-1]+1 if distances else 1)
        except (OverflowError, ValueError): raise ValueError('Layer spread exceeds model depth') from None
        if distance>num_layers: raise ValueError('Layer spread exceeds model depth; reduce count or base')
        distances.append(distance)
    return [num_layers-d for d in distances]


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str
    question_column: str = 'problem'
    reference_column: str = 'answer'
    solution_column: str | None = None
    dataset_config: str | None = None
    dataset_split: str = 'train'
    dataset_revision: str | None = None
    model_path: str | None = None
    model_revision: str | None = None
    model_id: str = MODEL_ID
    max_questions: int = 1000
    responses_per_question: int = 4
    max_new_tokens: int = 1024
    max_prompt_tokens: int = 512
    layer_base: float = 2.
    num_slices: int = 5
    compression_width: int = 768
    compression: str = 'sparse'
    method: str = 'ridge'
    critic_lambda: float = .99
    epsilon: float = .01
    prefixes_per_answer: int = 8
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = .0003
    seed: int = 42
    verifier_python: str | None = None

    def validate(self):
        if not isinstance(self.dataset,str) or not self.dataset.strip(): raise ValueError('dataset is required')
        for field in ('question_column','reference_column','dataset_split'):
            if not isinstance(getattr(self,field),str) or not getattr(self,field).strip(): raise ValueError(f'{field} is required')
        for name in ('max_questions','responses_per_question','max_new_tokens','max_prompt_tokens',
                     'num_slices','compression_width','prefixes_per_answer','epochs','batch_size'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError(f'{name} must be a positive integer')
        if self.max_questions<20: raise ValueError('Use at least 20 question families for train/validation/test splits')
        if self.model_path and self.model_revision is not None and not immutable_revision(self.model_revision):
            raise ValueError('Local model_revision, when supplied, must be an immutable commit SHA')
        if self.model_id!=MODEL_ID: raise ValueError('Version 1 supports the fixed Qwen2.5-Math-1.5B encoder')
        if self.compression not in ('plain','sparse') or self.method not in ('ridge','lstd'):
            raise ValueError('Unsupported compression or head method')
        if not 0<=self.critic_lambda<=1 or not math.isfinite(self.epsilon) or self.epsilon<=0:
            raise ValueError('Invalid trace lambda or epsilon')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0: raise ValueError('Invalid learning rate')
        if type(self.seed) is not int or self.seed<0: raise ValueError('Seed must be a nonnegative integer')
        if self.max_prompt_tokens+self.max_new_tokens>4096:
            raise ValueError('Prompt and response budgets exceed the supported native 4096-token context')
        select_layers(28,self.layer_base,self.num_slices)
        return self

    def preview(self):
        self.validate()
        layers=select_layers(28,self.layer_base,self.num_slices)
        return dict(config=asdict(self),blocks_one_based=[i+1 for i in layers],
                    input_dimension=len(layers)*1536,critic_dimension=self.compression_width+1,
                    maximum_answers=self.max_questions*self.responses_per_question,
                    maximum_generated_tokens=self.max_questions*self.responses_per_question*self.max_new_tokens,
                    split_rule='60/20/20 question families, deterministic seeded ordering',
                    generation='frozen model, sampling temperature=1, top_p=1',
                    fitting='compressor on sampled pre-action prefixes; head on all retained training transitions')
