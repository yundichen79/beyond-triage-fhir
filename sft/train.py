"""LoRA fine-tuning of Qwen3-8B on the per-turn SFT examples.

Each example is a prompt (the exact text the student saw) and a completion (the teacher's
turn). Loss is computed on completion tokens only. Predictions are computed only for
completion positions: prompts reach about 14K tokens and the vocabulary has about 152K
entries, so full logits would waste GPU memory. Training uses bf16 weights, LoRA on every
attention and MLP projection, gradient checkpointing, and one example per step with
gradient accumulation.

Outputs, in a new folder:
- train-config.json: arguments, data fingerprints and library versions
- train-log.jsonl: training and dev losses
- epoch-<n>/: one LoRA adapter per epoch

Pick the epoch and settings by running the student on the validation split
(`student.run --split validation --adapter <folder>/epoch-<n>`), not by dev loss alone.
`--dry-run` checks the data and tokenization without loading the model. `--smoke` trains
2 steps on the 16 longest training examples (peak memory and adapter saving) and writes
to the output folder like a real run.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

from dataset.common import file_digest
from student.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION

LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


@dataclass
class Config:
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_tokens: int = 16384
    eval_every: int = 50
    seed: int = 4880


def read_examples(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tokenize(examples, tokenizer, max_tokens):
    """Token ids plus the completion start. Prompt and completion must tokenize the same
    apart as joined, so the model trains on exactly the tokens it will generate."""
    encoded = []
    for example in examples:
        prompt = tokenizer.encode(example['prompt'], add_special_tokens=False)
        completion = tokenizer.encode(example['completion'], add_special_tokens=False)
        joined = tokenizer.encode(example['prompt'] + example['completion'], add_special_tokens=False)
        if joined != prompt + completion:
            raise ValueError(f"{example['id']}: prompt/completion boundary changes tokenization")
        if not completion:
            raise ValueError(f"{example['id']}: empty completion")
        if len(joined) > max_tokens:
            raise ValueError(f"{example['id']}: {len(joined)} tokens exceed --max-tokens {max_tokens}")
        encoded.append({'id': example['id'], 'input_ids': joined, 'start': len(prompt)})
    return encoded


def schedule(examples, config):
    steps_per_epoch = math.ceil(examples / config.effective_batch)
    total = steps_per_epoch * config.epochs
    return steps_per_epoch, total, max(1, round(total * config.warmup_ratio))


def completion_loss(model, example, torch):
    """Mean cross-entropy over completion tokens, with logits only for those positions."""
    causal = model.get_base_model()
    ids = torch.tensor([example['input_ids']], device='cuda')
    hidden = causal.model(input_ids=ids).last_hidden_state
    start = example['start']
    logits = causal.lm_head(hidden[:, start - 1:-1]).float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[0, start:])


def evaluate(model, examples, torch):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for example in examples:
            count = len(example['input_ids']) - example['start']
            total += completion_loss(model, example, torch).item() * count
            tokens += count
    model.train()
    return total / tokens if tokens else None


def train(args, config):
    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise ValueError('Training requires a GPU allocation; use scripts/train_sft.sbatch')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    kwargs = {'revision': MODEL_REVISION, 'cache_dir': str(MODEL_CACHE), 'local_files_only': True,
              'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    train_set = tokenize(read_examples(args.data / 'train.jsonl'), tokenizer, config.max_tokens)
    dev_set = tokenize(read_examples(args.data / 'dev.jsonl'), tokenizer, config.max_tokens)
    if args.smoke:
        # The longest examples set peak memory; two optimizer steps exercise the full loop.
        train_set = sorted(train_set, key=lambda e: -len(e['input_ids']))[:16]
        dev_set = dev_set[:4]
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map={'': 'cuda:0'}, **kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                                             lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS),
                                             task_type='CAUSAL_LM'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch, total_steps, warmup = schedule(len(train_set), config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    args.output.mkdir(parents=True)
    (args.output / 'train-config.json').write_text(json.dumps({
        'config': asdict(config), 'data': str(args.data),
        'data_sha256': {name: file_digest(args.data / name) for name in ('train.jsonl', 'dev.jsonl')},
        'base_model': {'model': MODEL_ID, 'revision': MODEL_REVISION}, 'lora_targets': LORA_TARGETS,
        'examples': {'train': len(train_set), 'dev': len(dev_set)}, 'smoke': args.smoke,
        'steps': {'per_epoch': steps_per_epoch, 'total': total_steps, 'warmup': warmup},
        'trainable_parameters': sum(p.numel() for p in trainable),
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                     'peft': peft.__version__}, 'gpu': torch.cuda.get_device_name(0)}, indent=2) + '\n')
    log = (args.output / 'train-log.jsonl').open('x')

    def record(**entry):
        entry['time'] = round(time.monotonic() - started, 1)
        log.write(json.dumps(entry) + '\n')
        log.flush()
        print(json.dumps(entry), flush=True)

    started, step = time.monotonic(), 0
    record(event='start', dev_loss=evaluate(model, dev_set, torch))
    model.train()
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train_set)))
        random.Random(config.seed + epoch).shuffle(order)
        for begin in range(0, len(order), config.effective_batch):
            batch = [train_set[i] for i in order[begin:begin + config.effective_batch]]
            tokens = sum(len(e['input_ids']) - e['start'] for e in batch)
            batch_loss = 0.0
            for example in batch:
                # Weight each example by its completion tokens: a token-level mean per step.
                count = len(example['input_ids']) - example['start']
                loss = completion_loss(model, example, torch) * (count / tokens)
                loss.backward()
                batch_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            entry = {'event': 'step', 'epoch': epoch, 'step': step, 'loss': round(batch_loss, 5),
                     'grad_norm': round(grad_norm, 4), 'lr': scheduler.get_last_lr()[0]}
            if step % config.eval_every == 0:
                entry['dev_loss'] = evaluate(model, dev_set, torch)
            record(**entry)
        model.save_pretrained(args.output / f'epoch-{epoch}')
        record(event='epoch', epoch=epoch, step=step, dev_loss=evaluate(model, dev_set, torch),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    log.close()


def dry_run(args, config):
    """Check data and tokenization without the model; print what training would do."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE),
                                              local_files_only=True, trust_remote_code=False)
    summary = {}
    for split in ('train', 'dev'):
        encoded = tokenize(read_examples(args.data / f'{split}.jsonl'), tokenizer, config.max_tokens)
        lengths = [len(e['input_ids']) for e in encoded]
        summary[split] = {'examples': len(encoded), 'tokens': sum(lengths), 'max_tokens': max(lengths),
                          'completion_tokens': sum(len(e['input_ids']) - e['start'] for e in encoded)}
    steps_per_epoch, total, warmup = schedule(summary['train']['examples'], config)
    summary['steps'] = {'per_epoch': steps_per_epoch, 'total': total, 'warmup': warmup}
    summary['config'] = asdict(config)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Folder written by sft.data')
    parser.add_argument('--output', type=Path, help='New folder for adapters and logs')
    parser.add_argument('--dry-run', action='store_true', help='Check data and tokenization only')
    parser.add_argument('--smoke', action='store_true', help='2 steps on the 16 longest examples (GPU check)')
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in asdict(defaults)})
    if args.smoke:
        config.epochs, config.effective_batch, config.eval_every = 1, 8, 1
    if config.epochs < 1 or config.effective_batch < 1 or config.learning_rate <= 0:
        parser.error('epochs, effective batch and learning rate must be positive')
    if args.dry_run:
        dry_run(args, config)
        return
    if not args.output:
        parser.error('--output is required for training')
    if args.output.exists():
        parser.error(f'Refusing to replace an existing training folder: {args.output}')
    train(args, config)


if __name__ == '__main__':
    main()
