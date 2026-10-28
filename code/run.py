"""Run the portable static-three-model recipe on prepared PVDAQ arrays."""
import argparse
import json
import os
from pathlib import Path
import shutil

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, default=Path('runs'), help='Prepared arrays and run outputs; never the repository itself')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='Paper training used CUDA bfloat16; CPU is an explicitly different execution mode')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('materialize', 'features', 'tree', 'predict'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--site', choices=('9069', '9068'), required=True)
        if name == 'materialize':
            cmd.add_argument('--source-root', type=Path, required=True, help='Strict power CSV, frozen queue and own-site archived GFS receipts/raw subsets')
    for name in ('freeze', 'score'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--sites', nargs='+', choices=('9069', '9068'), default=['9069', '9068'])
    cmd = sub.add_parser('neural')
    cmd.add_argument('--site', choices=('9069', '9068'), required=True)
    cmd.add_argument('--model', choices=('nhitsx', 'timemixer'), required=True)
    cmd.add_argument('--seed', type=int, choices=(2021, 2022, 2023), required=True)
    sub.add_parser('smoke', help='Tiny synthetic CPU forward/backward, feature and convex-fit checks; never uses real datasets')
    return p

def main():
    args = parser().parse_args()
    if args.command == 'smoke':
        from smoke import run
        print(json.dumps(run(), indent=2))
        return
    run_root = args.work_dir.resolve()
    code_root = Path(__file__).resolve().parent
    if run_root == code_root or run_root in code_root.parents:
        raise ValueError('Choose a dedicated output directory, not the code directory or an ancestor')
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ['PV3_RUN_DIR'] = str(run_root)
    os.environ['PV3_DEVICE'] = args.device
    if hasattr(args, 'source_root'):
        os.environ['PV3_SOURCE_DIR'] = str(args.source_root.resolve())
    for name in ('architecture_configs.json', 'tree_parameters.json'):
        source = code_root / 'configs' / name
        dest = run_root / name
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError(f'Existing {name} differs from the released configuration')
        if not dest.exists():
            shutil.copy2(source, dest)
    if args.command == 'materialize':
        from pv3.materialize import run
        run(args.site)
    elif args.command == 'freeze':
        from pv3.common import freeze
        print(json.dumps(freeze(args.sites)['counts'], indent=2))
    elif args.command == 'score':
        from pv3.score import main as score
        score(args.sites)
    else:
        import torch
        if args.command in ('neural', 'predict') and args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu only for an explicitly different execution mode')
        from pv3 import engine
        if args.command == 'features': engine.prepare_tree(args.site)
        elif args.command == 'tree': engine.tree(args.site)
        elif args.command == 'neural': engine.neural(args.site, args.model, args.seed)
        elif args.command == 'predict': engine.predict(args.site)

if __name__ == '__main__':
    main()
