from __future__ import annotations

import torch
import tqdm.std as tqdm_std
from pymongo import ASCENDING as ASC
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.typings import _DocumentType

import datasets
from chrisbase.io import *
from chrisbase.time import *
from chrisbase.util import *


def copy_ipynb_for_run(infile, run_opts=None):
    infile = Path(infile)
    outdir = infile.with_name(f"{infile.stem}-{now('%m.%d')}")
    run_command("rm", "-rf", outdir, bare=True)
    if run_opts:
        for dst in sorted([make_dir(outdir / f"{s}-{r}") for s, rs in run_opts.items() for r in rs]):
            run_command("cp", infile, dst, bare=True)
    else:
        for dst in sorted([make_dir(outdir)]):
            run_command("cp", infile, dst, bare=True)
    out_hr(title=f" * Input/Output Files")
    out_table(files_info(infile, outdir / '*.ipynb', outdir / '*' / '*.ipynb'))
    out_hr()


def copy_ipynb_for_debug(infile, opts):
    infile = Path(infile)
    outfiles = [infile.with_name(f"{infile.stem}={opt}.py") for opt in opts]
    for outfile in outfiles:
        with outfile.open('w') as out:
            for source in [x.source for x in load_attrs(infile).cells if x.cell_type == 'code']:
                out.writelines(source)
                out.writelines([hr(c='#', t=2, b=2)])
    out_hr(title=f" * Input/Output Files")
    out_table(files_info(infile, *outfiles))
    out_hr()


def get_options_from_path(default, valid_strategies=('dp', 'ddp', 'deepspeed')):
    final = default
    this = get_current_path()
    _opt = this.parent.name if this.stem.startswith('note') else this.stem
    if len(_opt.rsplit('=', maxsplit=1)) > 1:
        _opt = _opt.split('=', maxsplit=1)[-1]
    if len(_opt.rsplit('-')) >= 5:
        splits = _opt.rsplit('-', maxsplit=4)
        final['devices'] = [int(number_only(x)) for x in splits[-5].split(',')]
        final['batch'] = int(number_only(splits[-4]))
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    if len(_opt.rsplit('-')) >= 4:
        splits = _opt.rsplit('-', maxsplit=3)
        final['batch'] = int(number_only(splits[-4]))
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    if len(_opt.rsplit('-')) >= 3:
        splits = _opt.rsplit('-', maxsplit=2)
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    elif len(_opt.rsplit('-')) >= 2:
        splits = _opt.rsplit('-', maxsplit=2)
        final['strategy'] = splits[-2] if splits[-2] in valid_strategies else default['strategy']
        final['run'] = int(number_only(splits[-1]))
    else:
        final['run'] = int(number_only(_opt))
    return final


def set_devices_to_runs(runs, use_gpu, have_gpu=torch.cuda.device_count()):
    gpus_for_use = {
        1: {
            0: [0],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [0] if have_gpu < 8 else [4],
            6: [1] if have_gpu < 8 else [5],
            7: [2] if have_gpu < 8 else [6],
            8: [3] if have_gpu < 8 else [7],
        },
        2: {
            0: [0, 1],
            1: [0, 1],
            2: [2, 3],
            3: [0, 1] if have_gpu < 8 else [4, 5],
            4: [2, 3] if have_gpu < 8 else [6, 7],
            5: [0, 1],
            6: [2, 3],
            7: [0, 1] if have_gpu < 8 else [4, 5],
            8: [2, 3] if have_gpu < 8 else [6, 7],
        },
        4: {
            0: [0, 1, 2, 3],
            1: [0, 1, 2, 3],
            2: [0, 1, 2, 3] if have_gpu < 8 else [4, 5, 6, 7],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 3] if have_gpu < 8 else [4, 5, 6, 7],
            5: [0, 1, 2, 3],
            6: [0, 1, 2, 3] if have_gpu < 8 else [4, 5, 6, 7],
            7: [0, 1, 2, 3],
            8: [0, 1, 2, 3] if have_gpu < 8 else [4, 5, 6, 7],
        },
    }
    assert use_gpu in gpus_for_use, f"Not defined {use_gpu} in gpus_for_use: defined for {list(gpus_for_use.keys())}"
    for r in runs:
        assert r in gpus_for_use[use_gpu], f"Not defined {r} in gpus_for_use[{use_gpu}]: defined for {list(gpus_for_use[use_gpu].keys())}"
        for x in tupled(runs[r]):
            x['devices'] = gpus_for_use[use_gpu][r] if use_gpu in gpus_for_use and r in gpus_for_use[use_gpu] else None
    return runs


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        def empty_fn(*args, **kwargs):
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback_):
        return


class mute_tqdm_cls:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None

    def get_lock(self):
        pass


class time_tqdm_cls:
    def to_desc(self, desc, pre=None):
        return f"- {now(prefix=self.prefix)}{f' {pre}' if pre else ''} {desc:{self.aline}{self.desc_size}s}"

    def __init__(self, bar_size, desc_size, prefix=None, file=stdout, aline='right'):
        self.desc_size = desc_size
        self.bar_size = bar_size
        self.prefix = prefix
        self.file = file
        self.aline = '<' if str(aline).strip().lower() == 'left' else '>'

    def __call__(self, *args, **kwargs):
        if 'desc' not in kwargs or not kwargs['desc'] or ('position' in kwargs and kwargs['position'] and kwargs['position'] > 0):
            return EmptyTqdm(*args, **kwargs)
        else:
            if kwargs['desc'].endswith(' #0'):
                kwargs['desc'] = kwargs['desc'][:-3]
            kwargs['desc'] = self.to_desc(desc=kwargs['desc'],
                                          pre=kwargs.pop('pre') if 'pre' in kwargs else None)
            kwargs.pop('file', None)
            kwargs.pop('bar_format', None)
            return tqdm_std.tqdm(*args, bar_format=f"{{l_bar}}{{bar:{self.bar_size}}}{{r_bar}}", file=self.file, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        return tqdm_std.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        return tqdm_std.tqdm.get_lock()


def limit_num_samples(num, num_max, num_min=1):
    if isinstance(num, int):
        return min(max(num_min, num), num_max)
    elif isinstance(num, float):
        return min(max(num_min, int(num * num_max)), num_max)
    else:
        raise ValueError(f"Given number should be int or float: given_num={num}")


def to_tensor_batch(batch, input_keys):
    for key in input_keys:
        if isinstance(batch[key], list) and isinstance(batch[key][0], torch.Tensor):
            batch[key] = torch.stack(batch[key], dim=1)
    return batch


class MuteDatasetProgress:
    def __init__(self, mute=True):
        self.mute = mute

    def __enter__(self):
        if self.mute:
            datasets.utils.logging.disable_progress_bar()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self.mute:
            datasets.utils.logging.enable_progress_bar()


class StageMarker:
    def __init__(self, node_idx, world_size, milestones, db_name, tab_name, host="localhost", port=27017,
                 debug=False, trace=False, log_file=sys_stdout):
        self.node_idx = node_idx
        self.world_size = world_size
        self.db_name = db_name
        self.tab_name = tab_name
        self.host = host
        self.port = port
        self.milestones = dict(enumerate(milestones))
        self.debug = debug
        self.trace = trace
        self.mongo: MongoClient[_DocumentType] | None = None
        self.table: Collection | None = None
        self.log_file = log_file

    def __enter__(self) -> "StageMarker":
        self.mongo = MongoClient(host=self.host, port=self.port)
        self.table = self.mongo[self.db_name][self.tab_name]
        return self

    def __exit__(self, type_, value, traceback_) -> None:
        self.mongo.close()
        return

    def _sub_functions(self):
        return [
            lambda: f'#{self.node_idx + 1:01d}',
            lambda: now('%Y/%m/%d %H:%M:%S'),
            lambda yes: f"\n{to_dataframe(self.table.find().sort([('stage', ASC), ('agent', ASC)]), index='_id')}" if yes else "",
            lambda what, at: self.table.count_documents({what: 1, 'stage': at}),
        ]

    def initialize(self, stage, sleep_sec=1.0) -> None:
        by, at, data = self._sub_functions()[:3]
        self.table.delete_many({'stage': stage, 'agent': by()})
        self.table.insert_one(merge_dicts(
            {'stage': stage, 'agent': by()},
            {what: 0 for what in self.milestones.values()},
            {'started': at(), 'updated': at()},
        ))
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))
        if self.debug:
            print(f"[{by()}][{at()}]: initialized{data(yes=self.trace)}", file=self.log_file)

    def mark_done(self, what, stage, sleep_sec=1.0, max_sleep_times=3600) -> None:
        by, at, data, done = self._sub_functions()
        self.table.update_many({'stage': stage, 'agent': by()}, {'$set': {what: 0, 'updated': at()}})
        self.table.update_one({'stage': stage, 'agent': by()}, {'$set': {what: 1, 'updated': at()}})
        for _ in range(max_sleep_times):
            if done(what, at=stage) >= self.world_size:
                break
            if self.debug:
                print(f"[{by()}][{at()}]: waiting.... (#done={done(what, at=stage)})", file=self.log_file)
            sleep(sleep_sec)
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))
        if self.debug:
            print(f"[{by()}][{at()}]: finished~~! (#done={done(what, at=stage)}){data(yes=self.trace)}", file=self.log_file)
