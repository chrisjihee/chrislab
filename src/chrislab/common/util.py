from __future__ import annotations

import os
from pathlib import Path
from sys import stderr, stdout
from time import sleep

import datasets
import torch
import tqdm.std as tqdm_std
from pymongo import ASCENDING as ASC
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.typings import _DocumentType
from tabulate import tabulate

from chrisbase.io import load_attrs, merge_dicts
from chrisbase.io import make_dir, files_info, dirs, exists_or, hr
from chrisbase.io import prepend_to_global_path
from chrisbase.io import running_file, run_command
from chrisbase.time import now
from chrisbase.util import number_only, NO, tupled, to_dataframe


def working_gpus(gpus=None):
    if gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def num_cuda_devices():
    from torch.cuda import _device_count_nvml
    nvml_count = _device_count_nvml()
    if nvml_count >= 0:
        return nvml_count
    else:
        return torch.cuda.device_count()


def include_cuda_bin_dir(candidate_dirs=None) -> Path:
    if candidate_dirs is None:
        candidate_dirs = sorted(dirs("/usr/local/cuda*"), reverse=True)
    cuda_dir = None
    for candidate_dir in candidate_dirs:
        if exists_or(candidate_dir) and exists_or(f"{candidate_dir}/bin"):
            cuda_dir = Path(candidate_dir)
            break
    if cuda_dir:
        prepend_to_global_path(f"{cuda_dir}/bin")
    return cuda_dir


def set_torch_ext_path(dev=1):
    torch_ext_dir = Path(f"cache/torch_extensions/dev={dev}")
    os.environ['TORCH_EXTENSIONS_DIR'] = f"{torch_ext_dir.absolute()}"


def set_tokenizers_parallelism(value=False):
    os.environ["TOKENIZERS_PARALLELISM"] = f"{value}".lower()


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
    return files_info(infile, outdir / '*.ipynb', outdir / '*' / '*.ipynb')


def copy_ipynb_for_debug(infile, opts):
    infile = Path(infile)
    outfiles = [infile.with_name(f"{infile.stem}={opt}.py") for opt in opts]
    for outfile in outfiles:
        with outfile.open('w') as out:
            for source in [x.source for x in load_attrs(infile).cells if x.cell_type == 'code']:
                out.writelines(source)
                out.writelines([hr(c='#', t=2, b=2)])
    return files_info(infile, *outfiles)


def get_options_from_path(default, valid_strategies=('dp', 'ddp', 'deepspeed')):
    final = default
    this = running_file()
    _opt = this.stem if this.parent.name.startswith('note') else this.parent.name
    if len(_opt.rsplit('=', maxsplit=1)) > 1:
        _opt = _opt.split('=', maxsplit=1)[-1]
    if len(_opt.rsplit('-')) >= 6:
        splits = _opt.rsplit('-', maxsplit=5)
        final['epoch'] = int(number_only(splits[-6]))
        final['devices'] = [int(number_only(x)) for x in splits[-5].split(',')]
        final['batch'] = int(number_only(splits[-4]))
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    elif len(_opt.rsplit('-')) >= 5:
        splits = _opt.rsplit('-', maxsplit=4)
        final['devices'] = [int(number_only(x)) for x in splits[-5].split(',')]
        final['batch'] = int(number_only(splits[-4]))
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    elif len(_opt.rsplit('-')) >= 4:
        splits = _opt.rsplit('-', maxsplit=3)
        final['batch'] = int(number_only(splits[-4]))
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    elif len(_opt.rsplit('-')) >= 3:
        splits = _opt.rsplit('-', maxsplit=2)
        final['strategy'] = splits[-3] if splits[-3] in valid_strategies else default['strategy']
        final['precision'] = int(number_only(splits[-2]))
        final['run'] = int(number_only(splits[-1]))
    elif len(_opt.rsplit('-')) >= 2:
        splits = _opt.rsplit('-', maxsplit=2)
        final['strategy'] = splits[-2] if splits[-2] in valid_strategies else default['strategy']
        final['run'] = int(number_only(splits[-1]))
    return final


def set_devices_to_runs(runs, use_gpu, have_gpu=num_cuda_devices()):
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
    time_fmt = '[%m.%d %H:%M:%S]'

    def __init__(self, node_idx, world_size, milestones, db_name, tab_name, host="localhost", port=27017,
                 debug=False, trace=False, log_file=stdout):
        self.node_idx = node_idx
        self.world_size = world_size
        self.db_name = db_name
        self.tab_name = tab_name
        self.host = host
        self.port = port
        self.milestones = tuple(milestones)
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

    @classmethod
    def _at_func(cls):
        return lambda: now(fmt=StageMarker.time_fmt)

    def _by_func(self):
        return lambda: f'#{self.node_idx + 1:01d}'

    def _state_func(self):
        return lambda: self.table.find().sort([('stage', ASC), ('agent', ASC)])

    def _state_str_func(self):
        return lambda yes: f"\n{to_dataframe(self._state_func()(), index='_id')}" if yes else ""

    def _done_func(self):
        return lambda what, stage: self.table.count_documents({what: what, 'stage': stage})

    def clear(self, sleep_sec=0.5):
        if self.node_idx == 0:
            self.table.delete_many({})
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))

    def initialize(self, stage, sleep_sec=0.5) -> None:
        by = self._by_func()
        at = self._at_func()
        state = self._state_str_func()
        self.table.delete_many({'stage': stage, 'agent': by()})
        self.table.insert_one(merge_dicts(
            {'started': at(), 'updated': at()},
            {'stage': stage, 'agent': by()},
            {what: NO for what in self.milestones},
        ))
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))
        if self.debug:
            print(f"[{by()}][{at()}]: initialized{state(yes=self.trace)}", file=self.log_file)

    def mark_done(self, what, stage, state_table_file=None, sleep_sec=0.5, max_sleep_times=3600) -> None:
        by = self._by_func()
        at = self._at_func()
        done = self._done_func()
        state = self._state_str_func()
        self.table.update_many({'stage': stage, 'agent': by()}, {'$set': {what: NO, 'updated': at()}})
        self.table.update_one({'stage': stage, 'agent': by()}, {'$set': {what: what, 'updated': at()}})
        for _ in range(max_sleep_times):
            if done(what, stage) >= self.world_size:
                break
            if self.debug:
                print(f"[{by()}][{at()}]: waiting.... (#done={done(what, stage)})", file=self.log_file)
            sleep(sleep_sec)
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))
        if self.debug:
            print(f"[{by()}][{at()}]: finished~~! (#done={done(what, stage)}){state(yes=self.trace)}", file=self.log_file)
        if state_table_file:
            self.print_state_table(stage=stage, mt=1 if what == self.milestones[0] else 0, file=state_table_file)

    def print_state_table(self, stage, mt=0, file=stderr, sleep_sec=0.5):
        by = self._by_func()
        if self.node_idx == 0:
            df = to_dataframe(self.table.find({'stage': stage, 'agent': by()}), exclude=('_id', 'agent', 'started'))
            print(('\n' * mt if mt > 0 else NO) + tabulate(df, showindex='never', tablefmt='plain'), end='\n', file=file)
        sleep(sleep_sec * (1 if self.node_idx == 0 else 2))
