import torch
import tqdm.std as tqdm_std

import datasets
from chrisbase.io import *
from chrisbase.time import *
from chrisbase.util import *


def copy_ipynb_for_run(infile, run_opts):
    infile = Path(infile)
    outdir = infile.with_name(f"{infile.stem}-{now('%m.%d')}")
    run_command("rm", "-rf", outdir, bare=True)
    for dst in sorted([make_dir(outdir / f"{s}-{r}") for s, rs in run_opts.items() for r in rs]):
        run_command("cp", infile, dst, bare=True)
    out_hr(title=f" * Input/Output Files [{', '.join(run_opts)}]")
    out_table(files_info(infile, outdir / '*' / '*.ipynb'))
    out_hr()


def copy_ipynb_for_debug(infile, opts):
    infile = Path(infile)
    outfiles = [infile.with_name(f"{infile.stem}={opt}.py") for opt in opts]
    for outfile in outfiles:
        with outfile.open('w') as out:
            for source in [x.source for x in load_attrs(infile).cells if x.cell_type == 'code']:
                out.writelines(source)
                out.writelines([hr(c='#', t=2, b=2)])
    out_hr(title=f" * Input/Output Files [{', '.join(opts)}]")
    out_table(files_info(infile, *outfiles))
    out_hr()


def get_options_from_path(strategy_d, precision_d, run_d, quick_d=False, valid_strategies=('dp', 'ddp', 'ddp_sharded', 'fsdp', 'deepspeed')):
    current_path = get_current_path()
    _for_demo = 'demo' in current_path.name or 'note' in current_path.parent.name
    _quick_demo = quick_d
    _strategy, _precision, _run = strategy_d, precision_d, run_d
    if _for_demo:
        _opt = current_path.stem
        if len(_opt.rsplit('=', maxsplit=1)) > 1:
            _opt = _opt.split('=', maxsplit=1)[-1]
            if len(_opt.rsplit('-')) >= 3:
                _strategy, _precision, _run = _opt.rsplit('-')
            elif len(_opt.rsplit('-')) >= 2:
                _strategy, _run = _opt.rsplit('-')
            else:
                _run = _opt
    else:
        _opt = current_path.parent.name
        if len(_opt.rsplit('=', maxsplit=1)) > 1:
            _opt = _opt.split('=', maxsplit=1)[-1]
        if len(_opt.rsplit('-')) >= 3:
            _strategy, _precision, _run = _opt.rsplit('-', maxsplit=2)
        elif len(_opt.rsplit('-')) >= 2:
            _strategy, _run = _opt.rsplit('-')
        else:
            _run = _opt
    _strategy = _strategy if _strategy in valid_strategies else strategy_d
    _precision, _run = int(number_only(_precision)), int(number_only(_run))
    return _for_demo, _quick_demo, (_strategy, _precision, _run)


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

    def __exit__(self, type_, value, traceback):
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
