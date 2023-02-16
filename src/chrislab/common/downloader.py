import argparse

import datasets.utils.logging
from chrisbase.morp import *
from datasets import load_dataset, Dataset
from .util import *

time_tqdm = time_tqdm_cls(bar_size=40, desc_size=22)
mute_tqdm = mute_tqdm_cls()
sorted_splits = ('train', 'valid', 'validation', 'test')


def convert_json_lines(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)
    example_count = 0
    example_data = []
    with infile.open('r') as inp:
        for line in inp.readlines():
            example_data.append(json.loads(line))
            example_count += 1
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open('w') as out:
        json.dump({"version": f"datasets_1.0", "data": example_data}, out, ensure_ascii=False, indent=4)
    return example_count


def download_public_task_data(data_dir, data_name, sub_name, remove_temporary=True, mute_progress_bar=True):
    with MuteDatasetProgress(mute=mute_progress_bar):
        data_dir = Path(data_dir)
        outdir = data_dir / data_name / sub_name
        tmpdir = data_dir / data_name / (sub_name + "-temp")
        raw_datasets = datasets.load_dataset(data_name, sub_name)
        raw_datasets.save_to_disk(str(tmpdir))

        results = []
        for k, dataset in raw_datasets.items():
            k = k.replace('validation', 'valid')
            dataset.to_json(tmpdir / f"{k}.json", force_ascii=False)
            tmpfile, outfile = tmpdir / f'{k}.json', outdir / f'{k}.json'
            num_example = convert_json_lines(tmpfile, outfile)
            results.append({
                'data_name': data_name, 'sub_name': sub_name, 'split': outfile.stem,
                '#example': f'{num_example:,d}', 'size': f'{file_size(outfile):,d}', 'time': file_mtime(outfile)
            })
        info_file = tmpdir / "train" / "dataset_info.json"
        if info_file.exists() and info_file.is_file():
            shutil.copyfile(info_file, outdir / "info.json")
        if remove_temporary and tmpdir.exists() and tmpdir.is_dir():
            shutil.rmtree(tmpdir)
        return to_dataframe(results)


def download_public_dataset(data_dir, data_name, sub_names):
    return pd.concat([download_public_task_data(data_dir, data_name, sub_name)
                      for sub_name in sub_names]).reset_index(drop=True)


def reload_public_task_data(data_dir, data_name, sub_name):
    indir = data_dir / data_name / sub_name
    data_files = {x.stem: str(x) for x in files(indir / "*.json") if x.stem != "info"}
    with MyTimer(verbose=False):
        datasets.utils.logging.tqdm = mute_tqdm
        raw_datasets = load_dataset("json", data_files=data_files, field="data")
    results = []
    for split in sorted(raw_datasets, key=lambda x: sorted_splits.index(x) if x in sorted_splits else 999):
        dataset: Dataset = raw_datasets[split]
        results.append({
            'data_name': data_name, 'sub_name': sub_name, 'split': split,
            '#example': f'{dataset.num_rows:,d}', 'size': f'{file_size(data_files[split]):,d}', 'time': file_mtime(data_files[split])
        })
    return to_dataframe(results)


def reload_public_dataset(data_dir, data_name, sub_names):
    return pd.concat([reload_public_task_data(data_dir, data_name, sub_name)
                      for sub_name in sub_names]).reset_index(drop=True)


def add_column_with_token_tag(infile, outfile, suffix, targets, netloc="129.254.164.137:7100"):
    with MyTimer(verbose=True, flush_sec=0.3):
        tagger = MorpClient(netloc=netloc)
        infile = Path(infile)
        outfile = make_parent_dir(outfile)
        with infile.open() as inp:
            num_update = 0
            contents = json.load(inp)
            for example in time_tqdm(contents['data'], desc=f"({infile.stem}) analyzing text"):
                for target in targets:
                    if f'{target}_{suffix}' not in example:
                        example[f'{target}_{suffix}'] = tagger.token_tag(example[target])
                        num_update += 1
    with MyTimer(verbose=True, flush_sec=0.3):
        if num_update > 0:
            with outfile.open('w') as out:
                json.dump({"version": f"datasets_1.0", "data": contents['data']}, out, ensure_ascii=False, indent=4)


def add_column_with_token_only(infile, outfile, suffix, targets, netloc="129.254.164.137:7105"):
    with MyTimer(verbose=True, flush_sec=0.3):
        tagger = MorpClient(netloc=netloc)
        infile = Path(infile)
        outfile = make_parent_dir(outfile)
        with infile.open() as inp:
            num_update = 0
            contents = json.load(inp)
            for example in time_tqdm(contents['data'], desc=f"({infile.stem}) analyzing text"):
                for target in targets:
                    if f'{target}_{suffix}' not in example:
                        example[f'{target}_{suffix}'] = tagger.token_only(example[target])
                        num_update += 1
    with MyTimer(verbose=True, flush_sec=0.3):
        if num_update > 0:
            with outfile.open('w') as out:
                json.dump({"version": f"datasets_1.0", "data": contents['data']}, out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue", default="", type=str, required=False,
                        help=f"What tasks of GLUE to make: cola, sst2, mrpc, qqp, stsb, mnli, mnli_mismatched, mnli_matched, qnli, rte, wnli, ax")
    parser.add_argument("--klue", default="", type=str, required=False,
                        help=f"What tasks of KLUE to make: ynat, sts, nli, ner, re, dp, mrc, wos")
    args = parser.parse_args()

    if args.glue != "":
        download_public_dataset("data", "glue", [x.strip() for x in args.glue.split(',')])

    if args.klue != "":
        download_public_dataset("data", "klue", [x.strip() for x in args.klue.split(',')])
