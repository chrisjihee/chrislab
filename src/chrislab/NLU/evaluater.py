from .cli import *

tqdm = time_tqdm_cls(bar_size=20, desc_size=40, aline='left')


def rerank_output(*predicted_files, id_col, plain_col, tagged_col, predict_col, out_suffix=".out", seed=0, multiplier=1_000_000,
                  group_name=lambda x: x.name.split('=')[0]):
    ran = random.Random(seed)
    all_predicted_files = list(chain(*[files(x) for x in predicted_files]))
    if not all_predicted_files:
        out_hr(title=f"No predicted_files: {', '.join(map(str, predicted_files))}")
        return
    group_files = {k: list(vs) for k, vs in grouped(all_predicted_files, key=group_name)}
    with MyTimer(f"Rerank({', '.join(map(str, predicted_files))})", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True):
        for g, (group, grouped_files) in enumerate(group_files.items()):
            for predicted_file in tqdm(grouped_files, desc=f"({g + 1:02d}/{len(group_files):02d}) {group:<50s}", file=stdout):
                all_samples = []
                with predicted_file.open() as inp:
                    keys = inp.readline().strip().split('\t')
                    for line in inp.readlines():
                        sample = dict(zip(keys, line.strip().split('\t')))
                        sample[id_col] = dict([s.strip().split('=', maxsplit=1) for s in sample[id_col].split(',')])
                        sample[id_col]['sid'] = int(sample[id_col]['sid'])
                        sample[predict_col] = float(sample[predict_col])
                        sample['ssid'] = f"{sample[id_col]['src']:*<15s}*{sample[id_col]['sid']:010d}"
                        all_samples.append(sample)
                ssid_samples = {k: list(g) for k, g in grouped(all_samples, itemgetter='ssid')}
                outfile = new_path(predicted_file).with_suffix(out_suffix)
                with outfile.open("w") as out:
                    for ssid, samples in ssid_samples.items():
                        reranked_samples = sorted(shuffled(samples, ran=ran), key=itemgetter(predict_col), reverse=True)
                        plain = samples[0][plain_col]
                        tagged = '\t'.join(map(str, chain.from_iterable((sample[tagged_col], int(sample[predict_col] * multiplier), r + 1)
                                                                        for r, sample in enumerate(reranked_samples))))
                        print('\t'.join([group, ssid, plain, tagged]), file=out)


def evaluate_reranked(*predicted_files, nbest=10, verbose=True, group_name=lambda x: x.name.split('=')[0],
                      evaluater={"project": "/dat/proj/WiseChris3", "program": "WiseChris3-2022.09.jar"}):
    all_predicted_files = list(chain(*[files(x) for x in predicted_files]))
    if not all_predicted_files:
        out_hr(title=f"No predicted_files: {', '.join(map(str, predicted_files))}")
        return
    group_files = {k: list(vs) for k, vs in grouped(all_predicted_files, key=group_name)}
    with MyTimer(f"Evaluate({', '.join(map(str, predicted_files))})", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True):
        for g, (group, grouped_files) in enumerate(group_files.items()):
            for predicted_file in tqdm(grouped_files, desc=f"({g + 1:02d}/{len(group_files):02d}) {group:<50s}", file=stdout):
                if verbose:
                    print()
                predictor_state_path = first_path_or(predicted_file.parent / f'{group}=*predictor_state*.json')
                if verbose:
                    print(f"- predicted_file       = {predicted_file}")
                    print(f"- predictor_state_path = {predictor_state_path}")
                if not predictor_state_path or not predictor_state_path.is_file():
                    if verbose:
                        print(f"[X] No predictor state file: {predicted_file.parent / f'{group}=*predictor_state*.json'}")
                    continue
                predictor_state = load_attrs(predictor_state_path)
                gold_files = []
                for (k, data_file) in predictor_state.data_files.items():
                    if data_file and Path(data_file).exists():
                        data_file = Path(data_file)
                        if k in predictor_state.dataloader_splits and predictor_state.dataloader_splits[k] and \
                                k in predictor_state.predicting_splits and predictor_state.predicting_splits[k]:
                            gold_file = data_file.with_suffix(".tsv")
                            if gold_file.parent.name.startswith(group) and gold_file.exists():
                                gold_files.append(gold_file)
                if not gold_files:
                    if verbose:
                        print(f"[X] No gold files for : {predicted_file}")
                    continue
                copy_files = [make_dir(predicted_file.parent / predicted_file.stem) / f"{gold_file.stem}.out" for gold_file in gold_files]
                with predicted_file.open() as inp:
                    predicted_lines = inp.readlines()
                for copy_file in copy_files:
                    with copy_file.open('w') as out:
                        out.writelines(predicted_lines)
                jar_file = Path(evaluater["project"]) / "build/libs" / evaluater["program"]
                dep_files = Path(evaluater["project"]) / "build/libs" / "dep/*"
                result_file = predicted_file.with_suffix(".xlsx")
                run_class = "mlt.EvalKt"
                task = "재순위화_예측결과_평가"
                arg1 = gold_files[0].parent / "*.tsv"
                arg2 = copy_files[0].parent / "*.out"
                arg3 = result_file
                arg4 = f"{predicted_file.parent}/"
                arg5 = nbest
                assert jar_file.exists() and jar_file.is_file(), f"No jar_file: {jar_file}"
                run_command("java", "-cp", f"{jar_file}:{dep_files}", run_class, task, arg1, arg2, arg3, arg4, arg5, 240, 180, 15, verbose=verbose, bare=True)
                assert result_file.exists() and result_file.is_file(), f"No result_file: {result_file}"


def summarize_evaluated(*predicted_files, use_cols, colnames, skiprows=None, skipfooter=0, nbest=1, only_nbest=False, summary_file=None, verbose=False, detail_scores=False,
                        sort_cols=('nbest', 'grand', 'parent', 'filename', 'epoch'),
                        out_cols=('nbest', 'grand', 'parent', 'filename', 'epoch'),
                        group_name=lambda x: x.name.split('=')[0]):
    all_predicted_files = list(chain(*[files(x) for x in predicted_files]))
    if not all_predicted_files:
        out_hr(title=f"No predicted_files: {', '.join(map(str, predicted_files))}")
        return
    group_files = {k: list(vs) for k, vs in grouped(all_predicted_files, key=group_name)}
    flatted_cols = list(chain([colnames[0]], *colnames[1:]))
    with MyTimer(f"Summarize({', '.join(map(str, predicted_files))})", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True):
        results = []
        for g, (group, grouped_files) in enumerate(group_files.items()):
            for predicted_file in tqdm(grouped_files, desc=f"({g + 1:02d}/{len(group_files):02d}) {group:<50s}", file=stdout):
                if verbose:
                    print()
                if verbose:
                    print(f"- predicted_file       = {predicted_file}")
                with MyTimer(verbose=verbose, mute_warning="openpyxl", rt=1, rb=1):
                    result = pd.read_excel(predicted_file, usecols=use_cols, names=flatted_cols, skiprows=skiprows, skipfooter=skipfooter)
                    print(result)
                result['grand'] = predicted_file.parent.parent.name
                result['parent'] = predicted_file.parent.name
                result['filename'] = predicted_file.name
                result['model'] = result[colnames[0]].apply(lambda x: to_prefix(x, idx=-1))
                result['epoch'] = result['model'].apply(lambda x: to_prefix(to_prefix(x, sep='predict-', idx=-1), sep='e,')).astype(int)
                result['nbest'] = result['model'].apply(lambda x: to_postfix(x, sep='nbest=')).astype(int)
                res_cols = list(out_cols)
                for score_cols in colnames[1:]:
                    score_name = score_cols[0].split("_")[0]
                    result[score_name] = result[score_cols].astype(float).mean(axis=1)
                    res_cols.append(score_name)
                if detail_scores:
                    res_cols += list(chain(*colnames[1:]))
                result = result[res_cols]
                results.append(result)
        if not results:
            out_hr(title=f"No result!")
            return
        total = pd.concat(results)
        if only_nbest:
            final = total[total['nbest'] == nbest].sort_values(by=list(sort_cols)).reset_index(drop=True)
        else:
            final = total[total['nbest'] <= nbest].sort_values(by=list(sort_cols)).reset_index(drop=True)
        if summary_file:
            final.to_excel(make_parent_dir(summary_file))
        return final
