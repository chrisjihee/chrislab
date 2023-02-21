from .cli import *


def convert_mlt_rerank(prefix, infile, outdir, max_n, unit, score,
                       split_rates={"train": 0.8, "valid": 0.1, "test": 0.1},
                       include_truths={"train": True, "valid": False, "test": False},
                       mini=-1, seed=0):
    tqdm = time_tqdm_cls(bar_size=20, desc_size=42, aline='left')
    with MyTimer(f"Make Dataset({outdir.stem})", prefix=prefix, mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True):
        assert files(infile), f"No input file: {infile}"
        assert sum(x for x in split_rates.values()) == 1.0, "Sum of split rates should be 1.0"
        infile = files(infile)[0]
        infile_size = num_lines(infile, mini)
        outdir = make_dir(outdir)
        nbest_ranges = {k: list(range(0 if v else 1, max_n + 1)) for k, v in include_truths.items()}
        with MyTimer(verbose=True, rb=1):
            print(f"- {'infile':30s} = {infile}")
            print(f"- {'outdir':30s} = {outdir}")
            print(f"- {'split_rates':30s} = {' | '.join(k + ES + percent(v, '3.0f').strip() for k, v in split_rates.items())}")
            print(f"- {'nbest_ranges':30s} = {' | '.join(k + ES + str(v) for k, v in nbest_ranges.items())}")
            print(f"- {'other_options':30s} = unit={unit} | score=(cols[{score[0]}]**{score[1]})*{score[2]} | seed={seed}")

        sent_ids = set()
        with MyTimer(verbose=True, rb=1):
            for cols in tqdm(tsv_lines(infile, mini), total=infile_size, desc=f"splitting {infile.name:<30s}", file=stdout):
                eid = cols[0]
                meta: dict = dict([x.strip().split("=") for x in eid.split(",")])
                if meta['unit'] == unit:
                    meta['N'] = int(meta['N'])
                    meta['sid'] = int(meta['sid'])
                    sent_id = f"{meta['src']}_{meta['sid']:08d}"
                    if meta['N'] in nbest_ranges['train'] or meta['N'] > 20000:
                        sent_ids.add(sent_id)
            sent_ids = shuffled(sent_ids, seed=seed)
            num_test = int(len(sent_ids) * split_rates['test'])
            num_valid = int(len(sent_ids) * split_rates['valid'])
            split_ids = {
                'test': sent_ids[: num_test],
                'valid': sent_ids[num_test: num_test + num_valid],
                'train': sent_ids[num_test + num_valid:],
            }
            sent_counts = {k: len(split_ids[k]) for k in split_ids.keys()}
            sent_counts_sum = sum(sent_counts.values())
            print(f"- {f'sent_counts':30s} = {sent_counts_sum:11,d} sentences")
            for k in ('train', 'valid', 'test'):
                print(f"- {f'sent_counts[{k}]':30s} = {sent_counts[k]:11,d} sentences ({percent(sent_counts[k] / sent_counts_sum)})")

        data_splits = {'train': [], 'valid': [], 'test': []}
        gold_splits = {'train': [], 'valid': [], 'test': []}
        sample_counts = {'train': 0, 'valid': 0, 'test': 0}
        with MyTimer(verbose=True, rb=1):
            for cols in tqdm(tsv_lines(infile, mini), total=infile_size, desc=f"converting {infile.name:<30s}", file=stdout):
                (eid, sent, morp) = cols[:3]
                label = (float(cols[score[0]]) ** score[1]) * score[2]
                meta: dict = dict([x.strip().split("=") for x in eid.split(",")])
                if meta['unit'] == unit:
                    meta['N'] = int(meta['N'])
                    meta['sid'] = int(meta['sid'])
                    sent_id = f"{meta['src']}_{meta['sid']:08d}"
                    split_id = 'test' if sent_id in split_ids['test'] else ('valid' if sent_id in split_ids['valid'] else 'train')
                    if meta['N'] in nbest_ranges[split_id] or meta['N'] > 20000:
                        data_splits[split_id].append({
                            'id': eid,
                            'sentence1': sent,
                            'sentence1_morp': morp,
                            'label': label,
                        })
                        sample_counts[split_id] += 1
                    if meta['N'] == 0 and split_id != 'train':
                        gold_splits[split_id].append([
                            meta['src'],
                            sent_id,
                            sent,
                            morp,
                        ])
            sample_counts_sum = sum(sample_counts.values())
            print(f"- {f'sample_counts':30s} = {sample_counts_sum:11,d} samples")
            for k in ('train', 'valid', 'test'):
                print(f"- {f'sample_counts[{k}]':30s} = {sample_counts[k]:11,d} samples ({percent(sample_counts[k] / sample_counts_sum)})")

        with MyTimer(verbose=True):
            for split_id, data in gold_splits.items():
                if len(data) > 0:
                    outfile = outdir / f'{split_id}.tsv'
                    with outfile.open('w') as out:
                        out.writelines(HT.join(x) + LF for x in data)
                    print(f"{now()} exported {outfile}")
            for split_id, data in data_splits.items():
                if len(data) > 0:
                    outfile = outdir / f'{split_id}.json'
                    with outfile.open('w') as out:
                        json.dump({"version": "datasets_1.0", "data": data}, out, ensure_ascii=False, indent=2)
                    print(f"{now()} exported {outfile}")


def organize_dataset(*predicted_files, origin_dir, output_dir, id_col, plain_col, tagged_col, predict_col, pred_post, top_pred, incl_positive=True, desc_size=128, group_name_size=33, seed=0,
                     to_output_name=lambda x: to_prefix(to_prefix(x.stem), '-') + x.suffix,
                     to_group_name=lambda x: to_prefix(to_prefix(x.stem), '-'),
                     debug=False):
    ran = random.Random(seed)
    tqdm = time_tqdm_cls(bar_size=20, desc_size=desc_size, aline='left')
    origin_dir: Path = Path(origin_dir)
    assert origin_dir.exists() and origin_dir.is_dir(), f"No origin_dir: {origin_dir}"
    all_predicted_files: list = list(chain(*[files(x) for x in predicted_files]))
    assert all_predicted_files, f"No predicted_files: {predicted_files}"
    group_files: dict[str, list] = {k: list(vs) for k, vs in grouped(all_predicted_files, key=to_group_name)}
    if debug:
        print(f"origin_dir={origin_dir}")
    with MyTimer(f"Organize Dataset({', '.join(map(str, predicted_files))}, top_pred={top_pred})", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True):
        for g, (group, grouped_files) in enumerate(group_files.items()):
            origin_file: Path = first_path_or(Path(origin_dir) / f"{group}*.tsv")
            if debug:
                print(f"group={group}")
                print(f"origin_file={origin_file}")
            assert origin_file.exists() and origin_file.is_file(), f"No origin_file: {origin_file}"
            morp_scores: dict = dict()
            for cols in tqdm(tsv_lines(origin_file), total=num_lines(origin_file),
                             pre=f"({g + 1:02d}/{len(group_files):02d}) [{group:<{group_name_size}s}]",
                             desc=f"reading {origin_file}", unit=" lines", file=stdout):
                meta, morp, scores = cols[0], cols[2], tuple(float(x) for x in cols[3:])
                morp_scores[morp]: dict = merge_dicts(morp_scores.get(morp) or dict(), {meta: scores})
            for i, predicted_file in enumerate(grouped_files):
                predicted_file: Path = Path(predicted_file)
                if debug:
                    print(f"predicted_file={predicted_file}")
                all_samples: list = list()
                with predicted_file.open() as inp:
                    keys = inp.readline().strip().split('\t')
                    for line in tqdm(inp.readlines(),
                                     pre=f"{f'[{i + 1:02d}/{len(grouped_files):02d}]':>{group_name_size + 10}s}",
                                     desc=f"reading {predicted_file}", unit=" lines", file=stdout):
                        sample: dict = dict(zip(keys, line.strip().split('\t')))
                        sample[id_col]: dict = dict([s.strip().split('=', maxsplit=1) for s in sample[id_col].split(',')])
                        sample[predict_col]: float = float(sample[predict_col])
                        sample['rank']: int = int(sample[id_col]['N'])
                        sample['ssid']: str = f"{sample[id_col]['src']:*>20s}+{int(sample[id_col]['sid']):010d}"
                        all_samples.append(pop_keys(sample, id_col))
                ssid_samples: dict = {k: list(g) for k, g in grouped(all_samples, itemgetter='ssid')}
                if debug:
                    print(f"#ssid_samples={len(ssid_samples)}")
                output_specs: list = [(t, new_path(make_parent_dir(output_dir / predicted_file.parent.name / to_output_name(predicted_file)), post=f"{pred_post},T{t}", sep='=')) for t in top_pred]
                for j, (t, output_file) in enumerate(output_specs):
                    output_file: Path = Path(output_file)
                    if debug:
                        print(f"output_file={output_file}")
                    num_outputted_ssid = 0
                    with output_file.open("w") as out:
                        target_samples = list(ssid_samples.items()) if not debug else list(ssid_samples.items())[: 10]
                        for ssid, samples in tqdm(target_samples,
                                                  pre=f"{f'[{j + 1:02d}]':>{group_name_size + 10}s}",
                                                  desc=f"writing {output_file}", unit=" sentences", file=stdout):
                            ssid_has_outputs = False
                            samples_with_origin: list = [merge_dicts(x, {"origin_scores": morp_scores[x[tagged_col]]})
                                                         for x in samples if x[tagged_col] in morp_scores]
                            human_tagged: set = {x[tagged_col] for x in samples_with_origin if x['rank'] == 0}
                            if debug:
                                print()
                                out_hr('=')
                                print(f"ssid={ssid} / human_tagged={human_tagged}")
                                print(f"samples_with_origin({len(samples_with_origin)})={LF}- {f'{LF}- '.join(map(str, samples_with_origin))}")
                                out_hr('-')

                            human_samples: list = [x for x in samples_with_origin if x['rank'] == 0]
                            human_samples: list = [first_or(sorted(ss, key=itemgetter('rank'))) for _, ss in grouped(shuffled(human_samples, ran=ran), itemgetter=tagged_col)]  # 중복 제거
                            human_samples: list = sorted(human_samples, key=itemgetter(predict_col), reverse=True)  # 재순위 정렬
                            robot_samples: list = [x for x in samples_with_origin if x['rank'] > 0]
                            robot_samples: list = [first_or(sorted(ss, key=itemgetter('rank'))) for _, ss in grouped(shuffled(robot_samples, ran=ran), itemgetter=tagged_col)]  # 중복 제거
                            robot_samples: list = sorted(robot_samples, key=itemgetter(predict_col), reverse=True)  # 재순위 정렬
                            positive_samples: list = [x for x in robot_samples if x[tagged_col] in human_tagged]
                            negative_samples: list = [x for x in robot_samples if x[tagged_col] not in human_tagged]
                            positive_or_human_samples: list = positive_samples or human_samples
                            if debug:
                                print(f"human_samples({len(human_samples)})={LF}- {f'{LF}- '.join(map(str, human_samples))}")
                                print(f"robot_samples({len(robot_samples)})={LF}- {f'{LF}- '.join(map(str, robot_samples))}")
                                print(f"positive_samples({len(positive_samples)})={LF}- {f'{LF}- '.join(map(str, positive_samples))}")
                                print(f"negative_samples({len(negative_samples)})={LF}- {f'{LF}- '.join(map(str, negative_samples))}")
                                print(f"positive_or_human_samples({len(positive_or_human_samples)})={LF}- {f'{LF}- '.join(map(str, positive_or_human_samples))}")
                                out_hr('-')

                            if incl_positive:  # 정답 태깅이 포함되도록
                                samples_to_retrain: list = positive_or_human_samples + negative_samples[: t - len(positive_or_human_samples)]
                            else:
                                samples_to_retrain: list = robot_samples[: t]
                            if debug:
                                print(f"samples_to_retrain({len(samples_to_retrain)})={LF}- {f'{LF}- '.join(map(str, samples_to_retrain))}")
                                out_hr('-')

                            for sample in samples_to_retrain:
                                for (meta, scores) in sample["origin_scores"].items():
                                    meta_rev = f'{meta}, pred_{pred_post}={sample[predict_col]:.4f}'
                                    scores_rev = [f'{x:.8f}' for x in scores]
                                    print('\t'.join([meta_rev, sample[plain_col], sample[tagged_col], *scores_rev]), file=out)
                                    ssid_has_outputs = True
                            if ssid_has_outputs:
                                num_outputted_ssid += 1
                    output_file.replace(new_path(output_file, post=f"S{num_outputted_ssid}", sep=','))

                if debug:
                    exit(1)
