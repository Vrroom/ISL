from isl_utils import *
import json
import argparse

if __name__ == "__main__" : 
    df = cached_read_csv(VIDEO_HASH_METADATA)

    # process the words in the include set
    include_mask = ['include' in _ for _ in df['path']]
    include_data = df[include_mask]
    include_words = []
    for _ in include_data['path'] : 
        include_words.append(_.split('.')[1].split('/')[0].strip().lower())
    include_text_map = dict(zip(include_data['hash'], include_words))

    # process the words for the islrtc set
    islrtc_mask = ['islrtc' in _ for _ in df['path']] 
    islrtc_data = df[islrtc_mask] 
    path_keys = [getBaseName(_).split('.')[0] for _ in islrtc_data['path']]
    islrtc_jsons = [osp.join(VIDEO_ISLRTC_JSONS, f'{k}.json') for k in path_keys]
    islrtc_words = []
    for json_file in islrtc_jsons :
        with open(json_file) as fp : 
            title = json.load(fp)['title']
        islrtc_words.append(title)
    islrtc_text_map = dict(zip(islrtc_data['hash'], islrtc_words))

    # process the words for the rkm set
    rkm_mask = ['rkm' in _.split('/') for _ in df['path']] 
    rkm_data = df[rkm_mask] 
    path_keys = [getBaseName(_).split('.')[0] for _ in rkm_data['path']]
    rkm_jsons = [osp.join(VIDEO_RKM_JSONS, f'{k}.json') for k in path_keys]
    hash_to_path = dict(zip(df['hash'], df['path']))
    rkm_words = []
    for i, json_file in enumerate(rkm_jsons) : 
        with open(json_file) as fp :
            title = json.load(fp)['title']
        rkm_words.append(title)
    rkm_text_map = dict(zip(rkm_data['hash'], rkm_words))

    # process the words for the split_rkm set
    split_rkm_mask = ['split_rkm' in _.split('/') for _ in df['path']] 
    split_rkm_data = df[split_rkm_mask] 
    path_keys = [getBaseName(_).split('.')[0] for _ in split_rkm_data['path']]
    split_rkm_words = []
    for k in path_keys : 
        title = rkm_text_map[k.split('_')[0]]
        split_rkm_words.append(title)
    split_rkm_text_map = dict(zip(split_rkm_data['hash'], split_rkm_words))

    # merge all dictionaries
    final_dict = merge_dicts([include_text_map, islrtc_text_map, rkm_text_map, split_rkm_text_map])
    print(len(final_dict))
    data = {'hash': final_dict.keys(), 'text': final_dict.values()}
    df = pd.DataFrame(data) 
    # write it out
    df.to_csv(osp.join(ROOT, 'metadata', 'unnormalized_text.csv'), index=False)
