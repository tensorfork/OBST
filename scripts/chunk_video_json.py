import argparse
import random
import json
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('load_path', type=str,
                        help='The path to a json file containing video information, or a path to a folder containing '
                             'json files with video information.')
    parser.add_argument('min_duration', type=int, help='The Minimum duration a chunk is suppose to contain.')
    parser.add_argument('-prefix', type=str, default='', help='A save file prerfix.')

    args = parser.parse_args()

    load_path = args.load_path
    min_duration = args.min_duration
    prefix = args.prefix

    if os.path.isdir(load_path):
        load_path = [os.path.join(load_path, p) for p in os.listdir(load_path)]
    else:
        load_path = [load_path]

    ids = []
    duration = []

    for l in load_path:
        json_load = json.load(open(l))

        ids = ids + json_load['id']
        duration = duration + json_load['duration']
        
    chunks_ids = []
    chunks_duration = []
    
    _chunk_ids = []
    _chunk_duration = []
    _chunk_duration_sum = 0

    videos = list(zip(ids, duration))
    random.shuffle(videos)
    random.shuffle(videos)
    
    for i, d in videos:
        
        if _chunk_duration_sum < min_duration:
            _chunk_ids.append(i)
            _chunk_duration.append(d)
            _chunk_duration_sum = _chunk_duration_sum + d
        else:

            chunks_ids.append(_chunk_ids)
            chunks_duration.append(_chunk_duration)
            
            _chunk_ids = []
            _chunk_duration = []
            _chunk_duration_sum = 0

    chunks_ids.append(_chunk_ids)
    chunks_duration.append(_chunk_duration)

    ids = chunks_ids
    duration = chunks_duration

    chunk_video_count = 0
    chunk_video_duration = 0

    for i in range(len(ids)):
        buffer_video_count = len(ids[i])
        buffer_video_duration = sum(duration[i])

        print('chunk:', i, 'videos:', buffer_video_count, 'duration:', buffer_video_duration)

        chunk_video_count += buffer_video_count
        chunk_video_duration += buffer_video_duration

    print('')
    print('total num of videos:', chunk_video_count, 'total video duration:', chunk_video_duration)

    path = f"{prefix}work_chunks.json"
    dump = {'id': ids, 'duration': duration}

    json.dump(dump, open(path, 'w'))