import argparse
import json
import os

from tqdm import tqdm

if __name__ == '__main__':
    json_name = "person_keypoints_train2017"
    json_dir = "person_keypoints"

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file',
                        default='/users/l00522665/datasets/coco/annotations/' + json_name + '.json',
                        type=str, help="coco file path")
    parser.add_argument('--save_dir', default='/users/l00522665/datasets/coco/yolov5/' + json_dir, type=str,
                        help="where to save .txt labels")
    arg = parser.parse_args()
    data = json.load(open(arg.json_file, 'r'))

    # 如果存放txt文件夹不存在，则创建
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir)

    train_txt_path = json_name + ".txt"
    with open(os.path.join(arg.save_dir, train_txt_path), 'w') as f:
        for img in tqdm(data['images']):
            # 解析 images 字段，分别取出图片文件名、图片的宽和高、图片id
            filename = img["file_name"]
            f.writelines("/users/l00522665/datasets/coco/train2017/" + filename + '\r\n')


