from PIL import Image
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default='./data/imagenet50_4/train',
                        help='root directory of the imagenet dataset (containing "train","val" directories')
    parser.add_argument('--out-path', default='./data/raw',
                        help='root directory of the imagenet dataset (containing "train","val" directories')
    parser.add_argument('--height', default=256, help='height of output image')
    parser.add_argument('--width', default=256, help='width of output image')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    for folder in ['train','val']:
        subfolders = os.listdir(os.path.join(args.root_path, folder))
        for subfolder in subfolders:
            subfolder_path = os.path.join(args.root_path, folder, subfolder)
            images = os.listdir(subfolder_path)
            for image_name in images:
                image_path = os.path.join(subfolder_path, image_name)
                image = Image.open(image_path)
                if image.size[0] < args.width or image.size[1] < args.height:
                    continue
                cropped = image.crop((0,0,args.width,args.height))
                cropped.save(os.path.join(args.out_path,image_name))
            print(subfolder_path)

if __name__ == "__main__":
    main()
