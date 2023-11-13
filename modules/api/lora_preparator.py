import os
import subprocess

ROOT_DIR = "/home/ubuntu"
# ROOT_DIR = "/Users/apirat/Documents/GitHub/MeStyle"
KOHYA_DIR = "/".join([ROOT_DIR, "kohya_trainer"])
KOHYA_REPO = "https://github.com/kohya-ss/sd-scripts"
TAG_THRESHOLD = 0.35
CAPTION_MIN = 10
CAPTION_MAX = 75


class LoraDatasetPreparator:


    def __environment_preparation():
        os.chdir(ROOT_DIR)
        if not os.path.exists(KOHYA_DIR):
            subprocess.call(["git", "clone", KOHYA_REPO, KOHYA_DIR])
            os.chdir(KOHYA_DIR)
            subprocess.call(["git", "reset", "--hard", "5050971ac687dca70ba0486a583d283e8ae324e2"])
        os.chdir(ROOT_DIR)


    def __install_dependencies():
        pass


    def __generate_tags(images_folder: str):
        os.chdir(KOHYA_DIR)
        os.environ["PYTHONPATH"] = KOHYA_DIR
        subprocess.call([
            "python", 
            "".join([KOHYA_DIR, "/finetune/make_captions.py"]),
            "{}".format(images_folder),
            "--beam_search",
            "--max_data_loader_n_workers=2",
            "--batch_size=8",
            "--min_length={}".format(CAPTION_MIN),
            "--max_length={}".format(CAPTION_MAX),
            "--caption_extension=.txt",
        ])

        import random
        captions = [f for f in os.listdir(images_folder) if f.lower().endswith(".txt")]
        sample = []
        for txt in random.sample(captions, min(10, len(captions))):
            with open(os.path.join(images_folder, txt), 'r') as f:
                sample.append(f.read())

        os.chdir(ROOT_DIR)
        print(f"📊 Captioning complete. Here are {len(sample)} example captions from your dataset:")
        print("".join(sample))


    def tag_images(self, image_dir: str):
        self.__environment_preparation()
        self.__install_dependencies()
        self.__generate_tags(images_folder=image_dir)


if __name__ == "__main__":
    pass
