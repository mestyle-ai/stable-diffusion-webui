import os
import subprocess

# ROOT_DIR = "/home/ubuntu"
ROOT_DIR = "/Users/apirat/Documents/GitHub/MeStyle"
KOHYA_DIR = "/".join([ROOT_DIR, "kohya_trainer"])
KOHYA_REPO = "https://github.com/kohya-ss/sd-scripts"
COMMIT = "9a67e0df390033a89f17e70df5131393692c2a55"

project_name = ""
model_file = None

# These may be set by other cells, some are legacy
if "custom_dataset" not in globals():
    custom_dataset = None
if "override_dataset_config_file" not in globals():
    override_dataset_config_file = None
if "override_config_file" not in globals():
    override_config_file = None
if "optimizer" not in globals():
    optimizer = "AdamW8bit"
if "optimizer_args" not in globals():
    optimizer_args = None
if "continue_from_lora" not in globals():
    continue_from_lora = ""
if "weighted_captions" not in globals():
    weighted_captions = False
if "adjust_tags" not in globals():
    adjust_tags = False
if "keep_tokens_weight" not in globals():
    keep_tokens_weight = 1.0

COLAB = False
XFORMERS = True
BETTER_EPOCH_NAMES = True
LOAD_TRUNCATED_IMAGES = True

# project_name = ""
#@markdown The folder structure doesn't matter and is purely for comfort. Make sure to always pick the same one. I like organizing by project.
folder_structure = "Organize by project (MyDrive/Loras/project_name/dataset)" #@param ["Organize by category (MyDrive/lora_training/datasets/project_name)", "Organize by project (MyDrive/Loras/project_name/dataset)"]
#@markdown Decide the model that will be downloaded and used for training. These options should produce clean and consistent results. You can also choose your own by pasting its download link.
# TODO: 
model_url = ""
custom_model_is_based_on_sd2 = False

resolution = 512
flip_aug = False
caption_extension = ".txt"
shuffle_tags = True
shuffle_caption = shuffle_tags
activation_tags = "1"
keep_tokens = int(activation_tags)
num_repeats = 10
preferred_unit = "Epochs"
how_many = 10
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
save_every_n_epochs = 1
keep_only_last_n_epochs = 10
if not save_every_n_epochs:
    save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
    keep_only_last_n_epochs = max_train_epochs
train_batch_size = 2
unet_lr = 5e-4
text_encoder_lr = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_scheduler_number = 3
lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0
lr_warmup_ratio = 0.05
lr_warmup_steps = 0
min_snr_gamma = True
min_snr_gamma_value = 5.0 if min_snr_gamma else None
lora_type = "LoRA"
network_dim = 16
network_alpha = 8
conv_dim = 8
conv_alpha = 4

network_module = "networks.lora"
network_args = None
if lora_type.lower() == "locon":
    network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]

if optimizer.lower() == "prodigy" or "dadapt" in optimizer.lower():
    if override_values_for_dadapt_and_prodigy:
        unet_lr = 0.5
        text_encoder_lr = 0.5
        lr_scheduler = "constant_with_warmup"
        lr_warmup_ratio = 0.05
        network_alpha = network_dim

    if not optimizer_args:
        optimizer_args = ["decouple=True","weight_decay=0.01","betas=[0.9,0.999]"]
        if optimizer == "Prodigy":
            optimizer_args.extend(["d_coef=2","use_bias_correction=True"])
        if lr_warmup_ratio > 0:
            optimizer_args.append("safeguard_warmup=True")
        else:
            optimizer_args.append("safeguard_warmup=False")


deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "kohya-trainer")


main_dir = os.path.join(ROOT_DIR, "lora_training")


main_dir = os.path.join()
images_folder = os.path.join(main_dir, "datasets", project_name)
output_folder = os.path.join(main_dir, "output", project_name)
config_folder = os.path.join(main_dir, "config", project_name)
log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
accelerate_config_file = os.path.join(repo_dir, "accelerate_config/config.yaml")

class LoraModelTrainer:

    def __environment_preparation():
        os.chdir(ROOT_DIR)
        subprocess.call(["git", "clone", KOHYA_REPO, KOHYA_DIR])
        os.chdir(KOHYA_DIR)
        if COMMIT:
           subprocess.call(["git", "reset", "--hard", COMMIT])
        subprocess.call([
           "wget",
           "https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/xformers-fix/requirements.txt",
           "-q",
           "-O",
           "requirements.txt",
        ])
        os.chdir(ROOT_DIR)


    def __install_dependencies():
        # clone_repo()
        # !apt -y update -qq
        # !apt -y install aria2 -qq
        # !pip install --upgrade -r requirements.txt
        # if XFORMERS:
        #     !pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

        # # patch kohya for minor stuff
        # if COLAB:
        #     !sed -i "s@cpu@cuda@" library/model_util.py # low ram
        # if LOAD_TRUNCATED_IMAGES:
        #     !sed -i 's/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py # fix truncated jpegs error
        # if BETTER_EPOCH_NAMES:
        #     !sed -i 's/{:06d}/{:02d}/g' library/train_util.py # make epoch names shorter
        #     !sed -i 's/"." + args.save_model_as)/"-{:02d}.".format(num_train_epochs) + args.save_model_as)/g' train_network.py # name of the last epoch will match the rest

        # from accelerate.utils import write_basic_config
        # if not os.path.exists(accelerate_config_file):
        #     write_basic_config(save_location=accelerate_config_file)

        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        # os.environ["SAFETENSORS_FAST_GPU"] = "1"
        pass


    def validate_dataset():
        import toml
        global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
        supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        print("\n💿 Checking dataset...")


        if custom_dataset:
            try:
                datconf = toml.loads(custom_dataset)
                datasets = [d for d in datconf["datasets"][0]["subsets"]]
            except:
                print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
                return
            reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
            datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
            folders = datasets_dict.keys()
            files = [f for folder in folders for f in os.listdir(folder)]
            images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
        else:
            reg = []
            folders = [images_folder]
            files = os.listdir(images_folder)
            images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

        for folder in folders:
            if not os.path.exists(folder):
                print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
                return
        for folder, (img, rep) in images_repeats.items():
            if not img:
                print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
                return
        for f in files:
            if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
                print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
                return

        if not [txt for txt in files if txt.lower().endswith(".txt")]:
            caption_extension = ""
        if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
            print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
            return

        pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
        steps_per_epoch = pre_steps_per_epoch/train_batch_size
        total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
        estimated_epochs = int(total_steps/steps_per_epoch)
        lr_warmup_steps = int(total_steps*lr_warmup_ratio)

        for folder, (img, rep) in images_repeats.items():
            print("📁"+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
            print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
        print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
        if max_train_epochs:
            print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
        else:
            print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

        if total_steps > 10000:
            print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
            return

        if adjust_tags:
            print(f"\n📎 Weighted tags: {'ON' if weighted_captions else 'OFF'}")
            if weighted_captions:
                print(f"📎 Will use {keep_tokens_weight} weight on {keep_tokens} activation tag(s)")
            print("📎 Adjusting tags...")
            adjust_weighted_tags(folders, keep_tokens, keep_tokens_weight, weighted_captions)

        return True

    def adjust_weighted_tags(folders, keep_tokens: int, keep_tokens_weight: float, weighted_captions: bool):
        import re
        weighted_tag = re.compile(r"\((.+?):[.\d]+\)(,|$)")
        for folder in folders:
            for txt in [f for f in os.listdir(folder) if f.lower().endswith(".txt")]:
                with open(os.path.join(folder, txt), 'r') as f:
                    content = f.read()
                # reset previous changes
                content = content.replace('\\', '')
                content = weighted_tag.sub(r'\1\2', content)
                if weighted_captions:
                    # re-apply changes
                    content = content.replace(r'(', r'\(').replace(r')', r'\)').replace(r':', r'\:')
                    if keep_tokens_weight > 1:
                        tags = [s.strip() for s in content.split(",")]
                        for i in range(min(keep_tokens, len(tags))):
                            tags[i] = f'({tags[i]}:{keep_tokens_weight})'
                        content = ", ".join(tags)
                with open(os.path.join(folder, txt), 'w') as f:
                    f.write(content)


    def download_model():
        import re
        global old_model_url, model_url, model_file
        real_model_url = model_url.strip()

        if real_model_url.lower().endswith((".ckpt", ".safetensors")):
            model_file = f"/content{real_model_url[real_model_url.rfind('/'):]}"
        else:
            model_file = "/content/downloaded_model.safetensors"
            if os.path.exists(model_file):
                subprocess.call(["rm", "{}".format(model_file)])

        if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", model_url):
            real_model_url = real_model_url.replace("blob", "resolve")
        elif m := re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)", model_url):
            real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"

        subprocess.call([
            "aria2c",
            "{}".format(real_model_url),
            "--console-log-level=warn",
            "-c",
            "-s",
            "16",
            "-x",
            "16",
            "-k",
            "10M",
            "-d",
            "/",
            "-o",
            "{}".format(model_file),
        ])

        if model_file.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors
            try:
                test = load_safetensors(model_file)
                del test
            except Exception as e:
                #if "HeaderTooLarge" in str(e):
                new_model_file = os.path.splitext(model_file)[0]+".ckpt"
                subprocess.call(["mv", "{}".format(model_file), "{}".format(new_model_file)])
                model_file = new_model_file
                print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

        if model_file.lower().endswith(".ckpt"):
            from torch import load as load_ckpt
            try:
                test = load_ckpt(model_file)
                del test
            except Exception as e:
                return False
        
        return True


    def create_config():
        import toml
        global dataset_config_file, config_file, model_file

        if override_config_file:
            config_file = override_config_file
            print(f"\n⭕ Using custom config file {config_file}")
        else:
            config_dict = {
                "additional_network_arguments": {
                    "unet_lr": unet_lr,
                    "text_encoder_lr": text_encoder_lr,
                    "network_dim": network_dim,
                    "network_alpha": network_alpha,
                    "network_module": network_module,
                    "network_args": network_args,
                    "network_train_unet_only": True if text_encoder_lr == 0 else None,
                    "network_weights": continue_from_lora if continue_from_lora else None
                },
                "optimizer_arguments": {
                    "learning_rate": unet_lr,
                    "lr_scheduler": lr_scheduler,
                    "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
                    "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
                    "lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,
                    "optimizer_type": optimizer,
                    "optimizer_args": optimizer_args if optimizer_args else None,
                },
                "training_arguments": {
                    "max_train_steps": max_train_steps,
                    "max_train_epochs": max_train_epochs,
                    "save_every_n_epochs": save_every_n_epochs,
                    "save_last_n_epochs": keep_only_last_n_epochs,
                    "train_batch_size": train_batch_size,
                    "noise_offset": None,
                    "clip_skip": 2,
                    "min_snr_gamma": min_snr_gamma_value,
                    "weighted_captions": weighted_captions,
                    "seed": 42,
                    "max_token_length": 225,
                    "xformers": XFORMERS,
                    "lowram": COLAB,
                    "max_data_loader_n_workers": 8,
                    "persistent_data_loader_workers": True,
                    "save_precision": "fp16",
                    "mixed_precision": "fp16",
                    "output_dir": output_folder,
                    "logging_dir": log_folder,
                    "output_name": project_name,
                    "log_prefix": project_name,
                },
                "model_arguments": {
                    "pretrained_model_name_or_path": model_file,
                    "v2": custom_model_is_based_on_sd2,
                    "v_parameterization": True if custom_model_is_based_on_sd2 else None,
                },
                "saving_arguments": {
                    "save_model_as": "safetensors",
                },
                "dreambooth_arguments": {
                    "prior_loss_weight": 1.0,
                },
                "dataset_arguments": {
                    "cache_latents": True,
                },
            }

        for key in config_dict:
            if isinstance(config_dict[key], dict):
                config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

        with open(config_file, "w") as f:
            f.write(toml.dumps(config_dict))
        print(f"\n📄 Config saved to {config_file}")

        if override_dataset_config_file:
            dataset_config_file = override_dataset_config_file
            print(f"⭕ Using custom dataset config file {dataset_config_file}")
        else:
            dataset_config_dict = {
                "general": {
                    "resolution": resolution,
                    "shuffle_caption": shuffle_caption,
                    "keep_tokens": keep_tokens,
                    "flip_aug": flip_aug,
                    "caption_extension": caption_extension,
                    "enable_bucket": True,
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False,
                    "min_bucket_reso": 320 if resolution > 640 else 256,
                    "max_bucket_reso": 1280 if resolution > 640 else 1024,
                },
                "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
                    {
                        "subsets": [
                            {
                                "num_repeats": num_repeats,
                                "image_dir": images_folder,
                                "class_tokens": None if caption_extension else project_name
                            }
                        ]
                    }
                ]
            }

        for key in dataset_config_dict:
            if isinstance(dataset_config_dict[key], dict):
                dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

        with open(dataset_config_file, "w") as f:
            f.write(toml.dumps(dataset_config_dict))
        print(f"📄 Dataset config saved to {dataset_config_file}")


    def train(self, model_name: str, dataset_dir: str):
        import time
        global dependencies_installed

        '''Prepare folders structure'''
        main_dir = os.path.join(ROOT_DIR, "lora_training", model_name)
        images_folder = os.path.join(main_dir, "datasets", project_name)
        output_folder = os.path.join(main_dir, "output", project_name)
        config_folder = os.path.join(main_dir, "config", project_name)
        log_folder = os.path.join(main_dir, "log")
        for dir in (main_dir, deps_dir, repo_dir, log_folder, images_folder, output_folder, config_folder):
            os.makedirs(dir, exist_ok=True)

        '''Copy all dataset into the training folder structure'''
        # TODO:
        

        if not self.validate_dataset():
            return

        if not dependencies_installed:
            print("\n🏭 Installing dependencies...\n")
            t0 = time()
            self.install_dependencies()
            t1 = time()
            dependencies_installed = True
            print(f"\n✅ Installation finished in {int(t1-t0)} seconds.")
        else:
            print("\n✅ Dependencies already installed.")

        if old_model_url != model_url or not model_file or not os.path.exists(model_file):
            print("\n🔄 Downloading model...")
            if not self.download_model():
                print("\n💥 Error: The model you selected is invalid or corrupted, or couldn't be downloaded. You can use a civitai or huggingface link, or any direct download link.")
                return
            print()
        else:
            print("\n🔄 Model already downloaded.\n")

        self.create_config()

        print("\n⭐ Starting trainer...\n")
        os.chdir(repo_dir)

        subprocess.call([
            "accelerate",
            "launch",
            "--config_file={}".format(accelerate_config_file),
            "--num_cpu_threads_per_process=1",
            "train_network.py",
            "--dataset_config={}".format(dataset_config_file),
            "--config_file={}".format(config_file),
        ])


if __name__ == "__main__":
    # Local test
    trainer = LoraModelTrainer()
    trainer.train(
        model_name="lora_unique_model",
        dataset_dir="/Users/apirat/Desktop/NaRaYa/images",
    )
