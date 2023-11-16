import os
import subprocess
from modules.api.firebase_datastore import DataStore

ROOT_DIR = os.path.expanduser('~')
WORK_FOLDER = "dreambooth_training"
WORK_DIR = os.path.join(ROOT_DIR, WORK_FOLDER)
AUTO1111_MODEL_DIR = os.path.join(ROOT_DIR, "stable-diffusion-webui/models/Lora")
COLAB = False
XFORMERS = True
BETTER_EPOCH_NAMES = True
LOAD_TRUNCATED_IMAGES = True
BUCKET_NAME = "mestyle-app"


class DreamboothModelTrainer:
    
    project_name = ""
    model_file = ""
    model_folder = ""
    custom_dataset = None
    override_dataset_config_file = None
    override_config_file = None
    optimizer = "AdamW8bit"
    optimizer_args = None
    continue_from_lora = ""
    weighted_captions = False
    adjust_tags = False
    keep_tokens_weight = 1.0
    
    old_model_url = ""
    # Model: realspice (https://civitai.com/models/158734/realspice)
    model_url = "https://civitai.com/api/download/models/208629?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    base_model_file = os.path.join(ROOT_DIR, "base_models/v1-5-pruned-emaonly.safetensors")
    custom_model_is_based_on_sd2 = False
    
    dependencies_installed = False

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

    main_dir = ""
    deps_dir = ""
    repo_dir = ""
    hugging_face_cache_dir = ""
    images_folder = ""
    output_folder = ""
    config_folder = ""
    log_folder = ""
    config_file = ""
    dataset_config_file = ""
    accelerate_config_file = ""

    datastore = None

    def __init__(self):
        self.datastore = DataStore()


    def validate_dataset(self):
        import toml
        import time

        time.sleep(2)
        supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        print("\n💿 Checking dataset...")

        # TODO: Set custom_dataset so it can recognize and validate reg folder.
        if self.custom_dataset:
            try:
                datconf = toml.loads(self.custom_dataset)
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
            folders = [self.images_folder]
            files = os.listdir(self.images_folder)
            images_repeats = {self.images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), self.num_repeats)}

        for f in files:
            if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
                print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
                return

        if not [txt for txt in files if txt.lower().endswith(".txt")]:
            caption_extension = ""
        if self.continue_from_lora and not (self.continue_from_lora.endswith(".safetensors") and os.path.exists(self.continue_from_lora)):
            print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
            return

        pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
        print("pre_steps_per_epoch:", pre_steps_per_epoch)
        steps_per_epoch = pre_steps_per_epoch/self.train_batch_size
        print("steps_per_epoch:", steps_per_epoch)
        print("self.train_batch_size:", self.train_batch_size)
        total_steps = self.max_train_steps or int(self.max_train_epochs*steps_per_epoch)
        print("self.max_train_steps:", self.max_train_steps)
        print("int(self.max_train_epochs*steps_per_epoch):", int(self.max_train_epochs*steps_per_epoch))
        print("total_steps:", total_steps)
        estimated_epochs = int(total_steps/steps_per_epoch)
        lr_warmup_steps = int(total_steps* self.lr_warmup_ratio)

        for folder, (img, rep) in images_repeats.items():
            print("📁"+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
            print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
        print(f"📉 Divide {pre_steps_per_epoch} steps by {self.train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
        if self.max_train_epochs:
            print(f"🔮 There will be {self.max_train_epochs} epochs, for around {total_steps} total training steps.")
        else:
            print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

        if total_steps > 10000:
            print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
            return

        if self.adjust_tags:
            print(f"\n📎 Weighted tags: {'ON' if self.weighted_captions else 'OFF'}")
            if self.weighted_captions:
                print(f"📎 Will use {self.keep_tokens_weight} weight on {self.keep_tokens} activation tag(s)")
            print("📎 Adjusting tags...")
            self.adjust_weighted_tags(folders, self.keep_tokens, self.keep_tokens_weight, self.weighted_captions)

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


    def download_model(self):
        import re
        real_model_url = self.model_url.strip()

        if real_model_url.lower().endswith((".ckpt", ".safetensors")):
            # self.model_file = f"/content{real_model_url[real_model_url.rfind('/'):]}"
            self.model_file = f"{real_model_url[real_model_url.rfind('/'):]}"
            self.model_file = os.path.join(self.model_folder, self.model_file[1:])
        else:
            # self.model_file = "/content/downloaded_model.safetensors"
            self.model_file = "/downloaded_model.safetensors"
            self.model_file = os.path.join(self.model_folder, self.model_file[1:])
            if os.path.exists(self.model_file):
                subprocess.call(["rm", "{}".format(self.model_file)])

        if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", self.model_url):
            real_model_url = real_model_url.replace("blob", "resolve")
        elif m := re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)", self.model_url):
            real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"

        print("DOWNLOAD TO:", self.model_file)
        subprocess.call([
            "cp",
            "{}".format(self.base_model_file),
            "{}".format(self.model_file),
        ])
        '''
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
            "{}".format(self.model_file),
        ])
        '''

        if self.model_file.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors
            try:
                test = load_safetensors(self.model_file)
                del test
            except Exception as e:
                #if "HeaderTooLarge" in str(e):
                new_model_file = os.path.splitext(self.model_file)[0]+".ckpt"
                subprocess.call(["mv", "{}".format(self.model_file), "{}".format(new_model_file)])
                self.model_file = new_model_file
                print(f"Renamed model to {os.path.splitext(self.model_file)[0]}.ckpt")

        if self.model_file.lower().endswith(".ckpt"):
            from torch import load as load_ckpt
            try:
                test = load_ckpt(self.model_file)
                del test
            except Exception as e:
                return False
        
        return True


    def create_config(self):
        import toml

        # TODO: currently this is not use. Please try to use config_file in the training script.
        config_file = self.config_file
        if self.override_config_file:
            config_file = self.override_config_file
            print(f"\n⭕ Using custom config file {self.config_file}")
        else:
            config_dict = {
                "additional_network_arguments": {
                    "unet_lr": self.unet_lr,
                    "text_encoder_lr": self.text_encoder_lr,
                    "network_dim": self.network_dim,
                    "network_alpha": self.network_alpha,
                    "network_module": self.network_module,
                    "network_args": self.network_args,
                    "network_train_unet_only": True if self.text_encoder_lr == 0 else None,
                    "network_weights": self.continue_from_lora if self.continue_from_lora else None
                },
                "optimizer_arguments": {
                    "learning_rate": self.unet_lr,
                    "lr_scheduler": self.lr_scheduler,
                    "lr_scheduler_num_cycles": self.lr_scheduler_num_cycles if self.lr_scheduler == "cosine_with_restarts" else None,
                    "lr_scheduler_power": self.lr_scheduler_power if self.lr_scheduler == "polynomial" else None,
                    "lr_warmup_steps": self.lr_warmup_steps if self.lr_scheduler != "constant" else None,
                    "optimizer_type": self.optimizer,
                    "optimizer_args": self.optimizer_args if self.optimizer_args else None,
                },
                "training_arguments": {
                    "max_train_steps": self.max_train_steps,
                    "max_train_epochs": self.max_train_epochs,
                    "save_every_n_epochs": self.save_every_n_epochs,
                    "save_last_n_epochs": self.keep_only_last_n_epochs,
                    "train_batch_size": self.train_batch_size,
                    "noise_offset": None,
                    "clip_skip": 2,
                    "min_snr_gamma": self.min_snr_gamma_value,
                    "weighted_captions": self.weighted_captions,
                    "seed": 42,
                    "max_token_length": 225,
                    "xformers": XFORMERS,
                    "lowram": COLAB,
                    "max_data_loader_n_workers": 8,
                    "persistent_data_loader_workers": True,
                    "save_precision": "fp16",
                    "mixed_precision": "fp16",
                    "output_dir": self.output_folder,
                    "logging_dir": self.log_folder,
                    "output_name": self.project_name,
                    "log_prefix": self.project_name,
                },
                "model_arguments": {
                    "pretrained_model_name_or_path": self.model_file,
                    "v2": self.custom_model_is_based_on_sd2,
                    "v_parameterization": True if self.custom_model_is_based_on_sd2 else None,
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

        dataset_config_file = self.dataset_config_file
        if self.override_dataset_config_file:
            dataset_config_file = self.override_dataset_config_file
            print(f"⭕ Using custom dataset config file {dataset_config_file}")
        else:
            dataset_config_dict = {
                "general": {
                    "resolution": self.resolution,
                    "shuffle_caption": self.shuffle_caption,
                    "keep_tokens": self.keep_tokens,
                    "flip_aug": self.flip_aug,
                    "caption_extension": self.caption_extension,
                    "enable_bucket": True,
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False,
                    "min_bucket_reso": 320 if self.resolution > 640 else 256,
                    "max_bucket_reso": 1280 if self.resolution > 640 else 1024,
                },
                "datasets": toml.loads(self.custom_dataset)["datasets"] if self.custom_dataset else [
                    {
                        "subsets": [
                            {
                                "num_repeats": self.num_repeats,
                                "image_dir": self.images_folder,
                                "class_tokens": None if self.caption_extension else self.project_name
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


    def train(self, ref_id: str, model_name: str, dataset_dir: str):
        from time import time
        
        self.project_name = model_name

        '''Prepare folders structure'''
        self.main_dir = os.path.join(ROOT_DIR, WORK_FOLDER, ref_id)
        self.repo_dir = WORK_DIR
        self.hugging_face_cache_dir = os.path.join(self.main_dir, "cache")
        self.model_folder = os.path.join(self.main_dir, "model")
        self.images_folder = os.path.join(self.main_dir, "datasets/img")
        # TODO: Generate regularization images from a model.
        self.reg_images_folder = os.path.join(self.main_dir, "datasets/reg_img")
        self.output_folder = os.path.join(self.main_dir, "output")
        self.config_folder = os.path.join(self.main_dir, "config")
        self.log_folder = os.path.join(self.main_dir, "log")
        for dir in (self.main_dir, self.repo_dir, self.hugging_face_cache_dir, self.model_folder, self.images_folder, self.reg_images_folder, self.output_folder, self.config_folder, self.log_folder):
            os.makedirs(dir, exist_ok=True)
            
        self.config_file = os.path.join(self.config_folder, "training_config.toml")
        self.dataset_config_file = os.path.join(self.config_folder, "dataset_config.toml")
        self.accelerate_config_file = os.path.join(self.repo_dir, "accelerate_config/config.yaml")
        print("\n# Folders have been prepared")

        '''Copy all dataset into the training folder structure'''
        print("IMAGE FOLDER:", self.images_folder)
        subprocess.Popen(
            "cp {src} {dst}".format(src=os.path.join(dataset_dir, "*"), dst=self.images_folder), 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("\nAll images are copied to `datasets` folder")

        if not self.validate_dataset():
            return

        if self.old_model_url != self.model_url or not self.model_file or not os.path.exists(self.model_file):
            print("\n🔄 Downloading model...")
            if not self.download_model():
                print("\n💥 Error: The model you selected is invalid or corrupted, or couldn't be downloaded. You can use a civitai or huggingface link, or any direct download link.")
                return
            print()
        else:
            print("\n🔄 Model already downloaded.\n")

        self.create_config()

        print("\n⭐ Starting trainer...\n")
        
        os.chdir(self.repo_dir)

        proc = subprocess.call([
            "sudo",
            "docker",
            "run",
            "--rm",
            "--gpus=all",
            "-v",
            "{}:/work".format(self.main_dir),
            "-v", 
            ###
            "{}:/home/user/.cache/huggingface/hub".format(self.hugging_face_cache_dir),
            "aoirint/sd_scripts",
            "--num_cpu_threads_per_process=1",
            "train_network.py",
            "--pretrained_model_name_or_path={}".format(self.model_file.replace(self.main_dir, '/work')),
            "--dataset_config={}".format(self.dataset_config_file.replace(self.main_dir, '/work')),
            "--output_dir={}".format(self.output_folder.replace(self.main_dir, '/work')),
            "--output_name={}".format(self.project_name),
            "--save_model_as=safetensors",
            "--logging_dir=/work/logs",
            "--prior_loss_weight=1.0",
            "--max_train_steps=400",
            "--learning_rate=1e-4",
            '--optimizer_type=AdamW8bit',
            "--xformers",
            '--mixed_precision=fp16',
            "--cache_latents",
            "--gradient_checkpointing",
            "--save_every_n_epochs=1",
            "--network_module=networks.lora",
            "--v2",
            "--v_parameterization",
        ]) 
        print('Ran command: {}'.format(proc.args));

        '''Once completed, copy trained model to the folder'''
        model_file_name = "{}.safetensors".format(self.project_name)
        trained_model_file = os.path.join(self.output_folder, model_file_name)
        model_s3_path = "s3://{bucket}/models/{ref}/{file}".format(
            bucket=BUCKET_NAME, ref=ref_id, file=model_file_name,
        )
        print(f"📄 Trained model file: " + trained_model_file)
        print(f"📄 Automatic1111 model directory: " + AUTO1111_MODEL_DIR)
        if os.path.exists(trained_model_file):
            print(f"🔄 Saving trained model to automatic1111...")
            subprocess.call([
                "cp",
                "{}".format(trained_model_file),
                "{}".format(AUTO1111_MODEL_DIR),
            ])
            print(f"✅ Saved.")

            '''Backup model file on S3'''
            
            subprocess.call([
                "aws",
                "s3",
                "cp",
                "{}".format(trained_model_file),
                "{}".format(model_s3_path),
            ])

        if not os.path.exists(os.path.join(AUTO1111_MODEL_DIR, model_file_name)):
            print(f"⭕ Error: trained model not found in output folder.")
        
        '''Clean up training environment'''
        subprocess.call([
            "rm",
            "-rf",
            "{}".format(self.main_dir)
        ])

        '''Update status in Firebase to be `done`'''
        doc = self.datastore.get_doc(collection="models", key=ref_id)
        if doc is not None:
            doc["status"] = "done"
            doc["modelPath"] = "{}".format(model_s3_path)
        self.datastore.set_doc(collection="models", key=ref_id, data=doc)

        return os.path.join(AUTO1111_MODEL_DIR, (self.project_name + ".safetensors"))


if __name__ == "__main__":
    pass
    ''' Local test '''
    # datastore = DataStore()
    # trainer = LoraModelTrainer(datastore)
    # trainer.train(
    #     ref_id="00000000-0000-0000-0000-000000000000",
    #     model_name="mestyle-first-model",
    #     dataset_dir="/home/ubuntu/images/00000000-0000-0000-0000-000000000000",
    # )
