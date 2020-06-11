import subprocess
import configure

model_list = ['efficientnet-b1']
fold_list = [0]

for model in model_list:
    batch_size = configure.config[model]['batch_size']
    image_width = configure.config[model]['image_width']
    image_height = configure.config[model]['image_height']

    for fold in fold_list:
        job_name = f"{model}_{image_width}_{image_height}_fold_{fold}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "12:00:00",
                        "--mem", "60G",
                        "-c", "24",
                        "-A", "rwth0455",
                        "--gres", "gpu:1",
                        "run.zsh", model,
                        str(image_width),
                        str(image_height),
                        str(batch_size),
                        str(fold)])
