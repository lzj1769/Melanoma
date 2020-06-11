import subprocess

model_list = ['efficientnet-b0']
fold_list = [0]

batch_size = {'se_resnext50_32x4d': 5,
              'inceptionv4': 8,
              'efficientnet-b0': 64,
              'efficientnet-b1': 5}


for model in model_list:
    for fold in fold_list:
        job_name = f"{model}_fold_{fold}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "12:00:00",
                        "--mem", "180G",
                        "-c", "48",
                        "-A", "rwth0455",
                        "--gres", "gpu:2",
                        "run.zsh", model,
                        str(fold), str(batch_size[model])])
