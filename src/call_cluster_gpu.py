import subprocess
import configure

model_list = ['efficientnet-b1']
fold_list = [0, 1, 2, 3, 4]

for model in model_list:
    batch_size = configure.config[model]['batch_size']

    for fold in fold_list:
        job_name = f"{model}_fold_{fold}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "12:00:00",
                        "--mem", "60G",
                        "-c", "48",
                        "-A", "rwth0455",
                        "--gres", "gpu:2",
                        "run.zsh", model,
                        str(batch_size),
                        str(fold)])
