# MineDreamer Setup

The entire project environment is composed of two parts: one is the MineRL environment for Agent, and the other is the Imaginator environment for Imaginator. The interaction between the Agent in MineRL and Imaginator is accomplished through backend communication.

**The contained files in this repository are for installing the MineRL container environment by using provided container and are unrelated to the normal installation procedures for MineRL.**

# MineRL Env Setup
Due to the challenges in setting up the MineRL environment, we provide two different methods to install the MineRL environment:
1. Normal Installation Procedure (Recommend):
    - Advantage: It's flexible, allowing installation of desired packages through apt or pip at any time.
    - Disadvantage: Approximately 7GB of memory is necessary to compile the MineRL environment, which is hard for many machines.
        - **Suppose you encounter any errors related to `gradlew` and `gradle` while installing MineRL. In that case, it is likely due to insufficient memory on your machine, preventing the compilation of the MineRL environment from completing.**
2. Using the Provided Apptainer Container:
    - Advantage: Ensures a runnable environment and eliminates potential errors during the installation process. It also supports headless GPU rendering by VGL, which is  faster than CPU rendering.
    - Disadvantage: Any additional packages requiring apt or pip installation will necessitate container modification which may be time-consuming.
    - **Disadvantage 2: Even if your machine has a head, it can only operate in a headless manner.**

## Method 1: Normal Installation Procedure (Recommend)
<details> <summary>Expand to view Method 1.</summary>

We recommend running on linux using a conda environment, with python 3.10.
1. Install PyTorch 2.0: `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
2. Install MineRL: `pip install git+https://github.com/Zhoues/minerl`
    - See [MineRL Installation](https://minerl.readthedocs.io/en/latest/tutorials/index.html) for more details on how to setup MineRL
    - Note: We choose not to use the official MineRL repository. This decision is made because our repository adds the feature of `chat` action on top of MineRL, which allows setting agent initialization conditions through commands, making it more convenient for us to test the Agent.
3. Install MineDojo and MineCLIP: 
    ```bash
    pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
    pip install git+https://github.com/Zhoues/MineCLIP
    ```
    - See [MineDojo Installation](https://docs.minedojo.org/sections/getting_started/install.html) for more details such as setting **the correct Java version**
    - Note: We don't opt for the official MineDojo repository, as it has a specific requirement for the `importlib_resources` version during installation which is prone to errors. Hence, we provide a personal revised version instead.
4. Install VPT requirements: `pip install gym==0.19 gym3 attrs opencv-python`
    - Note: At the time of writing, MineDojo and VPT require different versions of gym. Please use the gym version required by VPT (gym==0.19). If the installation steps are run in the order listed here, the correct gym version will be installed at the end of setup (since VPT requirements are installed after MineDojo).
5. Install additional requirements: 
    ```bash
    ## additional requirements
    pip install -U gdown tqdm accelerate wandb pyyaml ipdb openai langchain chromadb tiktoken pyyaml
    pip install -U langchain-openai langchain-community langchain-experimental open_clip_torch

    ## Fix the bug of MineCLIP for python=3.10
    conda install chardet
    pip install importlib_resources==5.0.0
    ```
6. Git clone the MineDreamer repo and install minedreamer locally with: `pip install -e .` .


### Running on a headless server
If you are running on a headless server, you need to install `xvfb` and run each python script with `xvfb-run`. For example, `xvfb-run python script_name.py`.

Also, notice that we use the MineRL environment, not the MineDojo environment. Thus, setting `MINEDOJO_HEADLESS=1` as mentioned in the 'MineDojo Installation' instructions will have no effect.
</details>


## Method 2: Using the Provided Apptainer Container
<details> <summary>Expand to view Method 2.</summary>

We provide a pre-compiled Apptainer container. Compared to Docker, the Apptainer container requires fewer permissions making it suitable for use in a cluster environment like slurm.

1. Install `gdown` and download the pre-compiled Apptainer container, which is compiled by the `base-vgl-env.def`:
    ```bash
    pip install -U --no-cache-dir gdown --pre

    gdown https://drive.google.com/uc?id=1cOF5Bf6DEvuLXMrY-JCVT2XxDHDosgqU -O base-vgl-env.sif
    ```
    - Note: This environment has already completed steps 1 to 5 of the Normal Installation Procedure，and it supports headless GPU rendering by VGL. **If you are faced with `Too many users have viewed or downloaded this file recently...`, please email to me and I will give you another link, or use the method like [this](https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work)**
2. Clone this repo and move `vgl-env.def`, `base-vgl-env.sif`, `setupvgl.sh` of this repo to the same level as minedreamer in the MineDreamer Repo, which you should first git clone: 
    ```
    MineDreamer
    ├── README.md
    ├── minedreamer
    │   ├── agent code.
    ├── imaginator
    │   ├── imaginator code.
    ├── vgl-env.def: Install final container locally, and if you want to install additional package, just modify this file
    ├── base-vgl-env.sif: pre-compiled Apptainer container
    ├── setupvgl.sh: supports headless GPU rendering by VGL
    ```
3. Supplement the necessary information in the `vgl-env.def`.
    ```text
    ...
    BootStrap: localimage
    # Based on the base apptainer (Check Carefully!!)
    From: /path/to/MineDreamer/base-vgl-env.sif # The absolute path of base-vgl-env.sif. Need to check.

    %post
    # Go into git clone code root location (Check Carefully!!)
    cd /path/to/MineDreamer # The absolute path of MineDreamer folder. Need to check.
    ...
    ```
4. Compile the final Apptainer environment: 
    -  If you are using a standalone machine: `sudo apptainer build  --bind /MineDreamer_top_level_directory:/MineDreamer_top_level_directory vgl-env.sif vgl-env.def`. If you are using a cluster like slurm: `srun -p <your virtual partition> apptainer build ...(like above)`

    - `--bind`: The local directory you want to mount additionally in the container. `/MineDreamer_top_level_directory` means that if MineDreamer absolute path is like `/home/xxx/MineDreamer`, it should be `/home`.
    - Note: After compiling, there will be a `vgl-env.sif` in the folder. **If you want to install additional package in the container, you can add the bash command after `# Install local package` part in `vgl-env.def` and re-compile this final container.**

5. Turn the final container into writable sandbox, because minerl env will create running logs while `.sif` is read only.
    -  If you are using a standalone machine: `sudo apptainer build --sandbox vgl-env/ vgl-env.sif` 
    -  If you are using a cluster like slurm: `srun -p  <your virtual partition> apptainer build ...(like above)`


### Running on a headless server (even if your server has a head)
If you are running on a headless server, you can use `sudo apptainer exec -w --nv --bind /MineDreamer_top_level_directory:/MineDreamer_top_level_directory vgl-env xvfb-run /opt/conda/envs/minerl/bin/python script_name.py`.
- `--bind`: The local directory you want to mount additionally in the container. `/MineDreamer_top_level_directory` means that if MineDreamer absolute path is like `/home/xxx/MineDreamer`, it should be `/home`.
- `--nv`: nvidia flag to enable GPU capabilities. **If you don't use GPU, please remove it**. (some warnings will raise but ignore them)
- `-w`: enable writting inside the container.
- If you are using Wandb, you will need to specify the Wandb API key for remote monitoring. This can be done adding the following flag to the above command before vgl-env: `... --env WANDB_API_KEY=XXXXXXXXXXXXXXXXX vgl-env...`.

If you want to use GPU rendering, you need to create a script `script_name.sh` for your `script_name.py` like the following:
```bash
vglrun /opt/conda/envs/minerl/bin/python script_name.py # GPU Rendering
```
and then use `sudo apptainer exec -w --nv --bind /MineDreamer_top_level_directory:/MineDreamer_top_level_directory vgl-env bash setupvgl.sh script_name.sh`.

**It's worth noting that you should likely change the permissions of `script_name.sh` to `+x` using `chmod`.**

</details>



## 3. Test your MineRL Env

<details> <summary>Expand to view Test Method.</summary>


In MineDreamer repo, there is an `minerl_env_valid.py` to test the environment
- For standard installation and the server is headful, please test using the following command: `python minerl_env_valid.py`.
- For standard installation and the server is headless, please test using the following command: `xvfb-run python minerl_env_valid.py`.
- If you have installed via apptainer container, please test using the following command: `sudo apptainer exec -w --bind /MineDreamer_top_level_directory:/MineDreamer_top_level_directory vgl-env xvfb-run /opt/conda/envs/minerl/bin/python minerl_env_valid.py` and cluster is similar.

If an image appears in the current directory, showing an agent in front of a sheep holding a diamond axe, this means the environment has been successfully installed.

</details>
