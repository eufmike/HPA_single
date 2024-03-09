conda create -y -n mspytorch python=3.9
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install -y tqdm tensorboard pandas scikit-learn scikit-image matplotlib albumentations jupyterlab

# 03/08/2024
conda install mamba
mamba install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -y tqdm tensorboard pandas scikit-learn scikit-image matplotlib albumentations jupyterlab

