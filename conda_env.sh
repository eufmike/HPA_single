conda create -n mspytorch python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm tensorboard pandas scikit-learn scikit-image matplotlib albumentations