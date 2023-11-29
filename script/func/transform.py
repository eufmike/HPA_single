import torchvision.transforms as transforms

transform = transforms.Compose(
    [
    transforms.Resize((2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])  
