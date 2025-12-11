import kagglehub

# Download latest version
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas")

print("Path to dataset files:", path)
