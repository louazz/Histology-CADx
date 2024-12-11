import kagglehub, shutil

path = kagglehub.dataset_download("mrsaurov/warwick-qu-dataset")

shutil.copytree(path+"/Warwick QU Dataset", "ds/images")

