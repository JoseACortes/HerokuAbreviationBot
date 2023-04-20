import tarfile
import os

def tar_folder(output_filename: str, source_dir: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

OUT_FILE = 'BaseModel.tar.gz'

SOURCE_FILE = "AbreviationBot/HerokuAbreviationBot/AbreivBot/Models/TrainedModels/BaseModel/1681952286"

tar_folder(output_filename=OUT_FILE, source_dir=SOURCE_FILE)