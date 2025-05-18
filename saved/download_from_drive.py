from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from auth import authenticate
import logging
from pathlib import Path

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ID della cartella Google Drive da cui scaricare
FOLDER_ID = '1BBaKDyp6Fh_0dFI-Q40JSYXeCQ4utKCK'

def download_file(drive, filename: str, output_dir: Path = Path(".")) -> bool:
    """Scarica un file da Google Drive dato il nome e salva nella cartella specificata."""
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{FOLDER_ID}' in parents and trashed=false"
        }).GetList()

        if file_list:
            gfile = file_list[0]
            output_path = output_dir / filename
            gfile.GetContentFile(str(output_path))
            logging.info(f"✅ File '{filename}' scaricato in '{output_path}'.")
            return True
        else:
            logging.warning(f"❌ File '{filename}' non trovato nella cartella Drive.")
            return False
    except Exception as e:
        logging.error(f"❌ Errore durante il download di '{filename}': {e}")
        return False

def download_files(file_list: list[str], output_dir: str = "."):
    """Scarica una lista di file da Google Drive nella cartella specificata."""
    drive = authenticate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename in file_list:
        download_file(drive, filename, output_path)

if __name__ == "__main__":
    files_to_download = [
        "encoder.pt",
        "encoder_best.pt",
        "encoder_last.pt",
        "decoder.pt",
        "decoder_best.pt",
        "decoder_last.pt",
        "vocab.pkl"
    ]
    download_files(files_to_download, output_dir=".")
