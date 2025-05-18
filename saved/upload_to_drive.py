from pydrive2.drive import GoogleDrive
from auth import authenticate
import logging
from pathlib import Path

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FOLDER_ID = '1BBaKDyp6Fh_0dFI-Q40JSYXeCQ4utKCK'  # ID della cartella Google Drive

def upload_file(drive: GoogleDrive, filepath: Path, folder_id: str = FOLDER_ID) -> bool:
    """Carica un file su Google Drive nella cartella specificata. Elimina versioni precedenti."""
    if not filepath.exists():
        logging.warning(f"❌ Il file '{filepath}' non esiste.")
        return False

    try:
        # Elimina eventuali file con lo stesso nome nella stessa cartella
        file_list = drive.ListFile({
            'q': f"title='{filepath.name}' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        for old_file in file_list:
            logging.info(f"🗑️ Cancello file esistente: {old_file['title']}")
            old_file.Delete()

        # Crea nuovo file
        gfile = drive.CreateFile({'title': filepath.name, 'parents': [{'id': folder_id}]})
        gfile.SetContentFile(str(filepath))
        gfile.Upload()

        # Permessi pubblici di sola lettura
        gfile.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})

        shareable_link = f"https://drive.google.com/file/d/{gfile['id']}/view?usp=sharing"
        logging.info(f"✅ Caricato: {filepath.name}")
        logging.info(f"🔗 Link: {shareable_link}")
        return True

    except Exception as e:
        logging.error(f"❌ Errore durante l'upload di '{filepath.name}': {e}")
        return False

def upload_files(file_list: list[str], input_dir: str = "."):
    """Carica una lista di file da una directory locale su Google Drive."""
    drive = authenticate()
    input_path = Path(input_dir)

    for filename in file_list:
        filepath = input_path / filename
        upload_file(drive, filepath)

if __name__ == "__main__":
    files_to_upload = [
        "encoder.pt",
        "encoder_best.pt",
        "encoder_last.pt",
        "decoder.pt",
        "decoder_best.pt",
        "decoder_last.pt"
    ]
    upload_files(files_to_upload, input_dir=".")
