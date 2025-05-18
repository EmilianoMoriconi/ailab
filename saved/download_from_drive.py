from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from auth import authenticate

# Inserisci qui l'ID della cartella di Drive
FOLDER_ID = '1BBaKDyp6Fh_0dFI-Q40JSYXeCQ4utKCK'


def download_file(filename):
    drive = authenticate()

    # Cerca il file con quel nome nella cartella specificata
    file_list = drive.ListFile({
        'q': f"title='{filename}' and '{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    if file_list:
        gfile = file_list[0]
        gfile.GetContentFile(filename)
        print(f"✅ File '{filename}' scaricato con successo.")
    else:
        print(f"❌ File '{filename}' non trovato nella cartella Drive.")

if __name__ == "__main__":
    download_file("encoder.pt")
    download_file("encoder_best.pt")
    download_file("encoder_last.pt")
    download_file("decoder.pt")
    download_file("decoder_best.pt")
    download_file("decoder_last.pt")