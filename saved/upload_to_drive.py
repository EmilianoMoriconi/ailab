from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

FOLDER_ID = '1BBaKDyp6Fh_0dFI-Q40JSYXeCQ4utKCK'  # ID della cartella di Google Drive

def authenticate():
    gauth = GoogleAuth()

    # Prova a caricare le credenziali salvate (se esistono)
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        # Primo accesso: serve login via browser
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Token scaduto: viene aggiornato
        gauth.Refresh()
    else:
        # Token valido: usa quello salvato
        gauth.Authorize()

    # Salva o aggiorna le credenziali
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)

def upload_file(filename):
    drive = authenticate()

    # Elimina vecchio file con lo stesso nome(se esiste)
    file_list = drive.ListFile({
        'q': f"title='{filename}' and '{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    for old_file in file_list:
        print(f"🗑️ Cancello: {old_file['title']}")
        old_file.Delete()

    # Carica nuovo file
    gfile = drive.CreateFile({'title': filename, 'parents': [{'id': FOLDER_ID}]})
    gfile.SetContentFile(filename)
    gfile.Upload()
    gfile.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
    print(f"✅ Caricato: {filename}")
    print(f"🔗 Link: https://drive.google.com/file/d/{gfile['id']}/view?usp=sharing")

if __name__ == "__main__":
    upload_file("encoder.pt")
    upload_file("encoder_best.pt")
    upload_file("encoder_last.pt")
    upload_file("decoder.pt")
    upload_file("decoder_best.pt")
    upload_file("decoder_last.pt")
