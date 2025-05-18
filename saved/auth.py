from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import json

def authenticate():
    gauth = GoogleAuth()

    # Forza refresh token
    gauth.settings['get_refresh_token'] = True
    gauth.settings['oauth_scope'] = ['https://www.googleapis.com/auth/drive']
    gauth.settings['client_config_file'] = 'client_secrets.json'

    gauth.LoadCredentialsFile("mycreds.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("mycreds.json")
    # Riformatta il file JSON su più righe con indentazione
    with open("mycreds.json", "r") as f:
        creds = json.load(f)

    with open("mycreds.json", "w") as f:
        json.dump(creds, f, indent=4)


    return GoogleDrive(gauth)


if __name__ == "__main__":
    authenticate()
    print("Autenticazione completata.")