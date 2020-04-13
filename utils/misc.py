import os
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)

def get_module_name(obj):
    try:
        return re.findall(r"[A-Za-z0-9]+'",str(type(obj)))[0].replace("'",'')
    except:
        return "Name not found"

def upload_google_sheets(upload_dict, spreadsheet_id = '1nliojVYnyy-42Sy-OFy3rB28dOdNakTA06oHa-Gyg9c', sheet_name = 'CV_Record', logger = None):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # Authentication
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Get Current Row Number and headings
    sheet = service.spreadsheets()
    currentRowNumber = len(sheet.values().get(spreadsheetId=spreadsheet_id, range=f'{sheet_name}!1:1000').execute()['values'])
    headings = sheet.values().get(spreadsheetId=spreadsheet_id, range=f'{sheet_name}!1:1').execute()['values'][0]

    upload_list = []
    for header in headings:
        if header in upload_dict.keys():
            upload_list.append(str(upload_dict[header]))
        else:
            upload_list.append('')

    # Upload to Google Sheets
    body = {
        'values': [upload_list]
    }
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=f'{sheet_name}!{currentRowNumber+1}:{currentRowNumber+1}',
        valueInputOption='USER_ENTERED', body=body).execute()
    log_print('Results Uploaded', logger)