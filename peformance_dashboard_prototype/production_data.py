# production_data.py - Replace generate_test_data() with:

import gspread
from google.oauth2 import service_account

def load_production_data():
    # Authenticate with Google Sheets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    
    gc = gspread.authorize(credentials)
    
    # Open your Google Sheet
    spreadsheet = gc.open_by_key(st.secrets["spreadsheet_id"])
    worksheet = spreadsheet.sheet1
    
    # Get all data
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    
    return df