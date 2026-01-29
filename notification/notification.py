import smtplib
from email.mime.text import MIMEText
import os
import requests
import pandas as pd
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()

def send_mail(subject, text, df=None):
    # email details
    sender = os.getenv("sender")
    password = os.getenv("gmail_password")
    reciver = os.getenv("reciver")

    # create message 
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"]= reciver
    
    # body
    msg.attach(MIMEText(text, "plain"))
    
    # attach excel file 
    if isinstance(df, pd.DataFrame) and not df.empty:
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        
        part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.set_payload(excel_buffer.read())
        encoders.encode_base64(part)
        
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename="report.xlsx"
        )
        msg.attach(part)
        print("Excel Attachment Added")
    else:
        print("No attachment - Dataframe empty or not found")
        
    # connection to gamil SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  #start TLS encryption
        server.login(sender, password)
        server.send_message(msg)
        
    print("Email sent Successfully")




def send_slack_notification(text):
    
    webhook_url = os.getenv("webhook_url")
    
    message = {
        "text": text,
        "username":"AI Market Trend",
        "icon_emoji":":shield:",
    }
    
    requests.post(webhook_url, json=message)
    
    
def testing_function():
    print("hello world")