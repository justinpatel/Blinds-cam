import os
from twilio.rest import Client
import requests


def sendMsg():
    account_sid = 'ACbca7c3636b10f512dbe9c98a619de6ee'
    auth_token = '359bb891b50475d1527b6a3b9523fbb7'
    client = Client(account_sid, auth_token)

    data = requests.get("https://ipinfo.io/")

    location = data.json()

    body = "Hello, there is an emergency. I am currently at this coordinates "+ location['loc']+ ", "+location['city']+", "+location['region']

    message = client.messages \
                    .create(
                         body=body,
                         from_="+12563986643",
                         to="+917984436056"
                     )

    print(message.sid)
