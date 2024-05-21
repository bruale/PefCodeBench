# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import smtplib

def send_experimentcompleted_email(exper_id: str):
    # -----------------------------------------------------------------------------
    #  Function to send email once the recalibration execution is completed
    # ----------------------------------------------------------------------------
    confs = {
        'user': 'set_user_email',
        'pwd': 'set_pwd',
        'recipients': ['set_recipient']
    }
    email_user = confs['user']
    email_password = confs['pwd']
    sent_from = email_user
    to = confs['recipients']

    text_body = ''
    body = text_body

    subject = 'Done: ' + str(exper_id)

    email_text = """\
            From: %s
            To: %s
            Subject: %s
            %s
            """ % (sent_from, ", ".join(to), subject, body)

    try:
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server.login(email_user, email_password)
        smtp_server.sendmail(sent_from, to, email_text)
        smtp_server.close()
    except Exception as ex:
        print("Error while sending email….", ex)