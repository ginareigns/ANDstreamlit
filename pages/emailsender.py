import streamlit as st

import smtplib
from email.message import EmailMessage

def send_email(subject, body, to):
    # Define your SMTP email server details
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'threatguardian.info@gmail.com'  # Use environment variables or Streamlit secrets to store credentials
    smtp_password = 'emsl okpt vawy njgp'  # Use environment variables or Streamlit secrets to store credentials
    smtp_from_email = smtp_username  # Email from address

    # Create the email message
    message = EmailMessage()
    message.set_content(body)
    message['Subject'] = subject
    message['From'] = smtp_from_email
    message['To'] = to

    # Send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to secure
        server.login(smtp_username, smtp_password)
        server.send_message(message)
        server.quit()
        return True
    except Exception as e:
        print(e)
        return False








# Streamlit form to get user input
with st.form('email_form'):
    to_email = st.text_input('Enter the recipient email address')
    email_subject = st.text_input('Enter the email subject')
    email_body = st.text_area('Enter the email body')
    submit_button = st.form_submit_button('Send Email')

    if submit_button:
        if send_email(email_subject, email_body, to_email):
            st.success('Email sent successfully!')
        else:
            st.error('An error occurred while sending the email.')
