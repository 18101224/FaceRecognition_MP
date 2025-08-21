import requests, os 

def send_message(message):
    try: 
        data = {
            'token': os.getenv('alarm_app'),
            'user': os.getenv('alarm_user'),
            'message': message
        }
        requests.post(os.getenv('URL'), data=data)
        return True
    except Exception as e:
        print(e)
        return False
