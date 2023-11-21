import base64
import hashlib
import hmac
import json
import requests
import time
import urllib.parse


def dingtalk_reminder(webhook, secret=None, **kwargs):
    webhook_signed = None
    timestamp = str(round(time.time() * 1000))
    if secret is not None:
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        webhook_signed = webhook + '&timestamp=' + timestamp + '&sign=' + sign
    original_text = "#### 完成情况 \n> 已完成\n"
    headers = {'Content-Type': 'application/json'}
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": kwargs.get("title", "任务"),
            "text": kwargs.get("text", original_text)
        },
        "at": {
            "atMobiles": kwargs.get("mobiles", None),
            "isAtAll": False
        }
    }
    if secret is not None:
        res = requests.post(webhook_signed, data=json.dumps(data), headers=headers)
    else:
        res = requests.post(webhook, data=json.dumps(data), headers=headers)
    return res
