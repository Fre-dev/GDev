import os 
from composio import Composio

user_id = "default"
composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
result = composio.tools.execute(
    "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL",
    user_id=user_id,
    arguments={
        "channel": "C09537EJ5JB",
        "text": "Hello from simple Composio example! 🚀"
    }
)
print(result)