"""
Discord notification helpers for Prefect workflows
"""

import os
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment variables from .env
# ------------------------------------------------------------------
load_dotenv()  # <-- THIS is the key fix

logger = logging.getLogger(__name__)


def send_discord_webhook(message: str, webhook_url: str = None) -> bool:
    """
    Send notification to Discord via webhook.

    Parameters:
    -----------
    message : str
        Message to send
    webhook_url : str, optional
        Discord webhook URL.
        If not provided, DISCORD_WEBHOOK_URL is read from .env / environment.

    Setup:
    ------
    1. Discord Server → Settings → Integrations → Webhooks
    2. Create webhook and copy URL
    3. Add to .env:
       DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    """

    webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.error("❌ DISCORD_WEBHOOK_URL is not set")
        print("❌ DISCORD_WEBHOOK_URL not found. Check your .env file.")
        return False

    payload = {
        "content": message,
        "username": "ML Pipeline Bot"
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)

        if response.status_code == 204:
            logger.info("✅ Discord notification sent successfully")
            print("✅ Discord notification sent")
            return True
        else:
            logger.error(
                f"❌ Discord notification failed "
                f"(status={response.status_code}, response={response.text})"
            )
            print(f"❌ Discord failed: {response.status_code}")
            return False

    except Exception as e:
        logger.exception(f"❌ Error sending Discord notification: {e}")
        print(f"❌ Exception: {e}")
        return False


# ------------------------------------------------------------------
# Test mode
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("DISCORD NOTIFICATION TEST")
    print("=" * 70)

    print("🔍 Loaded DISCORD_WEBHOOK_URL:",
          "FOUND" if os.getenv("DISCORD_WEBHOOK_URL") else "NOT FOUND")

    test_message = f"""
🧪 **TEST NOTIFICATION**

⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ If you see this message, Discord notifications are working!
"""

    success = send_discord_webhook(test_message)

    if success:
        print("\n🎉 SUCCESS — Check your Discord channel!")
    else:
        print("\n❌ FAILED — Check webhook URL and logs")

    print("=" * 70)
