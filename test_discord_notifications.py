"""
Test Discord notification setup
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

print("=" * 70)
print("DISCORD NOTIFICATION TEST")
print("=" * 70)

# Check if webhook URL is set
webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

if not webhook_url:
    print("\n❌ DISCORD_WEBHOOK_URL not set!")
    print("\n📝 To fix this:")
    print("1. Create a .env file in your project root")
    print("2. Add the following line:")
    print('   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...')
    print("\n3. Run this test again:")
    print("   python test_discord_notification.py")
    exit(1)

print("\n✅ DISCORD_WEBHOOK_URL is set")
print(f"   URL: {webhook_url[:50]}...")

# Test sending notification
print("\n🔄 Sending test notification to Discord...")

from workflows.notifications import send_discord_webhook

test_message = f"""
🧪 **TEST NOTIFICATION**

⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ If you see this message in Discord, your notifications are working!

🎯 **What this means:**
   • Discord webhook is configured correctly
   • Prefect pipeline can send notifications
   • You'll get alerts when training completes

🚀 **Next step:** Run your Prefect pipeline!
   `python workflows/prefect_pipeline.py`
"""

success = send_discord_webhook(test_message)

if success:
    print("\n✅ SUCCESS! Check your Discord channel!")
    print("\n🎉 Your Discord notifications are ready!")
else:
    print("\n❌ Failed to send notification")
    print("\n🔍 Troubleshooting:")
    print("1. Check if the webhook URL is correct")
    print("2. Make sure the webhook hasn't been deleted")
    print("3. Check your internet connection")

print("\n" + "=" * 70)
