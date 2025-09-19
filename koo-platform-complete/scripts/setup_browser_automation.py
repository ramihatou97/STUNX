#!/usr/bin/env python3
"""
Setup script for browser automation in KOO Platform
Installs Playwright browsers and sets up automation environment
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

    return True

def setup_playwright():
    """Install Playwright and browser dependencies"""
    print("🎭 Setting up Playwright browser automation...")

    # Install Playwright browsers
    commands = [
        ("playwright install", "Installing Playwright browsers"),
        ("playwright install-deps", "Installing system dependencies"),
        ("playwright install chromium", "Installing Chromium browser"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Failed to setup Playwright: {description}")
            return False

    return True

def setup_directories():
    """Create necessary directories for browser automation"""
    print("📁 Setting up directories...")

    directories = [
        "data",
        "data/browser_sessions",
        "data/browser_downloads",
        "logs"
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False

    return True

def create_browser_config():
    """Create browser configuration files"""
    print("⚙️ Creating browser configuration...")

    # Browser launch configuration
    browser_config = """
# Browser Automation Configuration for KOO Platform

# Chromium Settings
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
viewport_width = 1920
viewport_height = 1080

# Automation Settings
headless = true
slow_mo = 100  # Milliseconds delay between actions
timeout = 30000  # Default timeout in milliseconds

# Download Settings
download_directory = "./data/browser_downloads"

# Session Settings
user_data_directory = "./data/browser_sessions"
persist_sessions = true

# Security Settings
disable_web_security = false
ignore_https_errors = false
"""

    try:
        with open("browser_config.ini", "w") as f:
            f.write(browser_config)
        print("✅ Browser configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create browser config: {e}")
        return False

def test_browser_setup():
    """Test browser automation setup"""
    print("🧪 Testing browser setup...")

    test_script = """
import asyncio
from playwright.async_api import async_playwright

async def test_browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.google.com")
        title = await page.title()
        await browser.close()
        return title

if __name__ == "__main__":
    try:
        title = asyncio.run(test_browser())
        print(f"✅ Browser test successful! Page title: {title}")
    except Exception as e:
        print(f"❌ Browser test failed: {e}")
        exit(1)
"""

    try:
        with open("test_browser.py", "w") as f:
            f.write(test_script)

        # Run the test
        result = subprocess.run([sys.executable, "test_browser.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Browser automation test passed")
            os.remove("test_browser.py")  # Clean up test file
            return True
        else:
            print(f"❌ Browser test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Browser test setup failed: {e}")
        return False

def create_service_selectors_config():
    """Create configuration file for web service selectors"""
    print("🎯 Creating service selectors configuration...")

    selectors_config = """
# Web Service Selectors for KOO Platform
# Update these if web interfaces change

[gemini]
base_url = "https://gemini.google.com/app"
input_selector = 'div[contenteditable="true"]'
send_button = 'button[aria-label*="Send"]'
response_selector = 'div[data-test-id="conversation-turn-2"] div[class*="markdown"]'
new_chat_button = 'button[aria-label*="New chat"]'
wait_for_response = 30000

[claude]
base_url = "https://claude.ai/chats"
input_selector = 'div[contenteditable="true"]'
send_button = 'button[aria-label="Send Message"]'
response_selector = 'div[class*="font-claude-message"]'
new_chat_button = 'button:has-text("Start new chat")'
wait_for_response = 30000

[perplexity]
base_url = "https://www.perplexity.ai"
input_selector = 'textarea[placeholder*="Ask anything"]'
send_button = 'button[aria-label="Submit"]'
response_selector = 'div[class*="prose"]'
new_chat_button = 'button[aria-label="New Thread"]'
wait_for_response = 30000
"""

    try:
        with open("service_selectors.ini", "w") as f:
            f.write(selectors_config)
        print("✅ Service selectors configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create selectors config: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 KOO Platform Browser Automation Setup")
    print("=" * 50)

    setup_steps = [
        ("Setting up directories", setup_directories),
        ("Installing Playwright", setup_playwright),
        ("Creating browser config", create_browser_config),
        ("Creating selectors config", create_service_selectors_config),
        ("Testing browser setup", test_browser_setup),
    ]

    failed_steps = []

    for step_name, step_function in setup_steps:
        print(f"\n📋 {step_name}...")
        if not step_function():
            failed_steps.append(step_name)

    print("\n" + "=" * 50)

    if not failed_steps:
        print("🎉 Browser automation setup completed successfully!")
        print("\n✅ You can now use hybrid AI services with:")
        print("   • API access (when keys are available)")
        print("   • Web interface automation (as fallback)")
        print("   • Intelligent cost optimization")
        print("\n🔧 Next steps:")
        print("   1. Add your API keys to .env file")
        print("   2. Configure access methods in settings")
        print("   3. Test the hybrid AI manager in the admin panel")

    else:
        print("❌ Setup completed with some failures:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\n🔧 Please resolve the issues and run setup again")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())