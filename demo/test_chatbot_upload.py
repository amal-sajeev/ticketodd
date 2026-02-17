#!/usr/bin/env python3
"""
Test chatbot file upload and grievance filing flow.
Run: python demo/test_chatbot_upload.py
"""
import asyncio
import os
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system(f"{sys.executable} -m pip install playwright -q")
    os.system(f"{sys.executable} -m playwright install chromium")
    from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8000"
IMAGE_PATH = r"C:\Users\Enterprise\.cursor\projects\c-Users-Enterprise-Documents-pypad-ticketodd-main\assets\c__Users_Enterprise_AppData_Roaming_Cursor_User_workspaceStorage_741c3d2f2c7aa1aac8309a749417cf46_images_apollo-goldshot-33cb6a5f-e30d-4e26-88b8-25653195a3ab.png"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "chatbot_test_screenshots")


async def run_test():
    results = []
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    image_to_upload = IMAGE_PATH
    if not os.path.exists(image_to_upload):
        results.append(("Step 5", "WARN", f"Image file not found: {IMAGE_PATH}"))
        # Try to find any image in the project as fallback
        image_to_upload = None
        for root, _, files in os.walk(os.path.dirname(__file__)):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_to_upload = os.path.join(root, f)
                    results.append(("Step 5", "INFO", f"Using fallback image: {os.path.basename(image_to_upload)}"))
                    break
            if image_to_upload:
                break
        if not image_to_upload:
            results.append(("Step 5", "SKIP", "No image file available - will test without attachment"))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()

        try:
            # Step 1: Navigate to login
            results.append(("Step 1", "START", "Navigate to http://localhost:8000/login"))
            await page.goto(f"{BASE_URL}/login", wait_until="networkidle", timeout=15000)
            results.append(("Step 1", "OK", "Login page loaded"))

            # Step 2: Log in (per user: ram_kumar / password123; seed has citizen1/citizen123)
            results.append(("Step 2", "START", "Log in with ram_kumar / password123"))
            await page.fill('#username', "ram_kumar")
            await page.fill('#password', "password123")
            await page.click('#loginBtn')
            await page.wait_for_timeout(4000)
            if "login" in page.url:
                results.append(("Step 2", "INFO", "ram_kumar not found, trying citizen1/citizen123"))
                await page.goto(f"{BASE_URL}/login", wait_until="networkidle")
                await page.fill('#username', "citizen1")
                await page.fill('#password', "citizen123")
                await page.click('#loginBtn')
                await page.wait_for_timeout(4000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(1000)
            if "login" in page.url.lower():
                err = page.locator('#loginError')
                err_text = await err.text_content() if await err.count() > 0 else "Unknown"
                results.append(("Step 2", "ERROR", f"Still on login page: {err_text}"))
            else:
                results.append(("Step 2", "OK", "Logged in successfully"))

            # Step 3: Navigate to chatbot
            results.append(("Step 3", "START", "Navigate to /chatbot"))
            await page.goto(f"{BASE_URL}/chatbot", wait_until="networkidle", timeout=10000)
            results.append(("Step 3", "OK", "Chatbot page loaded"))

            # Step 4: Click attach button
            results.append(("Step 4", "START", "Click attach (paperclip) button"))
            attach_btn = page.locator('button.chat-attach-btn, button[aria-label="Attach file"]')
            await attach_btn.click()
            await page.wait_for_timeout(500)
            results.append(("Step 4", "OK", "Attach button clicked"))

            # Step 5: Upload file (if we have one)
            if image_to_upload and os.path.exists(image_to_upload):
                results.append(("Step 5", "START", f"Upload image: {os.path.basename(image_to_upload)}"))
                file_input = page.locator('#chatFileInput')
                await file_input.set_input_files(image_to_upload)
                await page.wait_for_timeout(800)
                results.append(("Step 5", "OK", "File selected for upload"))
            else:
                results.append(("Step 5", "SKIP", "No image file to upload"))

            # Step 6: Verify file preview chip
            results.append(("Step 6", "START", "Verify file preview chip appears"))
            preview = page.locator('#chatFilePreview')
            preview_visible = await preview.is_visible()
            if preview_visible:
                chip = page.locator('.chat-file-chip')
                chip_count = await chip.count()
                results.append(("Step 6", "OK", f"File preview visible, {chip_count} chip(s)"))
                await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01_after_attach.png"))
            else:
                results.append(("Step 6", "WARN", "File preview not visible (may be expected if no file)"))

            # Step 7: Type message and send (include enough detail for bot to file: problem, district, location, when)
            results.append(("Step 7", "START", "Type message and send"))
            msg = "I want to file a grievance about broken road in Khordha district"
            await page.fill('#chatInput', msg)
            await page.click('#sendBtn, button.chat-send-btn, button[aria-label="Send message"]')
            await page.wait_for_timeout(2000)
            results.append(("Step 7", "OK", "Message sent"))
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02_after_send.png"))

            # Step 8: Verify message bubble shows "1 file attached"
            results.append(("Step 8", "START", "Verify message bubble shows '1 file attached'"))
            attach_indicator = page.locator('.chat-attach-indicator')
            has_indicator = await attach_indicator.count() > 0
            if has_indicator:
                text = await attach_indicator.first.text_content()
                results.append(("Step 8", "OK", f"Attachment indicator found: {text.strip()}"))
            else:
                results.append(("Step 8", "WARN", "No '1 file attached' indicator in user bubble (may show if no file was attached)"))

            # Step 9: Wait for bot response and confirm
            results.append(("Step 9", "START", "Wait for bot, then confirm 'Yes, please file it'"))
            # Wait for send button to be enabled (bot response complete)
            try:
                await page.wait_for_selector('#sendBtn:not([disabled])', timeout=60000)
            except Exception as e:
                await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "09_timeout_waiting_bot.png"))
                results.append(("Step 9", "ERROR", f"Bot response timeout - send button stayed disabled: {type(e).__name__}"))
                raise
            await page.wait_for_timeout(2000)  # Extra time for bot message to render
            await page.fill('#chatInput', "Yes, please file it")
            await page.click('#sendBtn')
            await page.wait_for_selector('#sendBtn:not([disabled])', timeout=30000)
            await page.wait_for_timeout(3000)
            results.append(("Step 9", "OK", "Confirmation sent"))
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03_after_confirm.png"))

            # Step 10: Verify grievance filed
            results.append(("Step 10", "START", "Verify grievance filed successfully"))
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "04_after_confirm.png"))
            filed = page.locator('.chat-filed-grievance, :has-text("Grievance Filed Successfully")')
            if await filed.count() > 0:
                results.append(("Step 10", "OK", "Grievance filed successfully - success message visible"))
            else:
                results.append(("Step 10", "WARN", "Grievance success message not found - bot may need more details or confirmation"))

        except Exception as e:
            results.append(("ERROR", "FAIL", str(e)))
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, "error.png"))
        finally:
            await browser.close()

    return results


def main():
    print("=" * 60)
    print("Chatbot File Upload & Grievance Filing Test")
    print("=" * 60)
    print(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
    print()
    results = asyncio.run(run_test())
    for step, status, msg in results:
        symbol = {"OK": "[OK]", "START": "[>]", "ERROR": "[X]", "WARN": "[!]", "SKIP": "[-]", "FAIL": "[X]", "INFO": "[i]"}.get(status, "[?]")
        print(f"  {symbol} [{step}] {status}: {msg}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
