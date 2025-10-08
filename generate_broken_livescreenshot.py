import os
import time
import random
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- Hardcoded config ----------
# Tip: if you're testing a local file you upload, point this to a file:// URL:
# BASE_URL = "file:///ABSOLUTE/PATH/TO/your_file.html"
BASE_URL = "https://www.google.com"
TEST_IMAGES_DIR = "./dataset/test_images"
PAGE_READY_SELECTOR = (By.TAG_NAME, "button")  # tweak if your page differs
# -------------------------------------

os.makedirs(TEST_IMAGES_DIR, exist_ok=True)


def _new_driver():
    opts = Options()
    opts.add_argument("start-maximized")
    opts.add_argument("--headless=new")
    return webdriver.Chrome(options=opts)


def _inject_layout_bugs(driver):
    # Hide the Google Logo
    driver.execute_script("""
        var logo = document.querySelector('[aria-label="Google"]');
        if (logo && Math.random() < 0.7) {
            logo.style.display = "none";
        }
    """)

    # Break search box
    driver.execute_script("""
        var searchbox = document.querySelector('input[name="q"]');
        if (searchbox) {
            if (Math.random() < 0.5) {
                searchbox.style.marginTop = "150px";
            } else {
                searchbox.style.visibility = "hidden";
            }
        }
    """)

    # Break search buttons
    driver.execute_script("""
        var buttons = document.querySelectorAll('input[type="submit"], button');
        buttons.forEach(function(btn) {
            if (btn && Math.random() < 0.8) {
                var choice = Math.random();
                if (choice < 0.3) {
                    btn.disabled = true;  // Disable 30%
                } else if (choice < 0.6) {
                    btn.style.marginLeft = Math.floor(Math.random() * 200) + "px";  // Random margin shift
                } else if (choice < 0.9) {
                    btn.style.width = "50px";  // Shrink button width
                    btn.style.height = "20px"; // Shrink button height
                }
            }
        });
    """)

    # Break links
    driver.execute_script("""
        var links = Array.from(document.querySelectorAll('a'))
            .filter(link => link.offsetParent !== null && link.innerText.trim().length > 0);

        links.forEach(function(link) {
            if (Math.random() < 0.5) {  // 50% of visible links
                var choice = Math.random();
                if (choice < 0.5) {
                    link.style.visibility = "hidden";
                } else {
                    link.style.marginLeft = Math.floor(Math.random() * 200) + "px";
                }
            }
        });
    """)

    # Randomly remove footer
    driver.execute_script("""
        var footer = document.getElementById('fbar');
        if (footer && Math.random() < 0.7) {
            footer.style.display = "none";
        }
    """)

    print("⚡ Heavy layout breaks applied.")


def take_screenshot(url: str, kind: str = "good", filename: str | None = None) -> str:
    """
    Take a single screenshot.
      kind: "good" for clean page, "bad" (aka "broken"/"buggy") to inject bugs first.
      filename: optional file name (e.g., "myshot.png"). If None, a timestamped name is used.
    Returns: absolute path to the saved PNG.
    """
    kind_norm = kind.strip().lower()
    is_bad = kind_norm in {"bad", "broken", "buggy"}

    driver = _new_driver()
    driver.set_window_size(1920, 1080)
    wait = WebDriverWait(driver, 10)
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located(PAGE_READY_SELECTOR))
        time.sleep(2.0)  # tiny settle

        if is_bad:
            _inject_layout_bugs(driver)
            time.sleep(0.5)  # let styles apply

        # Name the file
        if not filename:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{ts}_{'bad' if is_bad else 'good'}.png"

        out_path = os.path.abspath(os.path.join(TEST_IMAGES_DIR, filename))
        driver.save_screenshot(out_path)
        print(f"Saved {'BAD' if is_bad else 'GOOD'} screenshot -> {out_path}")
        return out_path
    finally:
        driver.quit()

# ---------------- Example usage ----------------
# take_screenshot("good")            # one clean screenshot into ./test_images
# take_screenshot("bad")             # one bug-injected screenshot into ./test_images
# take_screenshot("bad", "x.png")    # custom file name
