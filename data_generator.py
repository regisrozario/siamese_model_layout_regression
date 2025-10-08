import os
import time
import random
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- Configuration ----------
# Add your 3 URLs here
URLS_TO_TEST = [
    "https://www.google.com",
    "https://policies.google.com/terms?hl=en&fg=1",
    "https://arnab-datta.github.io/counter-app/",
]

BASELINE_DIR = Path("dataset/baseline")
REGRESSED_DIR = Path("dataset/regressed")
PAGE_READY_SELECTOR = (By.TAG_NAME, "body")  # Wait for body to load

# Number of screenshots per URL (recommended: 20-50 for good training)
SAMPLES_PER_URL = 50  # Will create 20 good + 20 bad = 40 images per URL
# Total: 3 URLs × 20 samples = 60 pairs (120 images)
# ------------------------------------

# Create directories
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
REGRESSED_DIR.mkdir(parents=True, exist_ok=True)


def _new_driver(headless=False):
    """Create a new Chrome driver instance."""
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")

    if headless:
        opts.add_argument("--headless=new")

    return webdriver.Chrome(options=opts)


def _inject_random_layout_bugs(driver, severity='medium'):
    """
    Inject random layout bugs into the page.
    severity: 'light', 'medium', 'heavy' controls how many bugs to inject
    """
    bug_probability = {
        'light': 0.3,
        'medium': 0.6,
        'heavy': 0.9
    }.get(severity, 0.6)

    bugs_applied = []

    # Bug 1: Hide random elements
    if random.random() < bug_probability:
        driver.execute_script("""
            var elements = Array.from(document.querySelectorAll('img, button, a, h1, h2, div'))
                .filter(el => el.offsetParent !== null && el.offsetWidth > 50);
            var randomElement = elements[Math.floor(Math.random() * Math.min(elements.length, 10))];
            if (randomElement) {
                randomElement.style.display = "none";
            }
        """)
        bugs_applied.append("hidden_element")

    # Bug 2: Shift elements randomly
    if random.random() < bug_probability:
        driver.execute_script("""
            var elements = Array.from(document.querySelectorAll('button, input, a'))
                .filter(el => el.offsetParent !== null);
            elements.slice(0, 3).forEach(function(el) {
                if (Math.random() < 0.5) {
                    el.style.marginLeft = Math.floor(Math.random() * 200 - 100) + "px";
                    el.style.marginTop = Math.floor(Math.random() * 100 - 50) + "px";
                }
            });
        """)
        bugs_applied.append("shifted_elements")

    # Bug 3: Break sizing
    if random.random() < bug_probability:
        driver.execute_script("""
            var buttons = document.querySelectorAll('button, input[type="submit"]');
            Array.from(buttons).slice(0, 2).forEach(function(btn) {
                if (Math.random() < 0.7) {
                    btn.style.width = Math.floor(Math.random() * 50 + 30) + "px";
                    btn.style.height = Math.floor(Math.random() * 20 + 15) + "px";
                    btn.style.overflow = "hidden";
                }
            });
        """)
        bugs_applied.append("broken_sizing")

    # Bug 4: Overlap elements
    if random.random() < bug_probability:
        driver.execute_script("""
            var elements = Array.from(document.querySelectorAll('div, section, nav'))
                .filter(el => el.offsetParent !== null);
            var el = elements[Math.floor(Math.random() * Math.min(elements.length, 5))];
            if (el) {
                el.style.position = "absolute";
                el.style.zIndex = "9999";
                el.style.top = Math.floor(Math.random() * 300) + "px";
            }
        """)
        bugs_applied.append("overlapping")

    # Bug 5: Break text/content
    if random.random() < bug_probability:
        driver.execute_script("""
            var textElements = Array.from(document.querySelectorAll('p, span, a, h1, h2'))
                .filter(el => el.innerText && el.innerText.length > 5);
            textElements.slice(0, 3).forEach(function(el) {
                var choice = Math.random();
                if (choice < 0.3) {
                    el.style.color = el.style.backgroundColor || "white";  // Hide text
                } else if (choice < 0.6) {
                    el.style.fontSize = Math.floor(Math.random() * 8 + 4) + "px";
                } else {
                    el.style.overflow = "hidden";
                    el.style.whiteSpace = "nowrap";
                    el.style.width = "50px";
                }
            });
        """)
        bugs_applied.append("broken_text")

    # Bug 6: Break visibility
    if random.random() < bug_probability * 0.8:
        driver.execute_script("""
            var visibleElements = Array.from(document.querySelectorAll('nav, header, footer, aside'))
                .filter(el => el.offsetParent !== null);
            if (visibleElements.length > 0) {
                var el = visibleElements[Math.floor(Math.random() * visibleElements.length)];
                el.style.visibility = "hidden";
            }
        """)
        bugs_applied.append("hidden_section")

    # Bug 7: Break alignment
    if random.random() < bug_probability:
        driver.execute_script("""
            var containers = Array.from(document.querySelectorAll('div, section'))
                .filter(el => el.children.length > 2);
            if (containers.length > 0) {
                var container = containers[Math.floor(Math.random() * Math.min(containers.length, 3))];
                container.style.textAlign = ["left", "right", "center"][Math.floor(Math.random() * 3)];
                container.style.display = "block";
            }
        """)
        bugs_applied.append("broken_alignment")

    return bugs_applied


def capture_pair(driver, url, index, url_name):
    """Capture a baseline and regressed pair for a single page state."""
    try:
        # Navigate to URL
        print(f"\n🌐 Loading: {url}")
        driver.get(url)

        # Wait for page to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located(PAGE_READY_SELECTOR))
        time.sleep(2)  # Let page fully render

        # Scroll randomly to get different viewport states
        scroll_position = random.randint(0, 500)
        driver.execute_script(f"window.scrollTo(0, {scroll_position});")
        time.sleep(0.5)

        # Capture BASELINE (good)
        baseline_filename = f"{url_name}_{index:03d}.png"
        baseline_path = BASELINE_DIR / baseline_filename
        driver.save_screenshot(str(baseline_path))
        print(f"  ✅ Baseline: {baseline_filename}")

        # Apply bugs
        severity = random.choice(['light', 'medium', 'heavy'])
        bugs = _inject_random_layout_bugs(driver, severity=severity)
        time.sleep(0.5)  # Let changes apply

        # Capture REGRESSED (bad)
        regressed_filename = f"{url_name}_{index:03d}.png"
        regressed_path = REGRESSED_DIR / regressed_filename
        driver.save_screenshot(str(regressed_path))
        print(f"  ❌ Regressed: {regressed_filename} (bugs: {', '.join(bugs)})")

        return True

    except Exception as e:
        print(f"  ⚠️  Error capturing pair {index}: {e}")
        return False


def generate_dataset(urls=None, samples_per_url=10, headless=False):
    """
    Generate a complete training dataset.

    Args:
        urls: List of URLs to test (default: URLS_TO_TEST)
        samples_per_url: Number of screenshot pairs per URL
        headless: Run browser in headless mode
    """
    if urls is None:
        urls = URLS_TO_TEST

    print("=" * 70)
    print("🎬 DATASET GENERATION STARTED")
    print("=" * 70)
    print(f"📁 Baseline directory: {BASELINE_DIR}")
    print(f"📁 Regressed directory: {REGRESSED_DIR}")
    print(f"🌐 URLs to test: {len(urls)}")
    print(f"📸 Samples per URL: {samples_per_url}")
    print(f"📊 Expected total pairs: {len(urls) * samples_per_url}")
    print("=" * 70)

    driver = _new_driver(headless=headless)

    total_captured = 0
    total_failed = 0

    try:
        for url_idx, url in enumerate(urls, 1):
            # Create a clean URL name for filenames
            url_name = url.replace('https://', '').replace('http://', '').replace('www.', '')
            url_name = url_name.split('/')[0].replace('.', '_')

            print(f"\n{'=' * 70}")
            print(f"📍 URL {url_idx}/{len(urls)}: {url}")
            print(f"{'=' * 70}")

            url_captured = 0

            for i in range(samples_per_url):
                success = capture_pair(driver, url, i + 1, url_name)
                if success:
                    url_captured += 1
                    total_captured += 1
                else:
                    total_failed += 1

                # Small delay between captures
                time.sleep(1)

            print(f"\n✅ Captured {url_captured}/{samples_per_url} pairs for {url}")

    finally:
        driver.quit()

    print("\n" + "=" * 70)
    print("🎉 DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully captured: {total_captured} pairs")
    print(f"❌ Failed: {total_failed} pairs")
    print(f"📊 Total images: {total_captured * 2} ({total_captured} baseline + {total_captured} regressed)")
    print(f"\n💾 Saved to:")
    print(f"   Baseline: {BASELINE_DIR}")
    print(f"   Regressed: {REGRESSED_DIR}")
    print("\n🔄 Next steps:")
    print("   1. Run: python generate_pairs.py")
    print("   2. Run: python train.py")
    print("=" * 70)


def quick_test(num_samples=5):
    """Generate a small test dataset quickly."""
    print("🧪 Quick test mode - generating small dataset...")
    generate_dataset(
        urls=["https://www.google.com"],
        samples_per_url=num_samples,
        headless=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate training dataset for layout regression detection')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of screenshot pairs per URL (default: 10)')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode - generate only 5 samples from Google')
    parser.add_argument('--urls', nargs='+',
                        help='Custom URLs to test (space-separated)')

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    else:
        urls = args.urls if args.urls else URLS_TO_TEST
        generate_dataset(
            urls=urls,
            samples_per_url=args.samples,
            headless=args.headless
        )