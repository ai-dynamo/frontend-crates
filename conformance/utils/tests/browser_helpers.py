# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from selenium.webdriver.support.ui import WebDriverWait

def settle(driver):
    WebDriverWait(driver, 90).until(lambda page: page.execute_script("return document.querySelector('.tab-panel.active')?.dataset.loadState === 'ready' && window.__conformanceLoader.pending === 0"))
    assert not driver.find_elements("css selector", ".tab-panel.active .load-status[role=alert], .error")

def click(element):
    element.click()
    settle(element.parent)

def navigate(driver, url):
    driver.get(url)
    settle(driver)

def script(driver, *args):
    result = driver.execute_script(*args)
    settle(driver)
    return result

def refresh(driver):
    driver.refresh()
    settle(driver)
