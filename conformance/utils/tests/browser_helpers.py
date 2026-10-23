# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from selenium.webdriver.support.ui import WebDriverWait

def raw_view_download(driver):
    result = driver.execute_async_script("""
        const done = arguments[arguments.length - 1];
        (async () => {
            const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
                .map(value => value.toString(16).padStart(2, '0')).join('');
            const links = [...document.querySelectorAll('#viewer > a')];
            const download = links.find(anchor => anchor.download);
            const bytes = await (await fetch(download.href)).arrayBuffer();
            done({hash: await sha256(bytes), size: bytes.byteLength,
                renderedHash: await sha256(new TextEncoder().encode(document.querySelector('#viewer > pre').textContent)),
                filename: download.download, back: links.find(anchor => !anchor.download).href});
        })().catch(error => done({error: String(error)}));
    """)
    assert 'error' not in result, result
    return result

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
