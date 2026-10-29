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

def settle(driver, tab=None, *, history_ready=False, idle=True):
    def ready(page):
        # Return state, not thousands of remote cell handles, on every poll.
        state = page.execute_script("""
            const panel = arguments[0] ? document.getElementById(arguments[0])
              : document.querySelector('.tab-panel.active');
            const error = document.querySelector('.error')
              || panel?.querySelector('.load-status[role=alert]');
            return {error: error ? error.textContent : null,
              ready: panel?.dataset.loadState === 'ready'
                && Boolean(panel.querySelector('td.cell'))
                && (!arguments[2] || window.__conformanceLoader?.pending === 0)
                && (!arguments[1] || document.getElementById('viewer')?.dataset.historyAvailable !== 'checking')};
        """, tab, history_ready, idle)
        assert state['error'] is None, state['error']
        return state['ready']

    WebDriverWait(driver, 90).until(ready)

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
