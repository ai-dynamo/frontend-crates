# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run publication phases while GitHub owns artifact transport and privileges."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from ci_publication_cache import CONTEXT, FILES, valid
from verify_publication_transport import publication_metadata


# Increment when publisher invocation or output policy changes. Cache admission
# remains owned by ci_publication_cache; GitHub's cache namespace stays stable.
PUBLICATION_POLICY = "native-history-context-v4"
# Preserve the existing workflow's candidate-archive ceiling before upload.
MAX_SITE_ARCHIVE_BYTES = 1_000_000_000
SITE_ARCHIVE, CATALOG_ARCHIVE, PUBLISHER_BINARY = FILES


class Publication:
    def __init__(self, root, temporary):
        self.root = Path(root).resolve(strict=True)
        self.temporary = Path(temporary).resolve()
        self.site = Path(os.environ.get("CONFORMANCE_PUBLISHED_SITE", self.temporary / "pages"))
        self.publisher = Path(os.environ.get("CONFORMANCE_PUBLISHER", self.root / "target/release" / PUBLISHER_BINARY))
        self.context = Path(os.environ.get("CONFORMANCE_PUBLICATION_CONTEXT", self.root / CONTEXT))
        self.conversion_cache = Path(os.environ.get("CONFORMANCE_CONVERSION_CACHE", self.temporary / "conversion-cache"))
        self.payload = self.temporary / "native-publication"
        self.inventory = self.temporary / "source-generation-inventory.json"

    def run(self, command, **kwargs):
        return subprocess.run([str(part) for part in command], cwd=self.root, check=True, **kwargs)

    def python(self, script, *args, **kwargs):
        return self.run([sys.executable, self.root / "conformance/utils/src" / script, *args], **kwargs)

    def history_args(self):
        # CI publication is the complete history artifact on every branch.
        if os.environ.get("CONFORMANCE_INCLUDE_HISTORY") != "true":
            raise ValueError("CI publication requires history mode")
        return ["--include-history"]

    def configure(self):
        values = {
            "CONFORMANCE_PUBLISHED_SITE": self.site,
            "CONFORMANCE_PUBLISHER": self.publisher,
            "CONFORMANCE_PUBLICATION_CONTEXT": self.context,
            "CONFORMANCE_CONVERSION_CACHE": self.conversion_cache,
        }
        with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in values.items())

    def identify(self):
        result = self.python("ci_publication_cache.py", "identify", "--context", self.context,
                             "--publication-policy", PUBLICATION_POLICY, *self.history_args(),
                             capture_output=True, text=True)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"key={result.stdout.strip()}\n")

    def restore(self, identity):
        usable = valid(self.payload, identity)
        if usable:
            self.site.mkdir(parents=True, exist_ok=True)
            self.publisher.parent.mkdir(parents=True, exist_ok=True)
            self.run(["tar", "-xf", self.payload / SITE_ARCHIVE, "-C", self.site])
            self.run(["tar", "-xf", self.payload / CATALOG_ARCHIVE, "-C", self.root])
            shutil.copyfile(self.payload / PUBLISHER_BINARY, self.publisher)
            self.publisher.chmod(self.publisher.stat().st_mode | stat.S_IXUSR)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(f"usable={str(usable).lower()}\n")
        return usable

    def build(self):
        self.run(["cargo", "build", "--locked", "--release", "-p", "conformance-publisher"])

    def snapshot(self):
        self.python("verify_publication_transport.py", "snapshot", "--inventory", self.inventory, *self.history_args())

    def catalog(self):
        self.history_args()
        self.python("viewer_catalog.py")
        shutil.copyfile(self.root / "conformance/viewer-catalog.json", self.root / "conformance/viewer-latest-catalog.json")
        self.python("viewer_catalog.py", "--include-history")

    def publish(self):
        self.run([self.publisher, "--root", self.root, "--output", self.site,
                  "--revision", os.environ["GITHUB_SHA"], "--cache-dir", self.conversion_cache, *self.history_args()])

    def verify(self):
        self.python("verify_publication_transport.py", "check", "--inventory", self.inventory, *self.history_args())

    def stamp(self):
        self.python("ci_publication_cache.py", "stamp", "--directory", self.site, "--revision", os.environ["GITHUB_SHA"])

    def package(self):
        self.payload.mkdir(parents=True, exist_ok=True)
        self.run(["tar", "-C", self.site, "-cf", self.payload / SITE_ARCHIVE, "."])
        if (self.payload / SITE_ARCHIVE).stat().st_size >= MAX_SITE_ARCHIVE_BYTES:
            raise ValueError("candidate publication exceeds archive size limit")
        if self.publisher.resolve() != (self.payload / PUBLISHER_BINARY).resolve():
            shutil.copyfile(self.publisher, self.payload / PUBLISHER_BINARY)
        self.run(["tar", "-cf", self.payload / CATALOG_ARCHIVE, "conformance/viewer-catalog.json",
                  "conformance/viewer-unified-catalog.json", "conformance/viewer-latest-catalog.json", CONTEXT])

    def seal(self, identity):
        self.python("ci_publication_cache.py", "seal", "--directory", self.payload, "--identity", identity)

    def retained(self):
        retained = self.temporary / "retained-publication"
        status = json.loads((retained / "publication-status.json").read_bytes())
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if status["commit"] != os.environ["GITHUB_SHA"] or status["run_url"] != run_url:
            raise ValueError("retained publication does not match this run")
        return retained

    def selections(self):
        retained = self.temporary / "retained-publication"
        evidence = self.retained() / "ci-evidence" if retained.exists() else self.temporary / "no-retained-evidence"
        fresh = self.temporary / "fresh-selections"
        sources = {path.name: path for path in evidence.glob("conformance-selection-*")}
        sources.update({path.name: path for path in fresh.glob("conformance-selection-*")})
        baseline = self.temporary / "fresh-baseline"
        if not list(baseline.glob("*.json")):
            baseline = evidence / "conformance-test-baseline"
        if not list(baseline.glob("*.json")):
            raise ValueError("complete collection baseline is unavailable in transfers and retained evidence")
        for directory, owners in (("baseline-selection", [baseline]), ("assignments", sources.values())):
            destination = self.temporary / directory
            destination.mkdir(exist_ok=True)
            # Receipt names are UUIDs, so filename overlay cannot replace a lane.
            # Select one complete artifact per owner before flattening for the gate.
            for path in destination.glob("*.json"):
                path.unlink()
            for owner in owners:
                for path in owner.glob("*.json"):
                    shutil.copyfile(path, destination / path.name)

    def recover(self):
        retained = self.retained()
        evidence = retained / "ci-evidence"
        sources = evidence / "publication-inputs"
        shutil.copytree(sources / "conformance", self.root / "conformance", dirs_exist_ok=True)
        self.python("ci_publication_cache.py", "context-check", "--context", self.context)
        for path in retained.iterdir():
            if path.name in {"ci-evidence", "publication-status.json"}:
                continue
            destination = self.site / path.name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        if publication_metadata(self.site)["revision"] != os.environ["GITHUB_SHA"]:
            raise ValueError("retained site does not match this revision")
        self.build()
        self.package()
        self.seal(json.loads((sources / "cache-manifest.json").read_bytes())["identity"])

    def pages_needed(self):
        result = self.run(["gh", "api", "--paginate", "--slurp",
                           f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100"],
                          capture_output=True, text=True)
        present = any(artifact["name"] == "github-pages" and not artifact["expired"]
                      for page in json.loads(result.stdout) for artifact in page["artifacts"])
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"needed={str(not present).lower()}\n")

    def delivery(self, url):
        # This exact regression owns both local HTTP and deployed-site delivery.
        self.run([sys.executable, "-m", "pytest", "conformance/utils/tests/test_native_publication_browser.py",
                  "-k", "test_publication_delivery_matches_qualified_artifact", "--publication-url", url,
                  *self.history_args(), "-vv"])

    def current(self):
        candidate = os.environ["GITHUB_SHA"]
        if publication_metadata(self.site)["revision"] != candidate:
            raise ValueError("qualified publication does not match deployment candidate")
        result = self.run(["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
                          capture_output=True, text=True)
        fields = result.stdout.split()
        if len(fields) != 2 or fields[1] != "refs/heads/main" or len(fields[0]) != 40:
            raise ValueError("cannot establish current main revision")
        current = fields[0] == candidate
        report = {"candidate": candidate, "main": fields[0], "outcome": "current" if current else "superseded"}
        (self.temporary / "conformance-deployment.json").write_text(json.dumps(report) + "\n")
        print(json.dumps(report))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"current={str(current).lower()}\n")
        return current


PHASES = ("configure", "identify", "restore", "build", "snapshot", "catalog", "publish", "verify", "stamp", "package", "seal", "delivery", "current", "recover", "selections", "pages-needed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--identity")
    parser.add_argument("--url")
    args = parser.parse_args()
    if (args.phase in {"restore", "seal"}) != bool(args.identity):
        parser.error("restore/seal require --identity; other phases do not accept it")
    if (args.phase == "delivery") != bool(args.url):
        parser.error("delivery requires --url; other phases do not accept it")
    publication = Publication(os.environ.get("GITHUB_WORKSPACE", Path.cwd()), os.environ["RUNNER_TEMP"])
    # Explicit dispatch keeps supported phases visible without dynamic attribute lookup.
    phases = {"configure": publication.configure, "identify": publication.identify, "restore": lambda: publication.restore(args.identity),
              "build": publication.build, "snapshot": publication.snapshot, "catalog": publication.catalog,
              "publish": publication.publish, "verify": publication.verify, "stamp": publication.stamp,
              "package": publication.package, "seal": lambda: publication.seal(args.identity),
              "delivery": lambda: publication.delivery(args.url), "current": publication.current,
              "recover": publication.recover, "selections": publication.selections, "pages-needed": publication.pages_needed}
    phases[args.phase]()


if __name__ == "__main__":
    main()
