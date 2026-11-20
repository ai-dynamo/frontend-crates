// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional renderer evidence, carried through the existing publication transport.

use crate::{
    codec::{self, json_bytes, sha256},
    pipeline::{self, SnapshotEncoding},
    source,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

const STORE: &str = "conformance/fixtures-renderer";

pub struct Suite {
    data: Value,
    sources: Vec<Value>,
}

impl Suite {
    pub fn load(root: &Path) -> Result<Option<Self>> {
        let Some(data) = dynamo_conformance_fixtures_v2::renderer::load_suite(root)? else {
            return Ok(None);
        };
        let mut paths = BTreeSet::from([format!("{STORE}/manifest.json")]);
        for row in data["models"]
            .as_array()
            .context("missing renderer models")?
        {
            let model = &row["model"];
            let id = model["id"].as_str().context("missing renderer model ID")?;
            source::relative(id)?;
            ensure!(!id.contains('/'), "invalid renderer model ID");
            for name in ["inputs.json", "reference.json", "dynamo.json"] {
                paths.insert(format!("{STORE}/families/{id}/{name}"));
            }
            for asset in model["assets"]
                .as_array()
                .context("missing renderer assets")?
            {
                let path = asset["path"]
                    .as_str()
                    .context("missing renderer asset path")?;
                source::relative(path)?;
                paths.insert(format!("{STORE}/{path}"));
            }
        }
        let sources = paths
            .into_iter()
            .map(|path| {
                let bytes = source::read(root, &path)?;
                Ok(json!({"path":path,"sha256":sha256(&bytes),"size":bytes.len(),"kind":"renderer-source"}))
            })
            .collect::<Result<Vec<_>>>()?;
        let suite = Self { data, sources };
        ensure!(
            dynamo_conformance_fixtures_v2::renderer::load_suite(root)?.as_ref()
                == Some(&suite.data),
            "renderer evidence changed during admission"
        );
        suite.verify_unchanged(root)?;
        Ok(Some(suite))
    }

    pub fn verify_unchanged(&self, root: &Path) -> Result<()> {
        for resource in &self.sources {
            source::verified(root, resource)
                .with_context(|| format!("renderer source changed: {}", resource["path"]))?;
        }
        Ok(())
    }

    pub fn append(
        &self,
        stage: &Path,
        catalog: &mut Value,
        encoding: SnapshotEncoding,
    ) -> Result<()> {
        let models = self.data["models"]
            .as_array()
            .context("missing renderer models")?;
        let versions = models
            .iter()
            .map(|row| {
                row["dynamo"]["producer"]["version"]
                    .as_str()
                    .context("missing renderer producer version")
            })
            .collect::<Result<BTreeSet<_>>>()?;
        ensure!(
            versions.len() == 1,
            "renderer captures must share one Dynamo version"
        );
        let version = versions.into_iter().next().unwrap();
        let envelope = json!({"schema":1,"kind":"renderer-suite","renderer":self.data});
        let canonical = json_bytes(&envelope)?;
        let path = format!(
            "conformance/viewer-data/renderer/{}.json.gz",
            sha256(&canonical)
        );
        let (bytes, destination, compression) = match encoding {
            SnapshotEncoding::CheckoutGzip => (codec::gzip(&canonical)?, path.clone(), "gzip"),
            SnapshotEncoding::CanonicalSpool => {
                (canonical, format!(".native-envelopes/{path}"), "none")
            }
            SnapshotEncoding::ValidateOnly => {
                anyhow::bail!("validation cannot encode renderer evidence")
            }
        };
        pipeline::write(stage, &destination, &bytes)?;
        catalog["sources"]["renderer"] = json!([{
            "kind":"renderer-suite","format":"typed-json-v1","path":path,
            "sha256":sha256(&bytes),"size":bytes.len(),"compression":compression,
            "source_fingerprint":sha256(&json_bytes(&json!(self.sources))?)
        }]);
        catalog["sources"]["original_renderer"] = json!(self.sources);
        catalog["metadata"]["renderer"] = json!({
            "schema":1,"dynamo_version":version,
            "models":models.iter().map(|row| json!({"id":row["model"]["id"],"label":row["model"]["label"]})).collect::<Vec<_>>()
        });
        let tabs = catalog["metadata"]["tabs"]
            .as_array_mut()
            .context("missing tabs")?;
        ensure!(
            !tabs.iter().any(|tab| tab["id"] == "tab-renderer"),
            "duplicate renderer tab"
        );
        tabs.push(json!({"id":"tab-renderer","suite":"renderer","label":"Renderer"}));
        catalog.as_object_mut().unwrap().remove("revision");
        catalog["revision"] = json!(sha256(&json_bytes(catalog)?));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resources::MemoryBudget, transport, web_transport};
    use std::fs;

    fn sample(root: &Path) -> Result<Suite> {
        let mut assets = Vec::new();
        for (role, name, bytes) in [
            (
                "config",
                "config.json",
                b"{\"model_type\":\"llama\"}\n".as_slice(),
            ),
            (
                "tokenizer_config",
                "tokenizer_config.json",
                b"{\"eos_token\":\"<eos>\"}\n".as_slice(),
            ),
            (
                "chat_template",
                "chat_template.jinja",
                b"{{ messages[0].content }}".as_slice(),
            ),
        ] {
            let path = format!("assets/sample/{name}");
            pipeline::write(root, &format!("{STORE}/{path}"), bytes)?;
            assets.push(json!({"role":role,"path":path,"sha256":sha256(bytes),
                "url":format!("https://huggingface.co/example/model/resolve/{}/{name}", "a".repeat(40))}));
        }
        let manifest = json!({"schema":1,"models":[{
            "id":"sample","label":"Sample","repository":"example/model","revision":"a".repeat(40),
            "formatter":"hf","assets":assets,"reference":{"kind":"hf","transformers":"5.12.1","jinja2":"3.1.6"}
        }]});
        let request = json!({"model":"sample","messages":[{"role":"user","content":"Hello 🦀\n"}]});
        let inputs = json!({"schema":1,"model":"sample","cases":[{
            "id":"RENDERER.hello","group":"conversations","description":"Single user turn",
            "request":request,"settings":{"add_generation_prompt":true,"request_mode":"typed", "content_format":"auto","exclude_tools_when_tool_choice_none":false},
            "metadata_expectation":{"pending_segments":0,"media_message_order":null},"metadata_origin":"authored-contract"
        }]});
        let manifest_bytes = json_bytes(&manifest)?;
        let inputs_bytes = json_bytes(&inputs)?;
        pipeline::write(root, &format!("{STORE}/manifest.json"), &manifest_bytes)?;
        pipeline::write(
            root,
            &format!("{STORE}/families/sample/inputs.json"),
            &inputs_bytes,
        )?;
        for implementation in ["reference", "dynamo"] {
            let mut observation = json!({"result":{"kind":"prompt","text":"Hello 🦀\n","segments":null,"pending_segments":0,"media_message_order":null}});
            if implementation == "reference" {
                observation["reference_input"] = request.clone();
                observation["transformations"] = json!([]);
                observation["metadata_origin"] = json!("authored-contract");
            }
            let capture = json!({"schema":1,"model":"sample","inputs_sha256":sha256(&inputs_bytes),"manifest_sha256":sha256(&manifest_bytes),
                "producer":{"implementation":implementation,"version":"10.3.2","source_sha256":"a".repeat(64)},
                "results":{"RENDERER.hello":observation}});
            pipeline::write(
                root,
                &format!("{STORE}/families/sample/{implementation}.json"),
                &json_bytes(&capture)?,
            )?;
        }
        Suite::load(root)?.context("sample renderer store was not loaded")
    }

    fn catalog() -> Value {
        json!({"schema":1,"revision":"source","metadata":{"tabs":[],"selection_policy":{"unified":{}},"unified_aliases":{}},
            "sources":{"unified":[],"archives":[],"candidates":[]}})
    }

    #[test]
    fn renderer_projection_transport_and_originals_share_verified_packs() -> Result<()> {
        let root = tempfile::tempdir()?;
        let stage = tempfile::tempdir()?;
        let suite = sample(root.path())?;
        let original = catalog();
        let mut resolved = original.clone();
        suite.append(
            stage.path(),
            &mut resolved,
            SnapshotEncoding::CanonicalSpool,
        )?;
        let budget = MemoryBudget::new(512 * 1024 * 1024);
        let runtime =
            web_transport::assemble(root.path(), stage.path(), &original, &resolved, &budget)?;
        assert_eq!(runtime["metadata"]["renderer"]["dynamo_version"], "10.3.2");
        assert_eq!(runtime["metadata"]["tabs"][0]["id"], "tab-renderer");
        assert!(runtime["sources"]["unified"].as_array().unwrap().is_empty());
        let resource = &runtime["sources"]["renderer"][0];
        assert_eq!(resource["transport"], "chunked-v2");
        let index = &resource["recipe_index"];
        let bytes = source::verified(stage.path(), index)?;
        let decoded = codec::gunzip(
            &bytes,
            index["sha256"].as_str().unwrap(),
            index["decoded_size"].as_u64().unwrap() as usize,
        )?;
        let index: Value = serde_json::from_slice(&decoded)?;
        let recipe = &index["resources"][resource["path"].as_str().unwrap()];
        let envelope: Value =
            serde_json::from_slice(&transport::reconstruct(stage.path(), recipe)?)?;
        assert_eq!(envelope["renderer"], suite.data);
        let embedded = web_transport::embedded(stage.path(), &runtime, &budget)?;
        let paths = embedded.value["resources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["descriptor"]["path"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert!(paths.contains(resource["recipe_index"]["path"].as_str().unwrap()));
        assert!(
            recipe["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|chunk| paths.contains(chunk["pack"]["path"].as_str().unwrap()))
        );
        suite.verify_unchanged(root.path())?;
        fs::write(
            root.path().join(suite.sources[0]["path"].as_str().unwrap()),
            b"changed",
        )?;
        assert!(suite.verify_unchanged(root.path()).is_err());
        Ok(())
    }

    #[test]
    fn checkout_renderer_bytes_use_the_same_envelope() -> Result<()> {
        let root = tempfile::tempdir()?;
        let stage = tempfile::tempdir()?;
        let suite = sample(root.path())?;
        let mut catalog = catalog();
        suite.append(stage.path(), &mut catalog, SnapshotEncoding::CheckoutGzip)?;
        let resource = &catalog["sources"]["renderer"][0];
        let bytes = source::verified(stage.path(), resource)?;
        let decoded = codec::gunzip(&bytes, resource["sha256"].as_str().unwrap(), 1024 * 1024)?;
        let envelope: Value = serde_json::from_slice(&decoded)?;
        assert_eq!(envelope["renderer"], suite.data);
        Ok(())
    }
}
