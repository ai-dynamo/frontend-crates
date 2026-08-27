// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod deepseek_dsml;
mod glm47;
mod qwen3_coder;
mod templated_tool_calls;

pub(crate) use deepseek_dsml::DEEPSEEK_DSML;
pub(crate) use glm47::GLM47;
pub(crate) use qwen3_coder::QWEN3_CODER;

use serde_json::Value;

use super::policy::{
    ResolvedToolCalling, ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_calling,
};
use super::wire::{
    AnyTextFormat, ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OrFormat,
    SequenceFormat, StructuralTag, TagFormat, TriggeredTagsFormat,
};
use super::{ReasoningBoundary, StructuralTagContext, StructuralTagOptions};

/// Builds model-family structural tags while enforcing shared request policy.
pub struct StructuralTagBuilder {
    grammar: &'static dyn ToolCallGrammar,
}

impl StructuralTagBuilder {
    pub(super) const fn new(grammar: &'static dyn ToolCallGrammar) -> Self {
        Self { grammar }
    }

    /// Returns `None` when this request needs no structural constraint.
    pub fn build(&self, context: &StructuralTagContext<'_>) -> anyhow::Result<Option<Value>> {
        self.build_with_options(context, &StructuralTagOptions::default())
    }

    /// Builds a structural tag with request-independent behavior overrides.
    pub fn build_with_options(
        &self,
        context: &StructuralTagContext<'_>,
        options: &StructuralTagOptions,
    ) -> anyhow::Result<Option<Value>> {
        let policy = match resolve_tool_calling(context)? {
            ResolvedToolCalling::Disabled => {
                return build_text_exclusion(self.grammar.tool_call_excludes())
                    .map(|format| serialize(StructuralTag { format }))
                    .transpose();
            }
            ResolvedToolCalling::Enabled(policy) => policy,
        };
        if policy.tools.is_empty() {
            return Ok(None);
        }
        let exclude_special_tokens = options
            .exclude_special_tokens
            .unwrap_or_else(|| self.grammar.default_exclude_special_tokens());

        let format = match (policy.mode, context.structured_output_schema) {
            (ToolCallingMode::Auto, Some(schema)) => {
                self.grammar.build_auto_with_structured_output(
                    &policy,
                    schema,
                    options.tool_arguments_any_order,
                )?
            }
            (ToolCallingMode::Auto, None) => self.grammar.build_triggered_calls(
                &policy,
                exclude_special_tokens,
                options.tool_arguments_any_order,
            )?,
            (ToolCallingMode::Required, _) => self.grammar.build_triggered_calls(
                &policy,
                exclude_special_tokens,
                options.tool_arguments_any_order,
            )?,
            (ToolCallingMode::Named, _) => self
                .grammar
                .build_tool_calls_only(&policy, options.tool_arguments_any_order)?,
        };
        let format = if options.reasoning_boundary == ReasoningBoundary::StructuralTag {
            wrap_reasoning_if_needed(
                context.starts_in_reasoning,
                self.grammar,
                exclude_special_tokens,
                format,
            )?
        } else {
            format
        };
        serialize(StructuralTag { format }).map(Some)
    }
}

pub(super) trait ToolCallGrammar: Send + Sync {
    fn reasoning_begin(&self) -> Option<&'static str>;

    fn reasoning_end(&self) -> Option<&'static str>;

    fn reasoning_suffix(&self) -> &'static str {
        ""
    }

    fn tool_call_excludes(&self) -> &'static [&'static str];

    fn default_exclude_special_tokens(&self) -> bool {
        true
    }

    /// Builds trigger-dispatched tool calls for `auto` or `required` tool choice.
    /// Required policy does not allow free text before the first tool call.
    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format>;

    /// Builds the `auto` choice between native tool calls and a constrained
    /// final response. Override this when the model wraps final responses in a
    /// model-specific envelope.
    fn build_auto_with_structured_output(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        schema: &Value,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Or(OrFormat {
            elements: vec![
                self.build_tool_calls_only(policy, tool_arguments_any_order)?,
                Format::JsonSchema(JsonSchemaFormat {
                    json_schema: schema.clone(),
                    style: JsonSchemaStyle::Json,
                    any_order: false,
                }),
            ],
        }))
    }

    /// Builds a format whose first generated bytes belong to a tool call and
    /// whose accepted output ends after the tool-call sequence.
    ///
    /// `policy.mode` can be `Auto` when tool calls are one branch of a composite
    /// output. In that branch the lower bound becomes one call, while the
    /// cardinality still controls whether generation stops after the first.
    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format>;
}

fn build_text_exclusion(excludes: &[&str]) -> Option<Format> {
    if excludes.is_empty() {
        return None;
    }

    Some(Format::Tag(TagFormat {
        begin: String::new(),
        content: Box::new(Format::AnyText(AnyTextFormat {
            excludes: excludes
                .iter()
                .map(|exclude| (*exclude).to_string())
                .collect(),
        })),
        end: String::new(),
    }))
}

pub(super) fn triggered_calls_format(
    trigger: &str,
    tags: Vec<TagFormat>,
    reasoning_begin: Option<&str>,
    reasoning_end: Option<&str>,
    free_text_excludes: &[&str],
    exclude_special_tokens: bool,
    policy: &ResolvedToolCallingPolicy<'_>,
) -> Format {
    Format::TriggeredTags(TriggeredTagsFormat {
        triggers: vec![trigger.to_string()],
        tags,
        excludes: if exclude_special_tokens {
            [reasoning_begin, reasoning_end]
                .into_iter()
                .flatten()
                .chain(free_text_excludes.iter().copied())
                .map(str::to_string)
                .collect()
        } else {
            Vec::new()
        },
        at_least_one: policy.mode == ToolCallingMode::Required,
        stop_after_first: policy.stop_after_first(),
    })
}

fn reasoning_content(grammar: &dyn ToolCallGrammar, exclude_special_tokens: bool) -> Format {
    let excludes = if exclude_special_tokens {
        grammar
            .tool_call_excludes()
            .iter()
            .map(|pattern| (*pattern).to_string())
            .collect()
    } else {
        Vec::new()
    };

    Format::AnyText(AnyTextFormat { excludes })
}

fn reasoning_tag(
    grammar: &dyn ToolCallGrammar,
    reasoning_end: &str,
    exclude_special_tokens: bool,
) -> Format {
    Format::Tag(TagFormat {
        begin: String::new(),
        content: Box::new(reasoning_content(grammar, exclude_special_tokens)),
        end: reasoning_end.to_string(),
    })
}

fn wrap_reasoning_if_needed(
    starts_in_reasoning: bool,
    grammar: &dyn ToolCallGrammar,
    exclude_special_tokens: bool,
    suffix: Format,
) -> anyhow::Result<Format> {
    if !starts_in_reasoning {
        return Ok(suffix);
    }

    let reasoning_end = grammar
        .reasoning_end()
        .ok_or_else(|| anyhow::anyhow!("reasoning end tag is not configured for structural tag"))?;

    let mut elements = vec![reasoning_tag(
        grammar,
        reasoning_end,
        exclude_special_tokens,
    )];
    if !grammar.reasoning_suffix().is_empty() {
        elements.push(Format::ConstString(ConstStringFormat {
            value: grammar.reasoning_suffix().to_string(),
        }));
    }
    elements.push(suffix);

    Ok(Format::Sequence(SequenceFormat { elements }))
}

fn serialize(tag: StructuralTag) -> anyhow::Result<Value> {
    serde_json::to_value(tag).map_err(Into::into)
}
