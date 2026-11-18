// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// The stock MiniMax M2 generation prompt. The template always pre-fills the
/// assistant turn with an opened `<think>` block and never consults a thinking
/// toggle, so `thinking=false` renders byte-identical to `thinking=true`.
///
/// Matches `MiniMaxAI/MiniMax-M2.7` `chat_template.jinja` at Hugging Face
/// revision `d494266a4affc0d2995ba1fa35c8481cbd84294b`.
const MINIMAX_M2_GENERATION_BLOCK: &str = r"{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}
{%- endif -%}";

/// Detects the stock MiniMax M2 family chat template (M2, M2.1, M2.5, M2.7): the
/// `<minimax:tool_call>` envelope plus the complete, unchanged generation block
/// at the end of the template. Custom thinking logic inside or after that block
/// does not match. MiniMax M3 uses `<mm:think>` and does not match either.
pub(super) fn is_minimax_m2_template_source(source: &str) -> bool {
    source.contains("<minimax:tool_call>")
        && source.trim_end().ends_with(MINIMAX_M2_GENERATION_BLOCK)
}

/// Make the MiniMax M2 generation prompt honor the standard thinking toggles.
///
/// The upstream template opens `<think>` unconditionally. When a request
/// disables thinking (`thinking=false`, `enable_thinking=false`, or the
/// DeepSeek-style `thinking_mode="chat"` that consumers map onto this family),
/// the opener is closed immediately so the completion starts after an empty
/// reasoning block instead of inside one. Requests without a toggle, or with
/// thinking enabled, render exactly as upstream.
pub(super) fn adapt_minimax_m2_thinking_template_source(source: &str) -> String {
    let trimmed = source.trim_end();
    let Some(prefix) = trimmed.strip_suffix(MINIMAX_M2_GENERATION_BLOCK) else {
        return source.to_string();
    };

    const NEW_GENERATION_BLOCK: &str = r"{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}
{%- set dyn_minimax_thinking_disabled = false -%}
{%- if thinking is boolean -%}
    {%- set dyn_minimax_thinking_disabled = not thinking -%}
{%- elif enable_thinking is boolean -%}
    {%- set dyn_minimax_thinking_disabled = not enable_thinking -%}
{%- elif thinking_mode is string and thinking_mode == 'chat' -%}
    {%- set dyn_minimax_thinking_disabled = true -%}
{%- endif -%}
{%- if dyn_minimax_thinking_disabled -%}
{{- '</think>' ~ '\n' }}
{%- endif -%}
{%- endif -%}";

    let trailing_whitespace = &source[trimmed.len()..];
    format!("{prefix}{NEW_GENERATION_BLOCK}{trailing_whitespace}")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use minijinja::{Environment, Value, context};
    use serde_json::json;

    use super::*;
    use crate::template::formatters::normalize_chat_template_source;
    use crate::template::{
        ChatTemplate, ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment,
    };
    use crate::{OAIChatLikeRequest, OAIPromptFormatter};

    /// Minimal M2-style template: message replay plus the stock generation block.
    /// The block is spelled out independently of the production matcher so changes
    /// to the matcher cannot silently change the test input too.
    fn minimax_m2_style_template() -> String {
        r"<minimax:tool_call>
{%- for message in messages -%}
{{- message.role ~ '\n' ~ message.content ~ '\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}
{%- endif -%}"
            .to_string()
    }

    fn env_with_default(source: &str) -> Environment<'static> {
        let mut env = JinjaEnvironment::default().env;
        env.add_template_owned("default", source.to_string())
            .unwrap();
        env
    }

    fn formatter_for(template: &str) -> HfTokenizerConfigJsonFormatter {
        let config: ChatTemplate = serde_json::from_value(json!({
            "chat_template": template,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }))
        .unwrap();
        HfTokenizerConfigJsonFormatter::new(config, ContextMixins::new(&[])).unwrap()
    }

    #[test]
    fn minimax_m2_template_is_detected_and_adapted_by_exact_match() {
        let template = minimax_m2_style_template();
        let template = template.as_str();
        assert!(is_minimax_m2_template_source(template));
        let adapted = normalize_chat_template_source(template);
        assert!(!is_minimax_m2_template_source(&adapted));
        let (prefix, _) = template.split_once(MINIMAX_M2_GENERATION_BLOCK).unwrap();
        assert!(adapted.starts_with(prefix));
        assert_eq!(normalize_chat_template_source(&adapted), adapted);
    }

    #[test]
    fn minimax_m2_adapter_ignores_other_templates() {
        let m3 = "{%- if add_generation_prompt -%}{{ ']~b]ai\n<mm:think>\n' }}{%- endif -%}]<]minimax[>[";
        assert!(!is_minimax_m2_template_source(m3));
        assert_eq!(normalize_chat_template_source(m3), m3);
        let custom = "<minimax:tool_call>{%- if add_generation_prompt -%}{{ ']~b]ai\n<think>\n' }}{%- endif -%}";
        assert!(!is_minimax_m2_template_source(custom));
        assert_eq!(normalize_chat_template_source(custom), custom);
        // A custom closer after the stock block must also be preserved.
        let custom = format!(
            "<minimax:tool_call>{MINIMAX_M2_GENERATION_BLOCK}\n{}",
            r"{%- if thinking is false -%}{{- '</think>' ~ '\n' }}{%- endif -%}"
        );
        assert!(!is_minimax_m2_template_source(&custom));
        assert_eq!(normalize_chat_template_source(&custom), custom);
        let qwen = "{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>\n{%- endif -%}";
        assert!(!is_minimax_m2_template_source(qwen));
        assert_eq!(normalize_chat_template_source(qwen), qwen);
        assert_eq!(adapt_minimax_m2_thinking_template_source(qwen), qwen);
    }

    #[test]
    fn minimax_m2_adapted_generation_prompt_honors_thinking_toggles() {
        let generation_prompt_only = format!("<minimax:tool_call>{MINIMAX_M2_GENERATION_BLOCK}");
        assert!(is_minimax_m2_template_source(&generation_prompt_only));
        let env = env_with_default(&normalize_chat_template_source(&generation_prompt_only));
        let tmpl = env.get_template("default").unwrap();
        let render = |ctx: minijinja::Value| tmpl.render(ctx).unwrap();

        let open = "<minimax:tool_call>]~b]ai\n<think>\n";
        let closed = "<minimax:tool_call>]~b]ai\n<think>\n</think>\n";
        assert_eq!(render(context! { add_generation_prompt => true }), open);
        assert_eq!(
            render(context! { add_generation_prompt => true, thinking => true }),
            open
        );
        assert_eq!(
            render(context! { add_generation_prompt => true, thinking => false }),
            closed
        );
        assert_eq!(
            render(context! { add_generation_prompt => true, enable_thinking => false }),
            closed
        );
        assert_eq!(
            render(context! { add_generation_prompt => true, thinking_mode => "chat" }),
            closed
        );
        assert_eq!(
            render(context! { add_generation_prompt => true, thinking_mode => "thinking" }),
            open
        );
        assert_eq!(
            render(context! { add_generation_prompt => true, thinking => "false" }),
            open
        );
        assert_eq!(
            render(context! { add_generation_prompt => false, thinking => false }),
            "<minimax:tool_call>"
        );
    }

    fn minimax_m2_formatter() -> HfTokenizerConfigJsonFormatter {
        formatter_for(&minimax_m2_style_template())
    }

    /// One user message plus the `chat_template_kwargs` and `add_generation_prompt`
    /// that Dynamo's request wrapper supplies.
    struct TemplateRequest {
        args: Option<HashMap<String, serde_json::Value>>,
        add_generation_prompt: bool,
    }

    impl OAIChatLikeRequest for TemplateRequest {
        fn model(&self) -> String {
            "MiniMaxAI/MiniMax-M2.7".to_string()
        }
        fn messages(&self) -> Value {
            Value::from_serialize(json!([{"role": "user", "content": "What is 937 * 18 + 42?"}]))
        }
        fn should_add_generation_prompt(&self) -> bool {
            self.add_generation_prompt
        }
        fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
            self.args.as_ref()
        }
    }

    fn minimax_request(chat_template_kwargs: Option<serde_json::Value>) -> TemplateRequest {
        TemplateRequest {
            args: chat_template_kwargs.map(|kwargs| serde_json::from_value(kwargs).unwrap()),
            add_generation_prompt: true,
        }
    }

    const MINIMAX_OPEN_TAIL: &str = "]~b]ai\n<think>\n";
    const MINIMAX_CLOSED_TAIL: &str = "]~b]ai\n<think>\n</think>\n";

    #[test]
    fn minimax_m2_default_and_enabled_thinking_render_open_think() {
        let f = minimax_m2_formatter();
        for kwargs in [
            None,
            Some(json!({})),
            Some(json!({"thinking": true})),
            Some(json!({"enable_thinking": true})),
            Some(json!({"thinking_mode": "thinking"})),
            Some(json!({"thinking_mode": "adaptive"})),
            // Non-bool spellings are not a toggle for the template; consumers
            // normalize them to bools before rendering.
            Some(json!({"thinking": "false"})),
        ] {
            let rendered = f.render(&minimax_request(kwargs.clone())).unwrap();
            assert!(
                rendered.ends_with(MINIMAX_OPEN_TAIL),
                "{kwargs:?}: expected open <think> tail, got {rendered:?}"
            );
            assert!(!rendered.contains("</think>"), "{kwargs:?}");
        }
    }

    #[test]
    fn minimax_m2_disabled_thinking_closes_the_empty_think_block() {
        let f = minimax_m2_formatter();
        for kwargs in [
            json!({"thinking": false}),
            json!({"enable_thinking": false}),
            // `thinking` wins over `enable_thinking`, matching thinking_bool_from_args.
            json!({"thinking": false, "enable_thinking": true}),
            json!({"thinking_mode": "chat"}),
        ] {
            let rendered = f.render(&minimax_request(Some(kwargs.clone()))).unwrap();
            assert!(
                rendered.ends_with(MINIMAX_CLOSED_TAIL),
                "{kwargs}: expected closed empty think block, got {rendered:?}"
            );
            assert_eq!(
                rendered.matches("<think>").count(),
                1,
                "{kwargs}: exactly one opener"
            );
        }
        let rendered = f
            .render(&minimax_request(Some(
                json!({"thinking": true, "enable_thinking": false}),
            )))
            .unwrap();
        assert!(rendered.ends_with(MINIMAX_OPEN_TAIL));
    }

    #[test]
    fn minimax_m2_preserves_custom_thinking_closure() {
        let template = minimax_m2_style_template();
        let stock_opener = r"{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}";
        assert!(template.contains(stock_opener));

        let custom_opener = [
            stock_opener,
            r"{%- if thinking is false -%}{{- '</think>' ~ '\n' }}{%- endif -%}",
        ]
        .join("\n");
        let custom_template = template.replacen(stock_opener, &custom_opener, 1);

        let formatter = formatter_for(&custom_template);
        let rendered = formatter
            .render(&minimax_request(Some(json!({"thinking": false}))))
            .unwrap();

        assert_eq!(rendered.matches("</think>").count(), 1);
        assert!(rendered.ends_with(MINIMAX_CLOSED_TAIL));
    }

    #[test]
    fn minimax_m2_disabled_thinking_only_changes_the_generation_prompt() {
        let f = minimax_m2_formatter();
        let enabled = f
            .render(&minimax_request(Some(json!({"thinking": true}))))
            .unwrap();
        let disabled = f
            .render(&minimax_request(Some(json!({"thinking": false}))))
            .unwrap();
        let common = enabled.strip_suffix(MINIMAX_OPEN_TAIL).unwrap();
        assert_eq!(disabled, format!("{common}{MINIMAX_CLOSED_TAIL}"));
    }

    #[test]
    fn minimax_m2_no_generation_prompt_never_appends_a_closer() {
        let f = minimax_m2_formatter();
        let mut request = minimax_request(Some(json!({"thinking": false})));
        request.add_generation_prompt = false;
        let rendered = f.render(&request).unwrap();
        assert!(!rendered.contains("<think>"));
        assert!(!rendered.contains("</think>"));
    }
}
