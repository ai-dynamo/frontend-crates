// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Chat custom tools and their function-template representation.

use super::*;
use serde::de::Error as _;
use serde::ser::{Error as _, SerializeStruct};

pub const CUSTOM_TOOL_INPUT_ARG: &str = "input";

/// One shared schema for Responses and Chat custom tools in function templates.
pub fn custom_tool_parameters() -> serde_json::Value {
    serde_json::json!({"type":"object", "properties": {CUSTOM_TOOL_INPUT_ARG: {
        "type":"string", "contentMediaType":"text/plain", "description":"The complete freeform input for this tool, passed through verbatim."
    }}, "required":[CUSTOM_TOOL_INPUT_ARG]})
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::CustomToolFormat))]
pub enum CustomToolFormat {
    Text,
    Grammar { grammar: CustomToolGrammar },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::CustomToolGrammar))]
pub struct CustomToolGrammar {
    pub syntax: CustomToolGrammarSyntax,
    pub definition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::CustomToolGrammarSyntax))]
pub enum CustomToolGrammarSyntax {
    Lark,
    Regex,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionCustomTool))]
pub struct ChatCompletionCustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<CustomToolFormat>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::CustomToolCall))]
pub struct CustomToolCall {
    pub name: String,
    pub input: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::CustomToolCallStream))]
pub struct CustomToolCallStream {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ChatCompletionTool {
    /// Borrow ordinary tools; adapt custom tools only for function-oriented engines.
    pub fn function_template_tool(&self) -> std::borrow::Cow<'_, Self> {
        if self.r#type != ChatCompletionToolType::Custom {
            return std::borrow::Cow::Borrowed(self);
        }
        if let Some(custom) = &self.custom {
            return std::borrow::Cow::Owned(Self {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: custom.name.clone(),
                    description: custom.description.clone(),
                    parameters: Some(custom_tool_parameters()),
                    ..FunctionObject::default()
                },
                custom: None,
                defer_loading: self.defer_loading,
            });
        }
        let mut tool = self.clone();
        tool.r#type = ChatCompletionToolType::Function;
        std::borrow::Cow::Owned(tool)
    }
}

impl ChatCompletionMessageToolCall {
    pub fn function_template_call(&self) -> std::borrow::Cow<'_, Self> {
        if self.r#type != FunctionType::Custom {
            return std::borrow::Cow::Borrowed(self);
        }
        if let Some(custom) = &self.custom {
            return std::borrow::Cow::Owned(Self {
                id: self.id.clone(),
                r#type: FunctionType::Function,
                function: FunctionCall {
                    name: custom.name.clone(),
                    arguments: serde_json::json!({CUSTOM_TOOL_INPUT_ARG:&custom.input}).to_string(),
                },
                custom: None,
            });
        }
        let mut call = self.clone();
        call.r#type = FunctionType::Function;
        std::borrow::Cow::Owned(call)
    }
}

impl ChatCompletionToolChoiceOption {
    pub fn function_template_choice(&self) -> std::borrow::Cow<'_, Self> {
        match self {
            Self::Named(choice) if choice.r#type == ChatCompletionToolType::Custom => {
                std::borrow::Cow::Owned(Self::Named(ChatCompletionNamedToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: choice.function.clone(),
                }))
            }
            _ => std::borrow::Cow::Borrowed(self),
        }
    }
}

impl CreateChatCompletionRequest {
    pub fn has_custom_tools(&self) -> bool {
        self.tools.iter().flatten().any(|tool|tool.r#type == ChatCompletionToolType::Custom)
            || matches!(&self.tool_choice, Some(ChatCompletionToolChoiceOption::Named(choice)) if choice.r#type == ChatCompletionToolType::Custom)
            || self.messages.iter().any(|message|matches!(message, ChatCompletionRequestMessage::Assistant(message) if message.tool_calls.iter().flatten().any(|call|call.r#type == FunctionType::Custom)))
    }

    /// Normalize only the engine-facing representation, after retaining API identity.
    /// Grammar custom tools need an enforcement owner and are rejected explicitly.
    pub fn normalize_custom_tools(&mut self) -> Result<Vec<String>, OpenAIError> {
        let tools = self.tools.as_deref().unwrap_or_default();
        let mut custom_names = Vec::new();
        for tool in tools {
            if tool.r#type == ChatCompletionToolType::Custom {
                let custom = tool.custom.as_ref().ok_or_else(|| {
                    OpenAIError::InvalidArgument("custom tool requires custom definition".into())
                })?;
                if matches!(custom.format, Some(CustomToolFormat::Grammar { .. })) {
                    return Err(OpenAIError::InvalidArgument(
                        "Custom tool grammar formats are not supported".into(),
                    ));
                }
                custom_names.push(custom.name.clone());
            }
        }
        // Response projection restores custom identity by name; a function cannot share it.
        if let Some(tool) = tools.iter().find(|tool| {
            tool.r#type != ChatCompletionToolType::Custom
                && custom_names.contains(&tool.function.name)
        }) {
            return Err(OpenAIError::InvalidArgument(format!(
                "tool name '{}' is declared as both custom and function",
                tool.function.name
            )));
        }
        match &self.tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(choice))
                if choice.r#type == ChatCompletionToolType::Custom
                    && !custom_names.contains(&choice.function.name) =>
            {
                return Err(OpenAIError::InvalidArgument(format!(
                    "custom tool '{}' in tool_choice is not present in tools",
                    choice.function.name
                )));
            }
            Some(ChatCompletionToolChoiceOption::Named(choice))
                if choice.r#type == ChatCompletionToolType::Function
                    && custom_names.contains(&choice.function.name) =>
            {
                return Err(OpenAIError::InvalidArgument("tool_choice type must match the selected tool; custom choices use the custom field".into()));
            }
            _ => {}
        }
        for tool in self.tools.iter_mut().flatten() {
            if let std::borrow::Cow::Owned(function) = tool.function_template_tool() {
                *tool = function;
            }
        }
        if let Some(choice) = self.tool_choice.as_mut()
            && let std::borrow::Cow::Owned(function) = choice.function_template_choice()
        {
            *choice = function;
        }
        for message in &mut self.messages {
            if let ChatCompletionRequestMessage::Assistant(message) = message {
                for call in message.tool_calls.iter_mut().flatten() {
                    if let std::borrow::Cow::Owned(function) = call.function_template_call() {
                        *call = function;
                    }
                }
            }
        }
        Ok(custom_names)
    }

    /// Serialize history for a function-oriented template without changing API replay.
    pub fn function_template_messages(&self) -> serde_json::Value {
        chat_messages_for_function_template(&self.messages).expect("typed messages serialize")
    }
}

/// Shared by every renderer, including the typed-message fast path.
pub fn chat_messages_for_function_template(
    messages: &[ChatCompletionRequestMessage],
) -> Result<serde_json::Value, serde_json::Error> {
    let mut value = serde_json::to_value(messages)?;
    for (original, rendered) in messages
        .iter()
        .zip(value.as_array_mut().into_iter().flatten())
    {
        if let ChatCompletionRequestMessage::Assistant(message) = original
            && let Some(calls) = &message.tool_calls
            && let Some(rendered_calls) = rendered
                .get_mut("tool_calls")
                .and_then(serde_json::Value::as_array_mut)
        {
            for (call, rendered) in calls.iter().zip(rendered_calls) {
                if let std::borrow::Cow::Owned(function) = call.function_template_call() {
                    *rendered = serde_json::to_value(function)?;
                }
            }
        }
    }
    Ok(value)
}

// Tagged wire types keep public schemas faithful while existing consumers retain
// their function fields. Custom declarations/calls cache a function view internally;
// serialization always returns the original custom API shape.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionTool))]
enum ToolWire {
    Function {
        function: FunctionObject,
        defer_loading: Option<bool>,
    },
    Custom {
        custom: ChatCompletionCustomTool,
    },
}

impl<'de> Deserialize<'de> for ChatCompletionTool {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut tool = match ToolWire::deserialize(deserializer)? {
            ToolWire::Function {
                function,
                defer_loading,
            } => Self {
                function,
                defer_loading,
                ..Self::default()
            },
            ToolWire::Custom { custom } => Self {
                r#type: ChatCompletionToolType::Custom,
                function: FunctionObject {
                    name: custom.name.clone(),
                    description: custom.description.clone(),
                    parameters: Some(custom_tool_parameters()),
                    ..FunctionObject::default()
                },
                custom: Some(custom),
                defer_loading: None,
            },
        };
        tool.normalize_defer_loading();
        Ok(tool)
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionMessageToolCall))]
enum CallWire {
    Function { id: String, function: FunctionCall },
    Custom { id: String, custom: CustomToolCall },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionNamedToolChoice))]
enum ChoiceWire {
    Function { function: FunctionName },
    Custom { custom: FunctionName },
}

impl<'de> Deserialize<'de> for ChatCompletionNamedToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match ChoiceWire::deserialize(deserializer)? {
            ChoiceWire::Function { function } => Self {
                r#type: ChatCompletionToolType::Function,
                function,
            },
            ChoiceWire::Custom { custom } => Self {
                r#type: ChatCompletionToolType::Custom,
                function: custom,
            },
        })
    }
}

impl Serialize for ChatCompletionNamedToolChoice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serializer.serialize_struct("ChatCompletionNamedToolChoice", 2)?;
        value.serialize_field("type", &self.r#type)?;
        value.serialize_field(
            if self.r#type == ChatCompletionToolType::Custom {
                "custom"
            } else {
                "function"
            },
            &self.function,
        )?;
        value.end()
    }
}

impl<'de> Deserialize<'de> for ChatCompletionMessageToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Historical Dynamo callers omit type on function calls; keep that alias.
        // Deserialize typed fields directly so duplicate fields remain errors.
        #[derive(Deserialize)]
        struct Input {
            id: String,
            #[serde(default)]
            r#type: FunctionType,
            function: Option<FunctionCall>,
            custom: Option<CustomToolCall>,
        }
        let input = Input::deserialize(deserializer)?;
        if input.function.is_some() && input.custom.is_some() {
            return Err(D::Error::custom(
                "tool call cannot contain both function and custom",
            ));
        }
        let wire = match input.r#type {
            FunctionType::Function => CallWire::Function {
                id: input.id,
                function: input
                    .function
                    .ok_or_else(|| D::Error::missing_field("function"))?,
            },
            FunctionType::Custom => CallWire::Custom {
                id: input.id,
                custom: input
                    .custom
                    .ok_or_else(|| D::Error::missing_field("custom"))?,
            },
        };
        Ok(match wire {
            CallWire::Function { id, function } => Self {
                id,
                r#type: FunctionType::Function,
                function,
                custom: None,
            },
            CallWire::Custom { id, custom } => Self {
                id,
                r#type: FunctionType::Custom,
                function: FunctionCall {
                    name: custom.name.clone(),
                    arguments: serde_json::json!({CUSTOM_TOOL_INPUT_ARG:custom.input}).to_string(),
                },
                custom: Some(custom),
            },
        })
    }
}

impl Serialize for ChatCompletionMessageToolCall {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serializer.serialize_struct("ChatCompletionMessageToolCall", 3)?;
        value.serialize_field("id", &self.id)?;
        value.serialize_field("type", &self.r#type)?;
        match self.r#type {
            FunctionType::Function => value.serialize_field("function", &self.function)?,
            FunctionType::Custom => value.serialize_field(
                "custom",
                self.custom
                    .as_ref()
                    .ok_or_else(|| S::Error::custom("custom call requires custom input"))?,
            )?,
        }
        value.end()
    }
}

#[cfg(feature = "protocol-schema")]
macro_rules! wire_schema {
    ($public:ty, $wire:ty) => {
        impl utoipa::PartialSchema for $public {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
                <$wire>::schema()
            }
        }
        impl utoipa::ToSchema for $public {
            fn name() -> std::borrow::Cow<'static, str> {
                <$wire>::name()
            }
            fn schemas(
                schemas: &mut Vec<(String, utoipa::openapi::RefOr<utoipa::openapi::Schema>)>,
            ) {
                <$wire>::schemas(schemas);
            }
        }
    };
}
#[cfg(feature = "protocol-schema")]
wire_schema!(ChatCompletionTool, ToolWire);
#[cfg(feature = "protocol-schema")]
wire_schema!(ChatCompletionMessageToolCall, CallWire);
#[cfg(feature = "protocol-schema")]
wire_schema!(ChatCompletionNamedToolChoice, ChoiceWire);
