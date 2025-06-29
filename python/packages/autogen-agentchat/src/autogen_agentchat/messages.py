"""
This module defines various message types used for agent-to-agent communication.
Each message type inherits either from the BaseChatMessage class or BaseAgentEvent
class and includes specific fields relevant to the type of message being sent.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Literal, Mapping, Optional, Type, TypeVar

from autogen_core import Component, ComponentBase, FunctionCall, Image
from autogen_core.code_executor import CodeBlock, CodeResult
from autogen_core.memory import MemoryContent
from autogen_core.models import (
    FunctionExecutionResult,
    LLMMessage,
    RequestUsage,
    UserMessage,
)
from autogen_core.utils import schema_to_pydantic_model
from pydantic import BaseModel, Field, computed_field
from typing_extensions import Annotated, Self


class BaseMessage(BaseModel, ABC):
    """Abstract base class for all message types in AgentChat.

    .. warning::

        If you want to create a new message type, do not inherit from this class.
        Instead, inherit from :class:`BaseChatMessage` or :class:`BaseAgentEvent`
        to clarify the purpose of the message type.

    """

    @abstractmethod
    def to_text(self) -> str:
        """Convert the message content to a string-only representation
        that can be rendered in the console and inspected by the user or conditions.
        This is not used for creating text-only content for models.
        For :class:`BaseChatMessage` types, use :meth:`to_model_text` instead."""
        ...

    '''
    这段代码中的三个点 `...` 是 Python 中的特殊语法，表示一个抽象方法的占位符。在这个上下文中，它有以下含义：
    1. **抽象方法标记** - 在抽象基类 (ABC) 中，`...` 用于表示一个必须由子类实现的方法。这是 Python 3.0+ 的官方语法，用于定义方法体为空的抽象方法。
    2. **pass 的现代替代** - 在早期的 Python 版本中，通常使用 `pass` 关键字作为空方法体的占位符，但 `...` 作为 Python 的官方语法提供了更明确的含义，特别是在抽象方法中。
    3. **实现要求** - 这个语法告诉子类的开发者：这个方法必须被重写，否则实例化该子类时会引发错误。
    '''


    def dump(self) -> Mapping[str, Any]:
        """Convert the message to a JSON-serializable dictionary.

        The default implementation uses the Pydantic model's
        :meth:`model_dump` method to convert the message to a dictionary.
        Override this method if you want to customize the serialization
        process or add additional fields to the output.
        """
        return self.model_dump()
    '''
    `Mapping` 是 `collections.abc` 模块中定义的一个抽象基类（ABC），它代表了键值对的集合，类似于字典
    1. **只读接口**：`Mapping` 只提供访问操作，不包括修改操作（如 `__setitem__`）[[2]](https://realpython.com/python-mappings/)
    2. **抽象性**：它是一个抽象接口，定义了所有映射类型都应该实现的方法，包括 `__getitem__`, `__contains__`, `keys()`, `items()`, `values()` 等
    3. **比 `Dict` 更通用**：在类型提示中，`Mapping` 可以接受任何实现了映射接口的对象，而不仅仅是 `dict` 类型 [[3]](https://stackoverflow.com/questions/52487663/python-type-hints-typing-mapping-vs-typing-dict)

    '''


    @classmethod
    def load(cls, data: Mapping[str, Any]) -> Self:
        """Create a message from a dictionary of JSON-serializable data.

        The default implementation uses the Pydantic model's
        :meth:`model_validate` method to create the message from the data.
        Override this method if you want to customize the deserialization
        process or add additional fields to the input data."""
        return cls.model_validate(data)
    '''
    model_validate 是继承自 pydantic 的 BaseModel
    '''


class BaseChatMessage(BaseMessage, ABC):
    """Abstract base class for chat messages.

    .. note::

        If you want to create a new message type that is used for agent-to-agent
        communication, inherit from this class, or simply use
        :class:`StructuredMessage` if your content type is a subclass of
        Pydantic BaseModel.

    This class is used for messages that are sent between agents in a chat
    conversation. Agents are expected to process the content of the
    message using models and return a response as another :class:`BaseChatMessage`.
    """

    '''
    agent-to-agent, 难道StructuredMessage不适合在agent之间通信吗？
    '''

    source: str
    """The name of the agent that sent this message."""

    models_usage: RequestUsage | None = None
    """The model client usage incurred when producing this message."""

    metadata: Dict[str, str] = {}
    """Additional metadata about the message."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """The time when the message was created."""

    @abstractmethod
    def to_model_text(self) -> str:
        """Convert the content of the message to text-only representation.
        This is used for creating text-only content for models.

        This is not used for rendering the message in console. For that, use
        :meth:`~BaseMessage.to_text`.

        The difference between this and :meth:`to_model_message` is that this
        is used to construct parts of the a message for the model client,
        while :meth:`to_model_message` is used to create a complete message
        for the model client.
        """
        ...
    '''
    这个to_model_text用于结构化，结构化部分信息？
    这个方法有以下特点：
    1. **转换为纯文本**：它将消息内容转换为纯文本表示形式，这是因为许多语言模型接受的是纯文本输入。
    2. **用于模型输入**：这个方法生成的文本是专门用于发送给语言模型的，而不是用于控制台显示。
    3. **部分而非整体**：它只负责转换消息的内容部分，而不是创建完整的模型消息。这些转换后的文本可能成为发送给模型的消息的一部分。
    
    ### 与其他方法的区别
    文档中提到了两个相关但不同的方法：
    1. **`to_text`**：用于控制台显示和用户查看，面向人类阅读。
    2. **`to_model_message`**：用于创建完整的模型消息，而不仅仅是内容部分。这可能包括消息类型、角色、元数据等

    '''

    @abstractmethod
    def to_model_message(self) -> UserMessage:
        """Convert the message content to a :class:`~autogen_core.models.UserMessage`
        for use with model client, e.g., :class:`~autogen_core.models.ChatCompletionClient`.
        """
        ...


class BaseTextChatMessage(BaseChatMessage, ABC):
    """Base class for all text-only :class:`BaseChatMessage` types.
    It has implementations for :meth:`to_text`, :meth:`to_model_text`,
    and :meth:`to_model_message` methods.

    Inherit from this class if your message content type is a string.
    """

    content: str
    """The content of the message."""

    def to_text(self) -> str:
        return self.content

    def to_model_text(self) -> str:
        return self.content

    # 你这里全部返回content，有啥区别呢！

    def to_model_message(self) -> UserMessage:
        return UserMessage(content=self.content, source=self.source)


class BaseAgentEvent(BaseMessage, ABC):
    """Base class for agent events.

    .. note::

        If you want to create a new message type for signaling observable events
        to user and application, inherit from this class.

    Agent events are used to signal actions and thoughts produced by agents
    and teams to user and applications. They are not used for agent-to-agent
    communication and are not expected to be processed by other agents.

    You should override the :meth:`to_text` method if you want to provide
    a custom rendering of the content.
    """

    source: str
    """The name of the agent that sent this message."""

    models_usage: RequestUsage | None = None
    """The model client usage incurred when producing this message."""

    '''
    1. **类型注解中的 `None`**: `RequestUsage | None` 中的 `None` 是类型的一部分，它告诉类型检查器（如 mypy、PyCharm 等）这个变量可以是 `None` 值。
    2. **赋值中的 `None`**: `= None` 中的 `None` 是实际赋给变量的默认值。
    '''

    metadata: Dict[str, str] = {}
    """Additional metadata about the message."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """The time when the message was created."""


StructuredContentType = TypeVar("StructuredContentType", bound=BaseModel, covariant=True)
"""Type variable for structured content types."""

'''
1. **TypeVar**：这是 Python `typing` 模块中的一个工具，用于创建泛型类型变量。
2. **"StructuredContentType"**：这是类型变量的名称，通常作为第一个参数传递给 `TypeVar`。
3. **bound=BaseModel**：这个参数限制了这个类型变量只能是 `BaseModel` 或其子类。`BaseModel` 很可能是来自 Pydantic 库的基类，用于创建可验证的数据模型。
4. **covariant=True**：这表示该类型变量是协变的（covariant）。

'''

'''
`Generic` 是一个特殊的类，用于创建参数化的类型，允许类根据不同的类型参数表现出不同的行为，同时保持类型安全。简单来说，它让你可以创建能够处理不同类型的容器类，并在类型检查时保持类型信息。
from typing import Generic, TypeVar

T = TypeVar('T')  # 定义一个类型变量

class Box(Generic[T]):
    def __init__(self, content: T):
        self.content = content
    
    def get_content(self) -> T:
        return self.content
# 在这个例子中：
# - `Box` 是一个泛型类，可以包含任何类型的内容
# - 当创建 `Box` 实例时，类型检查器会跟踪内容的类型
'''

# 这里 `StructuredMessage` 是一个泛型类，它可以处理任何 `BaseModel` 的子类作为内容
'''
class UserProfile(BaseModel):
    name: str
    age: int

# 创建一个包含 UserProfile 的消息
user_message = StructuredMessage[UserProfile](content=UserProfile(name="Alice", age=30))
当我们写 `StructuredMessage[UserProfile]` 时，我们是在做以下操作：
1. 选择泛型类 `StructuredMessage`
2. 用具体类型 `UserProfile` 替换泛型参数 `StructuredContentType`
3. 生成一个新的、特定于 `UserProfile` 的类型

'''
# 为啥不实用Any
'''
## Any 的问题
`Any` 类型本质上是关闭类型检查的一种方式。当你使用 `Any` 时：
1. **丢失类型安全**：编译器/类型检查器无法检测类型错误
2. **缺少代码补全**：IDE 无法提供准确的自动补全建议
3. **文档不明确**：代码阅读者无法知道实际预期的类型
4. **错误传播**：`Any` 类型会"感染"其他变量，导致更多类型信息丢失

'''

class StructuredMessage(BaseChatMessage, Generic[StructuredContentType]):
    """A :class:`BaseChatMessage` type with an unspecified content type.

    To create a new structured message type, specify the content type
    as a subclass of `Pydantic BaseModel <https://docs.pydantic.dev/latest/concepts/models/>`_.

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessage


        class MyMessageContent(BaseModel):
            text: str
            number: int


        message = StructuredMessage[MyMessageContent](
            content=MyMessageContent(text="Hello", number=42),
            source="agent1",
        )

        print(message.to_text())  # {"text": "Hello", "number": 42}

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessage


        class MyMessageContent(BaseModel):
            text: str
            number: int


        message = StructuredMessage[MyMessageContent](
            content=MyMessageContent(text="Hello", number=42),
            source="agent",
            format_string="Hello, {text} {number}!",
        )

        print(message.to_text())  # Hello, agent 42!

    """

    content: StructuredContentType
    """The content of the message. Must be a subclass of
    `Pydantic BaseModel <https://docs.pydantic.dev/latest/concepts/models/>`_."""

    format_string: Optional[str] = None
    """(Experimental) An optional format string to render the content into a human-readable format.
    The format string can use the fields of the content model as placeholders.
    For example, if the content model has a field `name`, you can use
    `{name}` in the format string to include the value of that field.
    The format string is used in the :meth:`to_text` method to create a
    human-readable representation of the message.
    This setting is experimental and will change in the future.
    """

    '''
    `@computed_field` 是 Pydantic 中的一个装饰器，用于在 Pydantic 模型中创建计算属性（computed properties）。这些属性会：
    1. 根据模型中其他字段的值动态计算
    2. 包含在模型的序列化输出中
    3. 包含在模型的 JSON Schema 中
    虽然 `@computed_field` 类似于 Python 的 `@property` 装饰器，但有几个重要区别：
    1. `@computed_field` 装饰的属性会包含在模型的 JSON 输出中
    2. 它们会被包含在模型的 JSON Schema 中
    3. 可以配置额外参数，如 `return_type`、`alias` 等

    '''

    @computed_field
    def type(self) -> str:
        return self.__class__.__name__

    def to_text(self) -> str:
        if self.format_string is not None:
            return self.format_string.format(**self.content.model_dump())
        else:
            return self.content.model_dump_json()

    def to_model_text(self) -> str:
        if self.format_string is not None:
            return self.format_string.format(**self.content.model_dump())
        else:
            return self.content.model_dump_json()

    def to_model_message(self) -> UserMessage:
        return UserMessage(
            content=self.content.model_dump_json(),
            source=self.source,
        )


class StructureMessageConfig(BaseModel):
    """The declarative configuration for the structured output."""

    json_schema: Dict[str, Any]
    format_string: Optional[str] = None
    content_model_name: str


class StructuredMessageFactory(ComponentBase[StructureMessageConfig], Component[StructureMessageConfig]):
    """:meta private:

    A component that creates structured chat messages from Pydantic models or JSON schemas.

    This component helps you generate strongly-typed chat messages with content defined using a Pydantic model.
    It can be used in declarative workflows where message structure must be validated, formatted, and serialized.

    You can initialize the component directly using a `BaseModel` subclass, or dynamically from a configuration
    object (e.g., loaded from disk or a database).

    ### Example 1: Create from a Pydantic Model

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessageFactory


        class TestContent(BaseModel):
            field1: str
            field2: int


        format_string = "This is a string {field1} and this is an int {field2}"
        sm_component = StructuredMessageFactory(input_model=TestContent, format_string=format_string)

        message = sm_component.StructuredMessage(
            source="test_agent", content=TestContent(field1="Hello", field2=42), format_string=format_string
        )

        print(message.to_model_text())  # Output: This is a string Hello and this is an int 42

        config = sm_component.dump_component()

        s_m_dyn = StructuredMessageFactory.load_component(config)
        message = s_m_dyn.StructuredMessage(
            source="test_agent",
            content=s_m_dyn.ContentModel(field1="dyn agent", field2=43),
            format_string=s_m_dyn.format_string,
        )
        print(type(message))  # StructuredMessage[GeneratedModel]
        print(message.to_model_text())  # Output: This is a string dyn agent and this is an int 43

    Attributes:
        component_config_schema (StructureMessageConfig): Defines the configuration structure for this component.
        component_provider_override (str): Path used to reference this component in external tooling.
        component_type (str): Identifier used for categorization (e.g., "structured_message").

    Raises:
        ValueError: If neither `json_schema` nor `input_model` is provided.

    Args:
        json_schema (Optional[str]): JSON schema to dynamically create a Pydantic model.
        input_model (Optional[Type[BaseModel]]): A subclass of `BaseModel` that defines the expected message structure.
        format_string (Optional[str]): Optional string to render content into a human-readable format.
        content_model_name (Optional[str]): Optional name for the generated Pydantic model.
    """

    component_config_schema = StructureMessageConfig
    component_provider_override = "autogen_agentchat.messages.StructuredMessageFactory"
    component_type = "structured_message"

    def __init__(
        self,
        json_schema: Optional[Dict[str, Any]] = None,
        input_model: Optional[Type[BaseModel]] = None,
        format_string: Optional[str] = None,
        content_model_name: Optional[str] = None,
    ) -> None:
        self.format_string = format_string

        if json_schema:
            self.ContentModel = schema_to_pydantic_model(
                json_schema, model_name=content_model_name or "GeneratedContentModel"
            )
        elif input_model:
            self.ContentModel = input_model
        else:
            raise ValueError("Either `json_schema` or `input_model` must be provided.")

        self.StructuredMessage = StructuredMessage[self.ContentModel]  # type: ignore[name-defined]

    def _to_config(self) -> StructureMessageConfig:
        return StructureMessageConfig(
            json_schema=self.ContentModel.model_json_schema(),
            format_string=self.format_string,
            content_model_name=self.ContentModel.__name__,
        )

    @classmethod
    def _from_config(cls, config: StructureMessageConfig) -> "StructuredMessageFactory":
        return cls(
            json_schema=config.json_schema,
            format_string=config.format_string,
            content_model_name=config.content_model_name,
        )


class TextMessage(BaseTextChatMessage):
    """A text message with string-only content."""

    type: Literal["TextMessage"] = "TextMessage"


class MultiModalMessage(BaseChatMessage):
    """A multimodal message."""

    content: List[str | Image]
    """The content of the message."""

    type: Literal["MultiModalMessage"] = "MultiModalMessage"

    def to_model_text(self, image_placeholder: str | None = "[image]") -> str:
        """Convert the content of the message to a string-only representation.
        If an image is present, it will be replaced with the image placeholder
        by default, otherwise it will be a base64 string when set to None.
        """
        text = ""
        for c in self.content:
            if isinstance(c, str):
                text += c
            elif isinstance(c, Image):
                if image_placeholder is not None:
                    text += f" {image_placeholder}"
                else:
                    text += f" {c.to_base64()}"
        return text

    def to_text(self, iterm: bool = False) -> str:
        result: List[str] = []
        for c in self.content:
            if isinstance(c, str):
                result.append(c)
            else:
                if iterm:
                    # iTerm2 image rendering protocol: https://iterm2.com/documentation-images.html
                    image_data = c.to_base64()
                    result.append(f"\033]1337;File=inline=1:{image_data}\a\n")
                else:
                    result.append("<image>")
        return "\n".join(result)

    def to_model_message(self) -> UserMessage:
        return UserMessage(content=self.content, source=self.source)


class StopMessage(BaseTextChatMessage):
    """A message requesting stop of a conversation."""

    type: Literal["StopMessage"] = "StopMessage"


class HandoffMessage(BaseTextChatMessage):
    """A message requesting handoff of a conversation to another agent."""

    target: str
    """The name of the target agent to handoff to."""

    context: List[LLMMessage] = []
    """The model context to be passed to the target agent."""

    type: Literal["HandoffMessage"] = "HandoffMessage"


class ToolCallSummaryMessage(BaseTextChatMessage):
    """A message signaling the summary of tool call results."""

    type: Literal["ToolCallSummaryMessage"] = "ToolCallSummaryMessage"

    tool_calls: List[FunctionCall]
    """The tool calls that were made."""

    results: List[FunctionExecutionResult]
    """The results of the tool calls."""


class ToolCallRequestEvent(BaseAgentEvent):
    """An event signaling a request to use tools."""

    content: List[FunctionCall]
    """The tool calls."""

    type: Literal["ToolCallRequestEvent"] = "ToolCallRequestEvent"

    def to_text(self) -> str:
        return str(self.content)


class CodeGenerationEvent(BaseAgentEvent):
    """An event signaling code generation event."""

    retry_attempt: int
    "Retry number, 0 means first generation"

    content: str
    "The complete content as string."

    code_blocks: List[CodeBlock]
    "List of code blocks present in content"

    type: Literal["CodeGenerationEvent"] = "CodeGenerationEvent"

    def to_text(self) -> str:
        return self.content


class CodeExecutionEvent(BaseAgentEvent):
    """An event signaling code execution event."""

    retry_attempt: int
    "Retry number, 0 means first execution"

    result: CodeResult
    "Code Execution Result"

    type: Literal["CodeExecutionEvent"] = "CodeExecutionEvent"

    def to_text(self) -> str:
        return self.result.output


class ToolCallExecutionEvent(BaseAgentEvent):
    """An event signaling the execution of tool calls."""

    content: List[FunctionExecutionResult]
    """The tool call results."""

    type: Literal["ToolCallExecutionEvent"] = "ToolCallExecutionEvent"

    def to_text(self) -> str:
        return str(self.content)


class UserInputRequestedEvent(BaseAgentEvent):
    """An event signaling a that the user proxy has requested user input. Published prior to invoking the input callback."""

    request_id: str
    """Identifier for the user input request."""

    content: Literal[""] = ""
    """Empty content for compat with consumers expecting a content field."""

    type: Literal["UserInputRequestedEvent"] = "UserInputRequestedEvent"

    def to_text(self) -> str:
        return str(self.content)


class MemoryQueryEvent(BaseAgentEvent):
    """An event signaling the results of memory queries."""

    content: List[MemoryContent]
    """The memory query results."""

    type: Literal["MemoryQueryEvent"] = "MemoryQueryEvent"

    def to_text(self) -> str:
        return str(self.content)


class ModelClientStreamingChunkEvent(BaseAgentEvent):
    """An event signaling a text output chunk from a model client in streaming mode."""

    content: str
    """A string chunk from the model client."""

    type: Literal["ModelClientStreamingChunkEvent"] = "ModelClientStreamingChunkEvent"

    def to_text(self) -> str:
        return self.content


class ThoughtEvent(BaseAgentEvent):
    """An event signaling the thought process of a model.
    It is used to communicate the reasoning tokens generated by a reasoning model,
    or the extra text content generated by a function call."""

    content: str
    """The thought process of the model."""

    type: Literal["ThoughtEvent"] = "ThoughtEvent"

    def to_text(self) -> str:
        return self.content


class SelectSpeakerEvent(BaseAgentEvent):
    """An event signaling the selection of speakers for a conversation."""

    content: List[str]
    """The names of the selected speakers."""

    type: Literal["SelectSpeakerEvent"] = "SelectSpeakerEvent"

    def to_text(self) -> str:
        return str(self.content)


class SelectorEvent(BaseAgentEvent):
    """An event emitted from the `SelectorGroupChat`."""

    content: str
    """The content of the event."""

    type: Literal["SelectorEvent"] = "SelectorEvent"

    def to_text(self) -> str:
        return str(self.content)


class MessageFactory:
    """:meta private:

    A factory for creating messages from JSON-serializable dictionaries.

    This is useful for deserializing messages from JSON data.
    """

    def __init__(self) -> None:
        self._message_types: Dict[str, type[BaseAgentEvent | BaseChatMessage]] = {}
        # Register all message types.
        self._message_types[TextMessage.__name__] = TextMessage
        self._message_types[MultiModalMessage.__name__] = MultiModalMessage
        self._message_types[StopMessage.__name__] = StopMessage
        self._message_types[ToolCallSummaryMessage.__name__] = ToolCallSummaryMessage
        self._message_types[HandoffMessage.__name__] = HandoffMessage
        self._message_types[ToolCallRequestEvent.__name__] = ToolCallRequestEvent
        self._message_types[ToolCallExecutionEvent.__name__] = ToolCallExecutionEvent
        self._message_types[MemoryQueryEvent.__name__] = MemoryQueryEvent
        self._message_types[UserInputRequestedEvent.__name__] = UserInputRequestedEvent
        self._message_types[ModelClientStreamingChunkEvent.__name__] = ModelClientStreamingChunkEvent
        self._message_types[ThoughtEvent.__name__] = ThoughtEvent
        self._message_types[SelectSpeakerEvent.__name__] = SelectSpeakerEvent
        self._message_types[CodeGenerationEvent.__name__] = CodeGenerationEvent
        self._message_types[CodeExecutionEvent.__name__] = CodeExecutionEvent
    '''
    1. `self._message_types` - 这是一个字典属性，用于存储消息类型的映射关系
    2. `CodeExecutionEvent.__name__` - 获取类 `CodeExecutionEvent` 的名称作为字典的键
    3. `CodeExecutionEvent` - 类本身作为字典的值

    '''
    def is_registered(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> bool:
        """Check if a message type is registered with the factory."""
        # Get the class name of the message type.
        class_name = message_type.__name__
        # Check if the class name is already registered.
        return class_name in self._message_types

    def register(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> None:
        """Register a new message type with the factory."""
        if self.is_registered(message_type):
            raise ValueError(f"Message type {message_type} is already registered.")
        if not issubclass(message_type, BaseChatMessage) and not issubclass(message_type, BaseAgentEvent):
            raise ValueError(f"Message type {message_type} must be a subclass of BaseChatMessage or BaseAgentEvent.")
        # Get the class name of the
        class_name = message_type.__name__
        # Check if the class name is already registered.
        # Register the message type.
        self._message_types[class_name] = message_type

    def create(self, data: Mapping[str, Any]) -> BaseAgentEvent | BaseChatMessage:
        """Create a message from a dictionary of JSON-serializable data."""
        # Get the type of the message from the dictionary.
        message_type = data.get("type")
        if message_type is None:
            raise ValueError("Field 'type' is required in the message data to recover the message type.")
        if message_type not in self._message_types:
            raise ValueError(f"Unknown message type: {message_type}")
        if not isinstance(message_type, str):
            raise ValueError(f"Message type must be a string, got {type(message_type)}")

        # Get the class for the message type.
        message_class = self._message_types[message_type]

        # Create an instance of the message class.
        assert issubclass(message_class, BaseChatMessage) or issubclass(message_class, BaseAgentEvent)
        return message_class.load(data)


ChatMessage = Annotated[
    TextMessage | MultiModalMessage | StopMessage | ToolCallSummaryMessage | HandoffMessage,
    Field(discriminator="type"),
]
"""The union type of all built-in concrete subclasses of :class:`BaseChatMessage`.
It does not include :class:`StructuredMessage` types."""

AgentEvent = Annotated[
    ToolCallRequestEvent
    | ToolCallExecutionEvent
    | MemoryQueryEvent
    | UserInputRequestedEvent
    | ModelClientStreamingChunkEvent
    | ThoughtEvent
    | SelectSpeakerEvent
    | CodeGenerationEvent
    | CodeExecutionEvent,
    Field(discriminator="type"),
]
"""The union type of all built-in concrete subclasses of :class:`BaseAgentEvent`."""


# __all__: 这是 Python 模块开发中的一种最佳实践，特别是在编写供他人使用的库时，它有助于维护清晰的 API 边界和接口稳定性。
# 1. **控制公共 API** - 它明确定义了这个模块想要公开给外部使用的类和函数
# 2. **限制通配符导入** - 当其他代码使用 `from autogen_agentchat.messages import *` 时，只有 列表中的名称会被导入，而不是模块中定义的所有名称 `__all__`
# 3. **文档目的** - 它为库的使用者和维护者提供了一个清晰的列表，表明哪些是模块的公共接口
# 4. **避免导入内部实现细节** - 它防止模块内部的辅助函数、变量或实现细节被意外导入
__all__ = [
    "AgentEvent",
    "BaseMessage",
    "ChatMessage",
    "BaseChatMessage",
    "BaseAgentEvent",
    "BaseTextChatMessage",
    "StructuredContentType",
    "StructuredMessage",
    "StructuredMessageFactory",
    "HandoffMessage",
    "MultiModalMessage",
    "StopMessage",
    "TextMessage",
    "ToolCallExecutionEvent",
    "ToolCallRequestEvent",
    "ToolCallSummaryMessage",
    "MemoryQueryEvent",
    "UserInputRequestedEvent",
    "ModelClientStreamingChunkEvent",
    "ThoughtEvent",
    "SelectSpeakerEvent",
    "MessageFactory",
    "CodeGenerationEvent",
    "CodeExecutionEvent",
]
