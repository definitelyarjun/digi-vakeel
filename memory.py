"""
Referencing the following documentation for LangChain Memory:
https://www.aurelio.ai/learn/langchain-conversational-memory#2-conversationbufferwindowmemory
"""

import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ""


# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

#basic memory for conversation
memory = ConversationBufferMemory(return_messages=True)

memory.chat_memory.add_user_message("Hello, my name is JD")
memory.chat_memory.add_ai_message("Hello JD, how can I help you today?")
# memory.chat_memory.add_user_message("I am looking for a job in the tech industry. Can you help me with that?")

chain = ConversationChain(
    llm = llm,
    memory = memory,
    verbose=True
)

# print(memory.chat_memory.messages)

# result = chain.invoke("What is my name?")
# print(result.get("response"))

"""useing runnable interface to invoke the chain"""

from langchain.prompts import (
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)

system_prompt = "You are legal assitant specializing in consumer laws in India"

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}")
    ]
)


pipeline = prompt_template | llm

from langchain_core.chat_history import InMemoryChatMessageHistory

chat_map = {}
# Function to get or create chat history for a session
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

from langchain_core.runnables.history import RunnableWithMessageHistory

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

# result = pipeline_with_history.invoke(
#     {"query": "Hi My name is JD?"},
#     config={"session_id": "id_123"}
# )

# res2 = pipeline_with_history.invoke(
#     {"query": "What is my name again?"},
#     config={"session_id": "id_123"}
# )

# print(res2.content)

"""Testing Conversation Summary"""
# from langchain.memory import ConversationSummaryBufferMemory

# memory = ConversationSummaryBufferMemory(llm=llm)
# chain = ConversationChain(
#     llm=llm, 
#     memory = memory,
#     verbose=True
# )

# chain.invoke({"input": "hello there my name is Josh"})
# chain.invoke({"input": "I am researching the different types of conversational memory."})
# chain.invoke({"input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory."})
# chain.invoke({"input": "Buffer memory just stores the entire conversation"})
# chain.invoke({"input": "Buffer window memory stores the last k messages, dropping the rest."})

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=4, return_messages=True)

memory.chat_memory.add_user_message("Hi, my name is Josh")
memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
memory.chat_memory.add_ai_message("Very cool!")

# print(memory.chat_memory.messages)

chain = ConversationChain(
    llm = llm,
    memory = memory,
    verbose=True
)

# print(chain.invoke("What is my name?")["response"])
# print(chain.invoke({"input": "what is my name again?"})["response"])


"""
Using modern chat history with Pydantic models
"""
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    # remove anything beyond the last
    return chat_map[session_id]

from langchain_core.runnables import ConfigurableFieldSpec

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

result = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)


print("PRINTING RESULT")
print(result)

chat_map["id_k4"].clear()  # clear the history

# manually insert history
chat_map["id_k4"].add_user_message("Hi, my name is Josh")
chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
chat_map["id_k4"].add_ai_message("Very cool!")

# if we now view the messages, we'll see that ONLY the last 4 messages are stored
print(chat_map["id_k4"].messages)

result2 = pipeline_with_history.invoke(
    {"query": "what is my name again?"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)

print("Printing Result 2")
print(result2)