import streamlit as st
from typing import List, Dict
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import logging
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any
import httpx
from dotenv import load_dotenv
import os
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            logging.info("All fine till here")
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


# def run():
#     st.title("Groq Chatbot with Streamlit")
#     config = Configuration()
#     server_config = config.load_config("servers_config.json")
#     servers = [
#         Server(name, srv_config)
#         for name, srv_config in server_config["mcpServers"].items()
#     ]

#     client = Groq(api_key=st.secrets["GROQ_API_KEY"])

#     if "groq_model" not in st.session_state:
#         st.session_state["groq_model"] = "llama3-8b-8192"

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("What is up?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             response_container = st.empty()
#             full_response = ""
#             for chunk in client.chat.completions.create(
#                 model=st.session_state["groq_model"],
#                 messages=st.session_state.messages,
#                 stream=True,
#             ):
#                 delta = chunk.choices[0].delta.content or ""
#                 full_response += delta
#                 response_container.markdown(full_response + "▌")
#             response_container.markdown(full_response)
#         st.session_state.messages.append(
#             {"role": "assistant", "content": full_response}
#         )


class ChatbotApp:
    def __init__(self):
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "initialized" not in st.session_state:
            st.session_state.initialized = False

    async def initialize_servers(self):
        """Initialize servers and tools"""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    st.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return False

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            st.session_state.tools_description = "\n".join(
                [tool.format_for_llm() for tool in all_tools]
            )
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            return False

    async def process_message(self, user_input: str) -> str:
        """Process a single message and return the response"""
        try:
            messages = self._get_conversation_context()
            messages.append({"role": "user", "content": user_input})

            llm_response = (
                self.client.chat.completions.create(
                    model=st.session_state["groq_model"],
                    messages=messages,
                    stream=False,
                )
                .choices[0]
                .message.content
            )

            # Process the LLM response if needed
            # Here you can add any additional processing logic if required

            return llm_response

        except Exception as e:
            return f"Error processing message: {e}"

    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation context with system message"""
        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{st.session_state.tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        return [
            {"role": "system", "content": system_message}
        ] + st.session_state.messages

    def run_streamlit(self):
        """Main Streamlit interface"""
        st.title("Groq Chatbot with Streamlit")
        config = Configuration()
        server_config = config.load_config("servers_config.json")
        self.servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        if "groq_model" not in st.session_state:
            st.session_state["groq_model"] = "llama3-8b-8192"
        # Initialize servers if not done
        if not st.session_state.initialized:
            if not asyncio.run(self.initialize_servers()):
                st.stop()

        # Chat input
        if user_input := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(self.process_message(user_input))
                    st.write(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )


def main():
    app = ChatbotApp()
    app.run_streamlit()


if __name__ == "__main__":
    main()
