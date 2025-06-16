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

            llm_response = self.llm_client.get_response(messages)
            result = await self.process_llm_response(llm_response)

            if result != llm_response:
                messages.extend(
                    [
                        {"role": "assistant", "content": llm_response},
                        {"role": "system", "content": result},
                    ]
                )
                final_response = self.llm_client.get_response(messages)
                return final_response
            return llm_response

        except Exception as e:
            return f"Error processing message: {e}"

    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation context with system message"""
        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{st.session_state.tools_description}\n"
            # ...existing system message content...
        )
        return [
            {"role": "system", "content": system_message}
        ] + st.session_state.messages

    def run_streamlit(self):
        """Main Streamlit interface"""
        st.title("AI Assistant")

        # Initialize servers if not done
        if not st.session_state.initialized:
            if not asyncio.run(self.initialize_servers()):
                st.stop()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if user_input := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

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
