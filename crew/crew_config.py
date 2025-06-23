from crewai import Crew, Task
from crew.agent import get_chat_agent


def get_crew(user_query: str, llm_callable):
    agent = get_chat_agent(llm_callable=llm_callable)

    task = Task(
        description=(
            f"You're an expert Data Science assistant. Explain the following term clearly:\n\n"
            f"'{user_query}'\n\n"
            "Make sure your answer includes definitions, examples, analogies, and code if applicable."
        ),
        expected_output="A technical yet clear explanation that teaches the concept to the user.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    return crew
