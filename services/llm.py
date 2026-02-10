from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from app.config import OPENAI_API_KEY


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question in one word only:\n{question}"
)


def ask_ai(question: str) -> str:
    prompt = prompt_template.format(question=question)

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    return response.content.strip()
