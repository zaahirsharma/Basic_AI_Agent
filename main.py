from dotenv import load_dotenv
from pydantic import BaseModel
# Make choose between ChatOpenAI and or Claude models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# For prompt templates and output parsing
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()


# Setting up a prompt template for the LLM
# Define class, specify content for LLM to generate

class ResearchResponse(BaseModel):
    # Generate topic (str), summary (str), sources (list of strings), and tools_used (list of strings)
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

# Defining llm models 

# OpenAI used for creating a script
llm = ChatOpenAI(model='gpt-4o')
# Cluade used for generating a of information in markdown format
# llm2 = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Take output of LLM and parse it into ReasearchResponse object
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Create a prompt template for the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful research assistant that will analyze a research topic from the user's query using the neccessary tools.
            You will generate a research topic, a brief summary of the topic, a list of sources in form of "Website Title: url", and tools used in the research.
            Wrap the output in this format and provide no other text \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create the agent and executor
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt, 
    tools = []
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    verbose=True, # To see the agent's reasoning and steps
)

# Need to pass in the variable {query} to the agent which is not filled in by agent executor
raw_response = agent_executor.invoke({"query": "What is the impact of climate change on global agriculture?"})


# Parse the raw response from the agent into a structured format (ResearchResponse)
# Model could mess up so implement error handling here
try:
    structured_response = parser.parse(raw_response.get('output'))
except Exception as e:
    print(f"Error parsing response: {e}\n Raw Response: {raw_response}")
