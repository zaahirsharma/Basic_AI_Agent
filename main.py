from dotenv import load_dotenv
from pydantic import BaseModel
# Make choose between ChatOpenAI and or Claude models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# For prompt templates and output parsing
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Import tools 
from tools import search_tool, save_tool, wiki_tool

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

# Define the tools to be used by the agent from import
tools = [search_tool, save_tool, wiki_tool]

# Create the agent and executor
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt, 
    tools = tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # To see the agent's reasoning and steps
)


# Get user input for query
query = input("What topic do you need help learning?\n")
# Need to pass in the variable {query} to the agent which is not filled in by agent executor
raw_response = agent_executor.invoke({"query": query})


# Parse the raw response from the agent into a structured format (ResearchResponse)
# Model could mess up so implement error handling here
try:
    structured_response = parser.parse(raw_response.get('output'))
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}\n Raw Response: {raw_response}")
