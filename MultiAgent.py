import warnings
from crewai import Agent, Task, Crew, Process
import os

from crewai_tools.tools.xml_search_tool.xml_search_tool import XMLSearchTool
from langchain_openai import ChatOpenAI
# from utils import get_openai_api_key
from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool
#which markets ADNOC interested in and trade in, where are the news about these markets,
from IPython.display import Markdown
import json
from pydantic import BaseModel, PrivateAttr
from utils import get_openai_api_key
from dotenv import load_dotenv
from tavily import TavilyClient
from crewai_tools import BaseTool

# Define a Pydantic model for news details
# class NewsDetails(BaseModel):
#     Link: str
#     Title: str
#     Summary: str

# Load environment variables from .env file

warnings.filterwarnings('ignore')

openai_api_key = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'

xml_tool = XMLSearchTool(xml='./RSS/GoogleNews.xml')

#define scraping tool
scrape_tool = ScrapeWebsiteTool()
# Define TavilyAPI tool
class TavilyAPI(BaseTool):
    name: str = "TavilyAPI"
    description: str = ("The best search engine to use. If you want to search for anything, USE IT! "
                        "Make sure your queries are very specific or else you will "
                        "get websites that have the same content and that will waste your time.")

    _client: TavilyClient = PrivateAttr()

    def __init__(self, api_key: str):
        super().__init__()
        self._client = TavilyClient(api_key=api_key)

    def _run(self, query: str) -> list:
        response = self._client.search(query=query, search_depth='basic', max_results=10)
        results = [{"Link": result["url"], "Title": result["title"]} for result in response["results"]]
        return results


load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

serper_tool = SerperDevTool()
# Initialize the TavilyAPI tool
tavily_tool = TavilyAPI(api_key=tavily_api_key)

# Define NewsSaver tool
# class NewsSaver(BaseTool):
#     name: str = "NewsSaver"
#     description: str = "Saves the provided news data into a JSON file."
#
#     def _run(self, news_data: list) -> str:
#         try:
#             with open('news_report.json', 'w') as f:
#                 json.dump(news_data, f, indent=2)
#             return "News data successfully saved to news_report.json."
#         except Exception as e:
#             return f"Failed to save news data: {e}"



docs_scrape_tool = ScrapeWebsiteTool(
    # website_url="https://www.worldoil.com/news/2024/6/23/adnoc-extends-vallourec-s-900-million-oil-and-gas-tubing-contract-to-2027/"
)

# Define the News Gatherer Agent
news_gatherer = Agent(
    role="News Gatherer",
    goal="To collect and compile a comprehensive list of URLs and titles "
         "from various news sources and RSS feeds related to specified topics in the energy market.",
    tools=[xml_tool],
    backstory="You are a dedicated and meticulous web crawler and aggregator, "
              "driven by a passion for information and data organization. "
              "Your skills in digital journalism and data scraping enable you "
              "to efficiently gather relevant news URLs from diverse sources.",
    allow_delegation=False,
    verbose=True,
)


news_analyst = Agent(
    role="Expert News Analyst",
    goal="Write insightful and factually accurate "
         "analytical piece from the data collected by the Researcher.",
    backstory="You're working on writing "
              "an analytical piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Senior News Researcher, who provides the relevant news "
              "and data about the topic: {topic}. "
              "You also provide accurate strategies and advice "
              "and back them up with information "
              "You acknowledge in your piece "
              "when your statements are analysis made by you.",
    verbose=True,
    memory=True,
)

writer = Agent(
    role="Editor",
    goal="Edit the analytical piece done by the Expert News Analyst "
         "into a professional and well structured news summary.",
    backstory="You are an experienced and well-trained editor who receives news analysis "
              "from the Expert News Analyst. "
              "Your goal is to review the analysis ensuring that the "
              "final report is clear and engaging.",
    verbose=True,
    memory=True
)

# Define the task for gathering news
news_gathering_task = Task(
    description=(
        "Collect a comprehensive list of URLs and their titles from various news sources,"
        " websites, and RSS feeds. Ensure that the URLs are current, relevant, and cover "
        "a wide range of perspectives. Your goal is to gather a diverse set of links that "
        "provide the latest updates and insights on: {topics}. Each collected "
        "entry should include the URL and the title of the corresponding article or news piece."
    ),
    expected_output="A JSON file containing a list of the collected URLs and their titles. "
                    "Each entry in the list should be a dictionary with four keys: 'Link' "
                    "for the URL and 'Title' for the article's title and 'Summary' for the "
                    "article's summary and 'Date' for date of article published. The final "
                    "output should reflect a wide range of sources and perspectives, ensuring"
                    "the information is current and relevant.",
    output_file='news_report.json',
    agent=news_gatherer
)


analysis_task = Task(
    description=(
        "Analyze the gathered news articles from the News Researcher and identify key trends and insights in topic:{topic}. "
        "Summarize the information in a concise manner."
    ),
    expected_output='A summary report highlighting the key trends and insights from the analyzed news articles.',
    agent=news_analyst
)

editing_task = Task(
    description=(
        "Compile the analyzed information into a well-structured report. "
        "Ensure that the report is clear, engaging, and free of errors."
    ),
    expected_output='A final report in markdown format that is well-structured and engaging.',
    agent=writer
)

crew = Crew(
    agents=[news_gatherer],
    tasks=[news_gathering_task],

    #manager_llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7),
    process=Process.sequential,
    verbose=True
)

topics = [
    "Renewable Energy ", "Green Energy Initiatives", "Energy Transition",
    "Crude Oil Prices",'LNG Market', 'Carbon Emissions', 'Energy Policy',
    'Climate Change Impact','Energy Infrastructure','Power Generation',
    'Energy Security','Global Energy Markets','Energy Supply Chain',
    'Oil Refining',"Fuel Efficiency"
]


result = crew.kickoff(inputs={"topics": topics})

print(result)