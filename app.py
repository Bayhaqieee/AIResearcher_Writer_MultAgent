import os
from flask import Flask, render_template, request
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

app = Flask(__name__)

# Initialize the AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_API_KEY"),
    api_version=os.environ.get("AZURE_API_VERSION")
)

# This creates a CrewAI-compatible tool from scratch, making it stable.
class SearchTool(BaseTool):
    name: str = "Internet Search"
    description: str = "A tool to search the internet for recent and relevant information. Use it to find information on any topic."
    
    def _run(self, search_query: str) -> str:
        """The tool's main function."""
        # Uses the stable LangChain wrapper internally
        serper_wrapper = GoogleSerperAPIWrapper()
        return serper_wrapper.run(search_query)

search_tool = SearchTool()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_crew', methods=['POST'])
def run_crew():
    topic = request.form['topic']
    year = request.form['year']
    output_format = request.form['output_format']

    # Define the Senior AI Research Analyst agent
    researcher = Agent(
        role='Senior AI Research Analyst',
        goal=f'Identify and analyze the most impactful and recent breakthroughs in {topic} from {year}, focusing on their potential real-world applications and implications.',
        backstory="You are a leading analyst at a prestigious technology think tank. Your expertise lies in sifting through vast amounts of information to identify key trends, disruptive technologies, and significant advancements in your field.",
        verbose=True,
        tools=[search_tool],
        llm=LLM(model=f"azure/{os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}")
    )

    # Define the Senior Content Strategist agent
    writer = Agent(
        role='Senior Content Strategist',
        goal=f'Develop and refine compelling narratives from complex research findings on {topic} from {year}, ensuring the content is engaging, informative, and tailored for a {output_format} format.',
        backstory="You are a master storyteller and content strategist, with a proven ability to transform technical research into accessible and captivating content that resonates with a broad audience. Your skill lies in identifying the core message and crafting narratives that inform, inspire, and persuade.",
        verbose=True,
        llm=LLM(model=f"azure/{os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}")
    )

    # Define the research task
    task1 = Task(
        description=f"Conduct thorough research on the most significant advancements in {topic} during {year}. Focus on identifying breakthroughs in areas such as machine learning algorithms, natural language processing, computer vision, and AI ethics. Analyze their potential impact across various industries. Synthesize your findings into a concise, well-structured report.",
        expected_output="A detailed research report presented in bullet points. The report should clearly outline the key breakthroughs and trends identified, discuss their potential real-world applications, and briefly touch upon their implications or challenges. Ensure the report is easy to understand while maintaining technical accuracy.",
        agent=researcher
    )

    # Define the writing task
    task2 = Task(
        description=f"Based on the provided research report on {topic} advancements from {year}, write a compelling and engaging piece for a general audience in a {output_format} format. Translate the technical findings into accessible language, highlighting their relevance and excitement. Aim for a narrative flow that captures the reader's attention.",
        expected_output=f"A full, well-structured piece in {output_format} format of at least 4 paragraphs. The content should be engaging and informative for a general audience, effectively explaining the advancements from the research report. The tone should be enthusiastic and forward-looking.",
        agent=writer
    )

    # Create and run the Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)