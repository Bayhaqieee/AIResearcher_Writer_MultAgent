import os
from flask import Flask, render_template, request
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

from crewai.tools import BaseTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from markdown_it import MarkdownIt

load_dotenv()

app = Flask(__name__)

# Initialize the AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_API_KEY"),
    api_version=os.environ.get("AZURE_API_VERSION"),
    model=f"azure/{os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}"
)

# Custom Search Tool
class SearchTool(BaseTool):
    name: str = "Internet Search"
    description: str = "A tool to search the internet for recent and relevant information. Use it to find information on any topic."
    
    def _run(self, search_query: str) -> str:
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

    # 1. Researcher Agent
    researcher = Agent(
        role='Senior Research Analyst',
        goal=f'Diligently research the topic of {topic} for the year {year}, focusing on groundbreaking advancements, key players, and statistical impacts. Synthesize your findings into a structured report.',
        backstory=(
            "You are a renowned Research Analyst with a Ph.D. in Technology Studies, known for your ability to distill complex topics into clear, actionable insights. "
            "Your work is methodical, relying on credible sources to build a comprehensive understanding of the subject."
        ),
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # 2. Writer Agent
    writer = Agent(
        role='Professional Content Strategist',
        goal=f'Using the research report on {topic} from {year}, craft a compelling and engaging piece of content tailored for the {output_format} format. Your writing must be clear, concise, and captivating for a general audience.',
        backstory=(
            "You are a celebrated Content Strategist, famous for your ability to weave intricate research findings into powerful narratives. "
            "You understand how to hook a reader and explain complex ideas in a simple, elegant manner, perfectly adapting your tone for blogs, social media, or academic papers."
        ),
        verbose=True,
        llm=llm
    )

    # 3. Editor Agent
    editor = Agent(
        role='Technical Editor & Fact-Checker',
        goal='Review the written article for technical accuracy, grammatical correctness, and clarity. Ensure the content aligns with the initial research report and is polished to a professional standard.',
        backstory=(
            "You are a meticulous editor from a top-tier tech publication. With a sharp eye for detail, you catch every error, clarify every ambiguity, and verify every fact. "
            "Your job is to ensure that every piece of content that crosses your desk is credible, flawless, and ready for publication."
        ),
        verbose=True,
        llm=llm
    )

    # Task for Researcher
    task1 = Task(
        description=f"Conduct a comprehensive investigation into the latest advancements in '{topic}' during {year}. Identify the top 3-5 key breakthroughs, the leading companies or researchers involved, and any significant statistics or data points. Compile your findings into a structured report with clear headings.",
        expected_output="A detailed, easy-to-read report in Markdown format. The report must contain a summary, followed by sections for each key breakthrough, including names, dates, and verifiable facts or statistics.",
        agent=researcher
    )

    # Task for Writer
    task2 = Task(
        description=f"Transform the research report on '{topic}' into a compelling article for the '{output_format}' format. The article should have a catchy headline, an engaging introduction, and a clear narrative. Translate technical jargon into accessible language without sacrificing accuracy. The final output must be in Markdown.",
        expected_output=f"A well-written article in Markdown format, perfectly suited for a '{output_format}'. It must be at least 4 paragraphs long and include headings and bullet points for readability.",
        agent=writer
    )

    # Task for Editor
    task3 = Task(
        description="Review the draft article. Cross-reference the facts and statistics with the original research report to ensure accuracy. Proofread for any grammatical or spelling errors. Improve sentence structure for better clarity and flow. The final output must be a polished, publication-ready article in Markdown.",
        expected_output=f"The final, polished version of the article as raw Markdown text. Do NOT include any code block formatting like ```markdown. The output should begin directly with the title.",
        agent=editor,
        context=[task1, task2]
    )
    
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    md = MarkdownIt()
    html_result = md.render(result.raw)

    return render_template('results.html', result=html_result)

if __name__ == '__main__':
    app.run(debug=True)