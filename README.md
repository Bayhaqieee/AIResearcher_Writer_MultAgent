# AI Researcher Writer Multi-Agent

Welcome to My Project! This project focuses on creating a multi-agent system for AI research and content creation using **CrewAI** and **LangChain**.

## Project Status

🚧 **Status**: `Completed`

## Project Target

  - **Automated Research and Writing**
      - Able to conduct research on a given topic using AI agents.
      - The input will be a topic for research.
      - The output will be a detailed research report and a blog post based on the report.

-----

## Technologies

  - **Python**: General-purpose programming language used for creating the multi-agent system.
  - **CrewAI**: A framework for orchestrating role-playing, autonomous AI agents.
  - **CrewAI Tools**: Tools for extending the capabilities of CrewAI agents.
  - **LangChain OpenAI**: Integration for using OpenAI's language models, in this case, Azure's chat models.
  - **SerperDevTool**: A tool for allowing the AI agents to perform Google searches.
  - **python-dotenv**: For managing environment variables.

-----

## Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/Bayhaqieee/AIResearcher_Writer_MultAgent.git
    ```

2.  Install the required dependencies:

    ```bash
    pip install crewai crewai_tools langchain-openai python-dotenv
    ```

3.  Set up your environment variables. You will need to create a `.env` file or set them in your environment with the following:

      - `AZURE_OPENAI_API_KEY`
      - `AZURE_OPENAI_ENDPOINT`
      - `OPENAI_API_VERSION`
      - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
      - `SERPER_API_KEY`

4.  Run the Notebooks, each of them is already labeled.
