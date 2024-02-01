from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type'],
        template = f"I have a {animal_type} pet and I want a cool name for it, it is of color {pet_color}. Suggest me five cool names for my pet."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key = "pet_names")
    response = name_chain({'animal_type': animal_type})
    # name = llm("I have a dog pet and I want a cool name for it. Suggest me five cool names for my pet.")

    return response

def langchain_agent():
    llm = OpenAI(temperature = 0.5)

    tools = load_tools(["wikipedia", "llm-math"], llm = llm)

    agent = initialize_agent(
        tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True
    )

    result = agent.run(
        "What is the average age of a dog? Multiply the age by 3"
    )

    print(result)

if __name__ == "__main__":
    langchain_agent()
    #  print(generate_pet_name("cat", "black"))