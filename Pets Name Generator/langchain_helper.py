from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

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


if __name__ == "__main__":
    print(generate_pet_name("cat", "black"))