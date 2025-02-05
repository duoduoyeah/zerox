from pyzerox import zerox
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
# placeholder for additional model kwargs
kwargs = {}

# system prompt to use for the vision model
custom_system_prompt = None

# to override
# custom_system_prompt = "For the below pdf page, do something..something..."

model = "gpt-4o"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")


# Define main async entrypoint
async def main(input_folder, input_pdf_name, output_dir, start, end):
    file_path = os.path.join(input_folder, input_pdf_name)

    # process only some pages or all
    # None for all, but could be int or list(int) page numbers (1 indexed)
    select_pages = list(range(start, end + 1))

    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )
    return result


if __name__ == "__main__":
    cache = {"input_folder": "/workspaces/simulation/paper",
             "pdf_name": "Path Forward Beyond Simulators: Fast and Accurate GPU.pdf"}
    input_folder = "/workspaces/other_papers"
    pdf_name = "xie.pdf"
    output_dir = "./shiyuan/simulation_papers"
    start_page = 1
    end_page = 2
    asyncio.run(
        main(
            input_folder,
            pdf_name,
            output_dir,
            start_page,
            end_page,
        )
    )
