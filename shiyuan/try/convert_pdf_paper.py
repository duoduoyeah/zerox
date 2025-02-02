from pyzerox import zerox
import os
import json
import asyncio

### Model Setup (Use only Vision Models) Refer: https://docs.litellm.ai/docs/providers ###

## placeholder for additional model kwargs which might be required for some models
kwargs = {}

## system prompt to use for the vision model
custom_system_prompt = None

# to override
# custom_system_prompt = "For the below pdf page, do something..something..." ## example

###################### Example for OpenAI ######################
model = "gpt-4o"  ## openai model
os.environ["OPENAI_API_KEY"] = "sk-5s8pX4yqCaJi3F4aXMIpsLftSTqVlmuMARucTxnsuewvu13i"
os.environ["OPENAI_API_BASE"] = "http://chatapi.littlewheat.com/v1"

###################### For other providers refer: https://docs.litellm.ai/docs/providers ######################


# Define main async entrypoint
async def main():
    folder = "/workspaces/simulation/paper"
    pdf_name = "Analyzing_CUDA_workloads_using_a_detailed_GPU_simulator.pdf"
    file_path = os.path.join(folder, pdf_name)  ## local filepath and file URL supported

    ## process only some pages or all
    ## None for all, but could be int or list(int) page numbers (1 indexed)
    select_pages = list(range(1, 12))  # Process pages 1 through 11

    output_dir = "./output_test"  ## directory to save the consolidated markdown file
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )
    return result


# run the main function:
result = asyncio.run(main())

# print markdown result
print(result)
