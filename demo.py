import gradio as gr

def upload_file(files):
    file_paths = [file.name for file in files]
    diffusion_input = files
    return file_paths, diffusion_input

def demo_text(text):
    return text + " 가 LLM으로 전달 됨."

title_markdown = """ 
Demo
"""
notice_markdown = """ 
사진을 업로드해 프로필을 완성하세요
"""

with gr.Blocks() as demo:

    with gr.Row():
        gr.Markdown(title_markdown)

    with gr.Accordion(label="사용법",open=False):
        gr.Markdown(notice_markdown)

    with gr.Row():
        file_output = gr.Gallery(height=400)
        diffusion_input = gr.Textbox(value="",label="diffusion_input")
        
    with gr.Row():
        upload_button = gr.UploadButton("반려동물 사진을 업로드하세요", file_types=["image"], file_count="multiple")
        upload_button.upload(upload_file, upload_button, [file_output, diffusion_input])
        

    with gr.Row():    
        msg = gr.Textbox(label="프롬프트를 입력하세요.")
        send_button = gr.Button("입력",scale=0)
        llm_input = gr.Textbox(value="",label="prompt")
        msg.submit(demo_text,msg,llm_input)
        send_button.click(demo_text,msg,llm_input)
        

if __name__ == "__main__":
    demo.launch()