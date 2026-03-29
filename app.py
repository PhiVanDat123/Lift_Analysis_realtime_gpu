"""
Gradio UI entry point.
All business logic lives in core/.
"""
import gradio as gr

from core.processor import process_video_streaming

_css = """
#zen-output { border: none !important; padding: 0 !important; margin: 0 !important; box-shadow: none !important; }
#zen-output.generating { border: none !important; animation: none !important; }
#zen-output > .wrap { border-top: none !important; padding-top: 0 !important; }
"""

with gr.Blocks(title="AI Exercise Pose Feedback", css=_css) as demo:
    gr.Markdown("# AI Exercise Pose Feedback")
    gr.Markdown("Chọn bài tập, upload video và nhận phản hồi tư thế theo thời gian thực.")

    exercise_radio = gr.Radio(
        choices=["Bench Press", "Back Squat", "Deadlift"],
        value="Back Squat",
        label="Bài tập",
    )

    with gr.Row():
        video_input = gr.Video(label="Video đầu vào", sources=["upload"], height=480)
        video_display = gr.HTML(
            '<p style="color:#888;font-size:14px;padding:16px;">Upload video và nhấn Phân tích để bắt đầu.</p>',
            label="Phân tích real-time",
            elem_id="zen-output",
        )

    video_feedback = gr.Textbox(
        label="Phản hồi tư thế",
        lines=10,
        interactive=False,
        placeholder="Phản hồi sẽ hiển thị ở đây sau mỗi rep...",
    )
    analyze_btn = gr.Button("Phân tích video", variant="primary")

    analyze_btn.click(
        fn=process_video_streaming,
        inputs=[video_input, exercise_radio],
        outputs=[video_display, video_feedback],
    )

if __name__ == "__main__":
    demo.launch()
