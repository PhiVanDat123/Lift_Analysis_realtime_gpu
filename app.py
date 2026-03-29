"""
Gradio UI entry point.
All business logic lives in core/.
"""
import gradio as gr

from core.processor import process_video_realtime

with gr.Blocks(title="AI Exercise Pose Feedback") as demo:
    gr.Markdown("# AI Exercise Pose Feedback")
    gr.Markdown("Chọn bài tập, upload video và nhận phản hồi tư thế theo thời gian thực.")

    exercise_radio = gr.Radio(
        choices=["Bench Press", "Back Squat", "Deadlift"],
        value="Back Squat",
        label="Bài tập",
    )

    with gr.Row():
        video_input  = gr.Video(label="Video đầu vào", sources=["upload"], height=480)
        video_output = gr.Image(label="Phân tích real-time", height=480)

    video_feedback = gr.Textbox(
        label="Phản hồi tư thế",
        lines=10,
        interactive=False,
        placeholder="Phản hồi sẽ hiển thị ở đây sau mỗi rep...",
    )
    analyze_btn = gr.Button("Phân tích video", variant="primary")

    analyze_btn.click(
        fn=process_video_realtime,
        inputs=[video_input, exercise_radio],
        outputs=[video_output, video_feedback],
    )

if __name__ == "__main__":
    demo.launch()
