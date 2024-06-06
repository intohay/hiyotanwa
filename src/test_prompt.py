from prompt_toolkit import PromptSession

session = PromptSession()
wrapped_text = "初期値をここに入力"

corrected_text = session.prompt(
    message="> ",
    default=wrapped_text,
)

print("入力が完了しました:", corrected_text)
