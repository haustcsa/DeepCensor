

class LanguageTool():
    def __init__(self) -> None:

        import language_tool_python
        self.language_tool = language_tool_python.LanguageTool('en-US')

    def after_attack(self, adversarial_sample):
        if adversarial_sample is not None:
            return len(self.language_tool.check(adversarial_sample))/len(adversarial_sample.split())

