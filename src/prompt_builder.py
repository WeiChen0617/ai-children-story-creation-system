"""
Prompt Builder Module
AI增强儿童故事创作系统 - Prompt构造模块

功能：
- 支持结构化关键词（角色、主题、语气等）生成高控制性Prompt
- 支持模板式、结构式、问句式三种Prompt风格
- 便于后续集成到主系统
"""

from typing import Dict, Optional

class PromptBuilder:
    """
    Prompt构造主类
    支持三种风格的Prompt生成：模板式、结构式、问句式
    """
    def __init__(self, age: int = 6):
        self.age = age

    def build_template_prompt(self, character: str, theme: str, word_limit: int = 300) -> str:
        """
        模板式Prompt生成
        :param character: 主角
        :param theme: 教育主题
        :param word_limit: 字数上限
        :return: Prompt字符串
        """
        return (
            f"请为{self.age}岁的儿童写一个关于\"{theme}\"的原创故事。"
            f"主角是一只{character}，语言简洁，积极正面，字数不超过{word_limit}词。"
        )

    def build_structured_prompt(self, character: str, theme: str) -> str:
        """
        结构式Prompt生成
        :param character: 主角
        :param theme: 教育主题
        :return: Prompt字符串
        """
        return (
            f"请分三段讲述一个儿童故事：1）介绍主人公（如{character}）；"
            f"2）经历挑战；3）得到成长。主题为\"{theme}\"，适合{self.age}岁儿童。"
        )

    def build_question_prompt(self, theme: str) -> str:
        """
        问句式Prompt生成
        :param theme: 教育主题
        :return: Prompt字符串
        """
        return (
            f"什么是{theme}？请通过一则适合{self.age}岁儿童的故事解释这个问题。"
            f"语言应通俗、有趣，表达积极价值观。"
        )

# 示例用法
if __name__ == "__main__":
    builder = PromptBuilder(age=7)
    print("[模板式]", builder.build_template_prompt("小狐狸", "合作"))
    print("[结构式]", builder.build_structured_prompt("小熊", "诚实"))
    print("[问句式]", builder.build_question_prompt("环保")) 