from setuptools import setup, find_packages

# 读取 requirements.txt
def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="nexusagent",
    version="2.0.0",
    description="NexusAgent - 多Agent协作 · RAG知识增强 · 安全可控的下一代透明智能体",
    author="NexusAgent Team",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=parse_requirements('requirements.txt'),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "nexusagent=entry.cli:main",
        ],
    },
)
