from setuptools import setup, find_packages

setup(
    name="opengait",                # 项目名称
    version="0.1.0",                         # 项目版本
    author="Your Name",                      # 作者名称
    author_email="your_email@example.com",   # 作者邮箱
    description="Setup for OpenGait project with required dependencies.",  # 项目描述
    long_description=open("README.md").read(),  # 从 README.md 获取详细描述
    long_description_content_type="text/markdown",  # README 文件格式
    # url="https://github.com/your_username/your_repo",  # 项目主页或仓库地址
    # packages=find_packages(),                # 自动发现包含的包
    packages=["opengait"],  # 需要包含的 Python 包
    install_requires=[                       # 项目的依赖包列表
        "opencv-python",
        "pyyaml",
        "tensorboard",
        "tqdm",
        "py7zr",
        "kornia",
        "einops",
        "torch",
        "torchvision",
        "scikit-learn",
        "imageio",
        "matplotlib",
    ],
    classifiers=[                            # 分类器，用于描述项目
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",                 # 支持的 Python 版本
    entry_points={                           # 定义命令行脚本入口
        "console_scripts": [
            "your_command=your_module:main_function",
        ],
    },
)