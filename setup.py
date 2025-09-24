# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import List

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("src", "llamafactory", "extras", "env.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version, ) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")
        ]
        return lines


def get_console_scripts() -> List[str]:
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    console_scripts.append("bisheng_ft = llmtuner.wrapers.interface:main")
    if os.environ.get("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
    "torch-npu": ["torch-npu==2.5.1", "torchvision==0.20.1", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.16.9"],
    "liger-kernel": ["liger-kernel>=0.5.5"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.24.0", "gptqmodel>=2.0.0"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3,<=0.10.0"],
    "sglang": ["sglang[srt]>=0.4.5", "transformers==4.51.1"],
    "galore": ["galore-torch"],
    "apollo": ["apollo-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "minicpm_v": [
        "soundfile",
        "torchvision",
        "torchaudio",
        "vector_quantize_pytorch",
        "vocos",
        "msgpack",
        "referencing",
        "jsonschema_specifications",
    ],
    "openmind": ["openmind"],
    "swanlab": ["swanlab"],
    "dev": ["pre-commit", "ruff", "pytest", "build"],
}



def main():
    setup(
        name="bisheng-ft",
        version=get_version(),
        author="gulixin",
        author_email="gulixin"
        "@"
        "dataelem.com",
        description="Easy-to-use LLM fine-tuning framework",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=[
            "LLaMA", "BLOOM", "Falcon", "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"
        ],
        license="Apache 2.0 License",
        url="https://github.com/dataelement/bisheng-ft",
        package_dir={"": "src"},
        packages=find_packages("src"),
        package_data={'': ['*.json', '*.csv']},
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
